#include <csignal>
#include <cstdlib>
#include <iostream>
#include "market_data_collector.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

static MarketDataCollector* g_collector = nullptr;

std::string createConnectionString(const std::string& server,
                                   const std::string& database,
                                   const std::string& user,
                                   const std::string& pwd) {
    return
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=" + server + ";"
        "DATABASE=" + database + ";"
        "UID=" + user + ";"
        "PWD=" + pwd + ";"
        "TrustServerCertificate=yes;"
        "MARS_Connection=yes;"
        "Connection Timeout=30;"
        "Command Timeout=60;";
}

void signalHandler(int) {
    std::cout << "\nSIGINT received, stopping collector...\n";
    if (g_collector) {
        g_collector->stop();
    }
}

MarketDataCollector::Config loadConfig(const std::string& path) {
    std::ifstream in(path);
    json j;

    if (!in) {
        // If config file doesn't exist, use default values
        std::cout << "Config file not found: " << path << ", using default configuration." << std::endl;
        
        // Create default configuration
        MarketDataCollector::Config cfg;
        
        // Default database connection (will be used if enable_mssql is true)
        cfg.db_connection_string = createConnectionString(
            "localhost",
            "market_data",
            "sa",
            "your_password");
        
        // Default timeframes
        cfg.timeframes = {"1m", "5m", "15m", "1h"};
        
        // Default exchange configuration
        ExchangeConnectionManager::ExchangeConfig ec;
        ec.exchange_name = "binance";
        ec.symbols = {"BTC-USDT", "ETH-USDT"};
        ec.channels = {"TRADE", "MARKET_DEPTH"};
        ec.market_type = "spot";
        cfg.exchanges.push_back(std::move(ec));
        
        // Default buffer sizes and intervals
        cfg.trade_buffer_size = 1000;
        cfg.candle_buffer_size = 200;
        cfg.orderbook_buffer_size = 100;
        cfg.flush_interval_ms = 1000;
        cfg.stats_report_interval_s = 10;
        
        // Default toggle settings - MS SQL enabled, hotswap disabled
        cfg.enable_mssql = true;
        cfg.enable_exclusive_hotspine = false;
        
        return cfg;
    }
    
    // Load configuration from file
    in >> j;

    MarketDataCollector::Config cfg;

    auto db = j.at("db");
    // Password is NEVER stored in the repo config. Resolution order:
    //   1. $BTQ_DB_PASSWORD   (export it, or source the secrets file)
    //   2. the "password" key, which is expected to be EMPTY in git
    // Keeping the key present means this file does not need changing when
    // the resolution order grows.
    std::string db_password = db.value("password", "");
    if (const char* env_pw = std::getenv("BTQ_DB_PASSWORD");
        env_pw && *env_pw) {
        db_password = env_pw;
    }
    if (db_password.empty()) {
        std::cerr << "FATAL: no DB password. Export BTQ_DB_PASSWORD, e.g.\n"
                  << "  eval \"$(python3 -c \"import btq_secrets;"
                  << " print('export BTQ_DB_PASSWORD=' + btq_secrets.get('password'))\")\"\n";
        return 1;
    }
    cfg.db_connection_string = createConnectionString(
        db.at("server").get<std::string>(),
        db.at("database").get<std::string>(),
        db.at("user").get<std::string>(),
        db_password);

    cfg.timeframes = j.at("timeframes").get<std::vector<std::string>>();

    for (const auto& ex : j.at("exchanges")) {
        ExchangeConnectionManager::ExchangeConfig ec;

        ec.exchange_name = ex.at("name").get<std::string>();
        ec.symbols       = ex.at("symbols")
                             .get<std::vector<std::string>>();
        ec.channels      = ex.value(
            "channels",
            std::vector<std::string>{"TRADE", "MARKET_DEPTH"}
        );
        ec.market_type   = ex.value(
            "market_type",
            std::string("spot")
        );

        cfg.exchanges.push_back(std::move(ec));
    }

    cfg.trade_buffer_size        = j.value("trade_buffer_size", 500);
    cfg.candle_buffer_size       = j.value("candle_buffer_size", 200);
    cfg.orderbook_buffer_size    = j.value("orderbook_buffer_size", 100);
    cfg.flush_interval_ms        = j.value("flush_interval_ms", 1000);
    cfg.stats_report_interval_s  = j.value("stats_report_interval_s", 10);

    // New configuration options
    cfg.enable_mssql              = j.value("enable_mssql", true);
    cfg.enable_exclusive_hotspine = j.value("enable_exclusive_hotspine", false);

    return cfg;
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signalHandler);

    std::string cfg_path = (argc > 1) ? argv[1] : "config.json";
    auto cfg = loadConfig(cfg_path);

    MarketDataCollector collector(cfg);
    g_collector = &collector;

    collector.start();
    collector.waitForShutdown();

    std::cout << "Shutdown complete.\n";
    return 0;
}

