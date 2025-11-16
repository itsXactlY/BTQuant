#include <csignal>
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
    if (!in) {
        throw std::runtime_error("Cannot open config file: " + path);
    }
    json j;
    in >> j;

    MarketDataCollector::Config cfg;

    auto db = j.at("db");
    cfg.db_connection_string = createConnectionString(
        db.at("server").get<std::string>(),
        db.at("database").get<std::string>(),
        db.at("user").get<std::string>(),
        db.at("password").get<std::string>());

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

