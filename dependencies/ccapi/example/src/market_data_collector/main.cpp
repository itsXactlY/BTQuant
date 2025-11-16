#include <csignal>
#include <iostream>

#include "market_data_collector.h"

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

int main(int argc, char** argv) {
    std::signal(SIGINT, signalHandler);

    MarketDataCollector::Config cfg;

    cfg.db_connection_string = createConnectionString(
        "localhost",
        "BTQ_MarketData",
        "SA",
        "q?}33YIToo:H%xue$Kr*");

    cfg.timeframes = {"1m"}; // , "5m", "15m", "1h", "4h"

    cfg.exchanges = {
        {
            "binance",
            {"BTCUSDT", "ETHUSDT", "SOLUSDT"},
            {"TRADE", "MARKET_DEPTH"},
            "spot"
        },
        {
            "okx",
            {"BTC-USDT", "ETH-USDT", "SOL-USDT"},
            {"TRADE", "MARKET_DEPTH"},
            "spot"
        }
    };

    cfg.bulk_insert_batch_size   = 500;
    cfg.trade_buffer_size        = 1000;
    cfg.candle_buffer_size       = 200;
    cfg.orderbook_buffer_size    = 100;
    cfg.flush_interval_ms        = 1000;
    cfg.stats_report_interval_s  = 10;

    MarketDataCollector collector(cfg);
    g_collector = &collector;

    collector.start();
    collector.waitForShutdown();

    std::cout << "Shutdown complete.\n";
    return 0;
}
