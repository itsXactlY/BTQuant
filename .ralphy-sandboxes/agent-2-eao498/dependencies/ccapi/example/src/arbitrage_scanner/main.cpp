#include <csignal>
#include <iostream>
#include <fstream>
#include <sstream>
#include "arbitrage_scanner.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

static ArbitrageScanner* g_scanner = nullptr;
static std::ofstream g_csv_file;

void signalHandler(int) {
    std::cout << "\nSIGINT received, stopping scanner...\n";
    if (g_scanner) {
        g_scanner->stop();
    }
}

ArbitrageScanner::Config loadConfig(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open config file: " + path);
    }
    json j;
    in >> j;

    ArbitrageScanner::Config cfg;
    
    cfg.exchanges = j.at("exchanges").get<std::vector<std::string>>();
    cfg.symbols = j.at("symbols").get<std::vector<std::string>>();
    cfg.market_type = j.value("market_type", std::string("spot"));
    cfg.min_profit_bps = j.value("min_profit_bps", 30.0);
    cfg.max_age_us = j.value("max_age_ms", 100) * 1000;
    cfg.buffer_size = j.value("buffer_size", 1000);
    cfg.flush_interval_ms = j.value("flush_interval_ms", 1000);
    cfg.stats_report_interval_s = j.value("stats_report_interval_s", 10);

    return cfg;
}

void initializeCSVFile() {
    g_csv_file.open("arbitrage_opportunities.csv", std::ios::out | std::ios::app);
    if (!g_csv_file.is_open()) {
        throw std::runtime_error("Cannot open CSV file for writing");
    }
    
    g_csv_file.seekp(0, std::ios::end);
    if (g_csv_file.tellp() == 0) {
        g_csv_file << "Timestamp,Buy Exchange,Sell Exchange,Symbol,Market Type,Buy Price,Sell Price,Profit (BPS),Max Volume,Latency (us),Buy Fee (BPS),Sell Fee (BPS),Net Profit (BPS)\n";
    }
}

void logOpportunityToCSV(const Arbitrage::Opportunity& opp) {
    double buy_fee_bps = Arbitrage::get_exchange_fee_bps(opp.buy_exchange);
    double sell_fee_bps = Arbitrage::get_exchange_fee_bps(opp.sell_exchange);
    double net_profit_bps = opp.profit_bps - buy_fee_bps - sell_fee_bps;
    
    g_csv_file << opp.toTimestamp() << ","
               << opp.buy_exchange << ","
               << opp.sell_exchange << ","
               << opp.symbol << ","
               << opp.market_type << ","
               << opp.buy_price << ","
               << opp.sell_price << ","
               << opp.profit_bps << ","
               << opp.max_volume << ","
               << opp.latency_us << ","
               << buy_fee_bps << ","
               << sell_fee_bps << ","
               << net_profit_bps << "\n";
    
    g_csv_file.flush();
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signalHandler);

    std::string cfg_path = (argc > 1) ? argv[1] : "arbitrage_config.json";
    auto cfg = loadConfig(cfg_path);

    ArbitrageScanner scanner(cfg);
    g_scanner = &scanner;
    initializeCSVFile();
    
    scanner.setOpportunityCallback([](const Arbitrage::Opportunity& opp) {
        logOpportunityToCSV(opp);
    });

    scanner.start();
    scanner.waitForShutdown();
    
    if (g_csv_file.is_open()) {
        g_csv_file.close();
    }

    std::cout << "Shutdown complete.\n";
    return 0;
}
