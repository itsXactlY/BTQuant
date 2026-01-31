#include <csignal>
#include <iostream>
#include "market_data_collector.h"

static MarketDataCollector* g_collector = nullptr;

// Forward declaration of the loadConfig function from config_loader.cpp
MarketDataCollector::Config loadConfig(const std::string& path);

void signalHandler(int) {
    std::cout << "\nSIGINT received, stopping collector...\n";
    if (g_collector) {
        g_collector->stop();
    }
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

