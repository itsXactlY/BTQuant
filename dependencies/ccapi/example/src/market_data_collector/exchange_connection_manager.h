#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ccapi_cpp/ccapi_session.h"
#include "market_data_processor.h"

class ExchangeConnectionManager {
public:
    struct ExchangeConfig {
        std::string exchange_name;
        std::vector<std::string> symbols;
        std::vector<std::string> channels;  // "TRADE", "MARKET_DEPTH"
        std::string market_type;            // "spot", "perpetual"
    };

    explicit ExchangeConnectionManager(
        std::shared_ptr<MarketDataProcessor> processor);

    void subscribe(const std::vector<ExchangeConfig>& cfgs);
    void start();
    void stop();

    bool isRunning() const { return running_; }

private:
    std::unique_ptr<ccapi::SessionOptions> session_options_;
    std::unique_ptr<ccapi::SessionConfigs> session_configs_;
    std::unique_ptr<ccapi::Session> session_;
    std::shared_ptr<MarketDataProcessor> processor_;

    std::atomic<bool> running_{false};
};
