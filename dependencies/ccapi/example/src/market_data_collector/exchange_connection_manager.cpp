#include "exchange_connection_manager.h"

#include <iostream>

ExchangeConnectionManager::ExchangeConnectionManager(
    std::shared_ptr<MarketDataProcessor> processor)
    : session_options_(std::make_unique<ccapi::SessionOptions>()),
      session_configs_(std::make_unique<ccapi::SessionConfigs>()),
      processor_(std::move(processor)) {

    session_ = std::make_unique<ccapi::Session>(
        *session_options_, *session_configs_, processor_.get());
}

void ExchangeConnectionManager::subscribe(
    const std::vector<ExchangeConfig>& cfgs) {

    std::vector<ccapi::Subscription> subs;

    for (const auto& cfg : cfgs) {
        for (const auto& sym : cfg.symbols) {
            for (const auto& ch : cfg.channels) {
                // correlation id encodes exchange:symbol:market_type
                std::string cid =
                    cfg.exchange_name + ":" + sym + ":" + cfg.market_type;

                std::string options;
                if (ch == "MARKET_DEPTH") {
                    // 🔹 Ask ccapi for 20 levels per side, full snapshot
                    //    whenever any of those levels change.
                    options = "MARKET_DEPTH_MAX=20";
                    // (No MARKET_DEPTH_RETURN_UPDATE=1 => snapshot mode)
                } else {
                    options.clear();
                }

                ccapi::Subscription sub(
                    cfg.exchange_name,  // exchange
                    sym,                // instrument
                    ch,                 // "TRADE" / "MARKET_DEPTH"
                    options,
                    cid                 // correlationId
                );

                subs.push_back(std::move(sub));
            }
        }
    }

    session_->subscribe(subs);
    std::cout << "Subscribed to " << subs.size() << " streams\n";
}

void ExchangeConnectionManager::start() {
    running_ = true;
    // ccapi manages its own threads internally.
}

void ExchangeConnectionManager::stop() {
    running_ = false;
    if (session_) {
        session_->stop();
    }
}
