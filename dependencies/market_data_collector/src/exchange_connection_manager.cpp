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
                ccapi::Subscription s(cfg.exchange_name,
                                      sym,
                                      ch);
                s.addCorrelationId(
                    cfg.exchange_name + ":" + sym + ":" + cfg.market_type);
                subs.push_back(std::move(s));
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
