#include "arbitrage_scanner.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <algorithm>

// convert normalized symbol to exchange-specific format
std::string ArbitrageScanner::getExchangeSymbol(const std::string& exchange, const std::string& normalized_symbol) {
    // normalized format: BTC-USDT, ETH-USDT, SOL-USDT
    
    if (exchange == "binance" || exchange == "bybit") {
        // remove dash: BTC-USDT -> BTCUSDT
        std::string result = normalized_symbol;
        result.erase(std::remove(result.begin(), result.end(), '-'), result.end());
        return result;
    }
    else if (exchange == "okx") {
        // keep dash: BTC-USDT
        return normalized_symbol;
    }
    else if (exchange == "coinbase") {
        // change USDT to USD: BTC-USDT -> BTC-USD
        size_t pos = normalized_symbol.find("-USDT");
        if (pos != std::string::npos) {
            return normalized_symbol.substr(0, pos) + "-USD";
        }
        return normalized_symbol;
    }
    else if (exchange == "kraken") {
        // use XBT instead of BTC, slash separator: BTC-USDT -> XBT/USD
        std::string result = normalized_symbol;
        
        // replace BTC with XBT
        size_t btc_pos = result.find("BTC");
        if (btc_pos != std::string::npos) {
            result.replace(btc_pos, 3, "XBT");
        }
        
        // replace USDT with USD
        size_t usdt_pos = result.find("USDT");
        if (usdt_pos != std::string::npos) {
            result.replace(usdt_pos, 4, "USD");
        }
        
        // replace dash with slash
        std::replace(result.begin(), result.end(), '-', '/');
        
        return result;
    }
    
    return normalized_symbol;
}

// convert exchange-specific symbol back to normalized format
std::string ArbitrageScanner::normalizeSymbol(const std::string& exchange_symbol, const std::string& exchange) {
    if (exchange == "binance" || exchange == "bybit") {
        // BTCUSDT -> BTC-USDT
        // find common quote currencies
        const std::vector<std::string> quotes = {"USDT", "USDC", "USD", "BTC", "ETH"};
        for (const auto& quote : quotes) {
            size_t pos = exchange_symbol.rfind(quote);
            if (pos != std::string::npos && pos + quote.length() == exchange_symbol.length()) {
                return exchange_symbol.substr(0, pos) + "-" + quote;
            }
        }
    }
    else if (exchange == "okx") {
        // already normalized: BTC-USDT
        return exchange_symbol;
    }
    else if (exchange == "coinbase") {
        // BTC-USD -> BTC-USDT (normalize to USDT)
        std::string result = exchange_symbol;
        size_t pos = result.find("-USD");
        if (pos != std::string::npos && result.find("-USDT") == std::string::npos && 
            result.find("-USDC") == std::string::npos) {
            result.replace(pos, 4, "-USDT");
        }
        return result;
    }
    else if (exchange == "kraken") {
        // XBT/USD -> BTC-USDT
        std::string result = exchange_symbol;
        
        // replace slash with dash
        std::replace(result.begin(), result.end(), '/', '-');
        
        // replace XBT with BTC
        size_t xbt_pos = result.find("XBT");
        if (xbt_pos != std::string::npos) {
            result.replace(xbt_pos, 3, "BTC");
        }
        
        // replace USD with USDT (if not USDC)
        size_t usd_pos = result.find("-USD");
        if (usd_pos != std::string::npos && result.find("-USDT") == std::string::npos && 
            result.find("-USDC") == std::string::npos) {
            result.replace(usd_pos, 4, "-USDT");
        }
        
        return result;
    }
    
    return exchange_symbol;
}

ArbitrageScanner::ArbitrageScanner(const Config& cfg)
    : config_(cfg) {
    
    // build symbol mapping
    for (const auto& exchange : config_.exchanges) {
        for (const auto& norm_symbol : config_.symbols) {
            std::string ex_symbol = getExchangeSymbol(exchange, norm_symbol);
            symbol_map_[exchange][norm_symbol] = ex_symbol;
            
            std::cout << "Symbol mapping: " << exchange << " " << norm_symbol 
                      << " -> " << ex_symbol << "\n";
        }
    }
    
    order_books_.reserve(config_.exchanges.size() * config_.symbols.size());
    
    for (const auto& exchange : config_.exchanges) {
        for (const auto& symbol : config_.symbols) {
            order_books_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(OrderBookKey{exchange, symbol}),
                std::forward_as_tuple()
            );
        }
        exchange_update_counts_[exchange] = 0;
    }
    
    for (const auto& symbol : config_.symbols) {
        symbol_update_counts_[symbol] = 0;
    }
    
    std::cout << "\nArbitrageScanner initialized:\n";
    std::cout << "  Exchanges: " << config_.exchanges.size() << "\n";
    std::cout << "  Symbols: " << config_.symbols.size() << "\n";
    std::cout << "  Order books: " << order_books_.size() << "\n";
    std::cout << "  Min profit: " << config_.min_profit_bps << " bps\n";
    std::cout << "  Max age: " << config_.max_age_us / 1000 << " ms\n\n";
}

void ArbitrageScanner::start() {
    if (running_.exchange(true)) {
        std::cerr << "Scanner already running\n";
        return;
    }
    
    ccapi::SessionOptions options;
    ccapi::SessionConfigs configs;
    
    options.maxEventQueueSize = 100000;
    
    session_ = std::make_unique<ccapi::Session>(options, configs, this);
    
    std::vector<ccapi::Subscription> subscriptions;
    subscriptions.reserve(order_books_.size());
    
    std::cout << "Creating subscriptions...\n";
    for (const auto& exchange : config_.exchanges) {
        for (const auto& norm_symbol : config_.symbols) {
            std::string ex_symbol = symbol_map_[exchange][norm_symbol];
            std::string correlation_id = exchange + "|" + norm_symbol;  // Use normalized symbol in correlation
            
            ccapi::Subscription sub(exchange, ex_symbol, "MARKET_DEPTH", "", correlation_id);
            subscriptions.push_back(std::move(sub));
            
            std::cout << "  " << exchange << "/" << ex_symbol << " -> " << correlation_id << "\n";
        }
    }
    
    std::cout << "\nSubscribing to " << subscriptions.size() << " streams...\n";
    session_->subscribe(subscriptions);
    std::cout << "Subscribed! Waiting for data...\n\n";
    
    stats_thread_ = std::thread([this]() { runStatsReporter(); });
}

void ArbitrageScanner::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    
    if (session_) {
        session_.reset();
    }
    
    if (stats_thread_.joinable()) {
        stats_thread_.join();
    }
    
    std::cout << "Scanner stopped\n";
}

void ArbitrageScanner::waitForShutdown() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void ArbitrageScanner::processEvent(const ccapi::Event& event, ccapi::Session*) {
    if (event.getType() == ccapi::Event::Type::SUBSCRIPTION_DATA) {
        subscription_data_events_.fetch_add(1, std::memory_order_relaxed);
    } else {
        other_events_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    
    const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    
    for (const auto& msg : event.getMessageList()) {
        const auto& correlation_ids = msg.getCorrelationIdList();
        if (correlation_ids.empty()) {
            continue;
        }
        
        const std::string& correlation_id = correlation_ids[0];
        
        size_t pos = correlation_id.find('|');
        if (pos == std::string::npos) {
            continue;
        }
        
        std::string exchange = correlation_id.substr(0, pos);
        std::string instrument = correlation_id.substr(pos + 1);  // Already normalized
        
        static std::atomic<int> debug_count{0};
        if (debug_count.fetch_add(1) < 10) {
            std::cout << "DEBUG: " << exchange << "/" << instrument << " received\n";
        }
        
        for (const auto& elem : msg.getElementList()) {
            processMarketDepth(exchange, instrument, elem, now_us);
        }
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            exchange_update_counts_[exchange]++;
            symbol_update_counts_[instrument]++;
        }
    }
    
    updates_processed_.fetch_add(1, std::memory_order_relaxed);
}

void ArbitrageScanner::processMarketDepth(
    const std::string& exchange,
    const std::string& symbol,
    const ccapi::Element& element,
    int64_t timestamp_us) {
    
    OrderBookKey key{exchange, symbol};
    auto it = order_books_.find(key);
    if (it == order_books_.end()) {
        return;
    }
    
    OrderBook& book = it->second;
    
    const auto bid_price = element.getValue("BID_PRICE");
    const auto bid_size = element.getValue("BID_SIZE");
    const auto ask_price = element.getValue("ASK_PRICE");
    const auto ask_size = element.getValue("ASK_SIZE");
    
    bool updated = false;
    
    if (!bid_price.empty() && !bid_size.empty()) {
        book.bid.update(std::stod(bid_price), std::stod(bid_size), timestamp_us);
        updated = true;
    }
    
    if (!ask_price.empty() && !ask_size.empty()) {
        book.ask.update(std::stod(ask_price), std::stod(ask_size), timestamp_us);
        updated = true;
    }
    
    if (updated) {
        book.update_count.fetch_add(1, std::memory_order_relaxed);
        book.last_update_us.store(timestamp_us, std::memory_order_relaxed);
        scanArbitrage(symbol, timestamp_us);
    }
}

void ArbitrageScanner::scanArbitrage(const std::string& symbol, int64_t now_us) {
    for (size_t i = 0; i < config_.exchanges.size(); ++i) {
        for (size_t j = i + 1; j < config_.exchanges.size(); ++j) {
            const auto& buy_exchange = config_.exchanges[i];
            const auto& sell_exchange = config_.exchanges[j];
            
            OrderBookKey buy_key{buy_exchange, symbol};
            OrderBookKey sell_key{sell_exchange, symbol};
            
            auto buy_it = order_books_.find(buy_key);
            auto sell_it = order_books_.find(sell_key);
            
            if (buy_it == order_books_.end() || sell_it == order_books_.end()) {
                continue;
            }
            
            auto buy_ask = buy_it->second.ask.load();
            auto sell_bid = sell_it->second.bid.load();
            
            checkArbitrageDirection(buy_exchange, sell_exchange, symbol,
                                     buy_ask, sell_bid, now_us);
            
            auto sell_ask = sell_it->second.ask.load();
            auto buy_bid = buy_it->second.bid.load();
            
            checkArbitrageDirection(sell_exchange, buy_exchange, symbol,
                                     sell_ask, buy_bid, now_us);
        }
    }
}

void ArbitrageScanner::checkArbitrageDirection(
    const std::string& buy_exchange,
    const std::string& sell_exchange,
    const std::string& symbol,
    const Arbitrage::OrderBookLevel::Snapshot& buy_ask,
    const Arbitrage::OrderBookLevel::Snapshot& sell_bid,
    int64_t now_us) {
    
    int64_t buy_age_us = now_us - buy_ask.timestamp_us;
    int64_t sell_age_us = now_us - sell_bid.timestamp_us;
    
    if (buy_age_us > config_.max_age_us || sell_age_us > config_.max_age_us) {
        stale_data_rejections_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    
    if (buy_ask.price <= 0.0 || sell_bid.price <= 0.0 ||
        buy_ask.volume <= 0.0 || sell_bid.volume <= 0.0) {
        zero_price_rejections_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    
    const auto buy_fee_bps = Arbitrage::get_exchange_fee_bps(buy_exchange);
    const auto sell_fee_bps = Arbitrage::get_exchange_fee_bps(sell_exchange);
    
    const auto buy_cost = buy_ask.price * (1.0 + buy_fee_bps / 10000.0);
    const auto sell_revenue = sell_bid.price * (1.0 - sell_fee_bps / 10000.0);
    
    const auto profit_bps = ((sell_revenue - buy_cost) / buy_cost) * 10000.0;
    
    double current_best = best_spread_bps_.load();
    if (profit_bps > current_best && best_spread_bps_.compare_exchange_strong(current_best, profit_bps)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        best_spread_info_ = symbol + " " + buy_exchange + "->" + sell_exchange + 
                           " " + std::to_string(profit_bps) + "bps";
    }
    
    if (profit_bps < config_.min_profit_bps) {
        insufficient_profit_rejections_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    
    Arbitrage::Opportunity opp{};
    opp.timestamp_us = now_us;
    opp.buy_exchange = buy_exchange;
    opp.sell_exchange = sell_exchange;
    opp.symbol = symbol;
    opp.market_type = config_.market_type;
    opp.buy_price = buy_ask.price;
    opp.sell_price = sell_bid.price;
    opp.profit_bps = profit_bps;
    opp.max_volume = std::min(buy_ask.volume, sell_bid.volume);
    opp.latency_us = std::abs(buy_ask.timestamp_us - sell_bid.timestamp_us);
    
    opportunities_found_.fetch_add(1, std::memory_order_relaxed);
    
    if (opp.is_valid(now_us, config_.max_age_us)) {
        opportunities_valid_.fetch_add(1, std::memory_order_relaxed);
        onOpportunity(opp);
    }
}

void ArbitrageScanner::onOpportunity(const Arbitrage::Opportunity& opp) {
    std::cout << "\n🎯 ARBITRAGE: " << opp.symbol
              << " | Buy " << opp.buy_exchange << " @ " << std::fixed << std::setprecision(2) << opp.buy_price
              << " → Sell " << opp.sell_exchange << " @ " << opp.sell_price
              << " | Profit: " << std::setprecision(2) << opp.profit_bps << " bps"
              << " | Vol: " << std::setprecision(4) << opp.max_volume
              << " | Latency: " << (opp.latency_us / 1000.0) << " ms\n";
    
    if (opportunity_callback_) {
        opportunity_callback_(opp);
    }
}

void ArbitrageScanner::runStatsReporter() {
    int cycle = 0;
    while (running_.load()) {
        std::this_thread::sleep_for(
            std::chrono::seconds(config_.stats_report_interval_s)
        );
        printStats();
        
        if (++cycle % 2 == 0) {
            printDetailedStats();
        }
    }
}

void ArbitrageScanner::printStats() {
    std::cout << "\n=== Arbitrage Scanner Stats ===\n";
    std::cout << "Events received: " << subscription_data_events_.load() << " (data) + " 
              << other_events_.load() << " (other)\n";
    std::cout << "Updates processed: " << updates_processed_.load() << "\n";
    std::cout << "Opportunities found: " << opportunities_found_.load() << "\n";
    std::cout << "Valid opportunities: " << opportunities_valid_.load() << "\n";
    std::cout << "Rejections - Stale: " << stale_data_rejections_.load() 
              << " | Zero: " << zero_price_rejections_.load()
              << " | Low profit: " << insufficient_profit_rejections_.load() << "\n";
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        if (!best_spread_info_.empty()) {
            std::cout << "Best spread seen: " << best_spread_info_ << "\n";
        }
    }
    
    std::cout << "Order books tracked: " << order_books_.size() << "\n";
    std::cout << "================================\n\n";
}

void ArbitrageScanner::printDetailedStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::cout << "\n=== DETAILED STATS ===\n";
    
    std::vector<std::pair<std::string, uint64_t>> ex_updates(
        exchange_update_counts_.begin(), exchange_update_counts_.end()
    );
    std::sort(ex_updates.begin(), ex_updates.end(), 
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "\nTop Exchanges by Updates:\n";
    for (const auto& [ex, count] : ex_updates) {
        std::cout << "  " << std::setw(15) << ex << ": " << count << "\n";
    }
    
    std::vector<std::pair<std::string, uint64_t>> sym_updates(
        symbol_update_counts_.begin(), symbol_update_counts_.end()
    );
    std::sort(sym_updates.begin(), sym_updates.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "\nTop Symbols by Updates:\n";
    for (const auto& [sym, count] : sym_updates) {
        std::cout << "  " << std::setw(15) << sym << ": " << count << "\n";
    }
    
    std::cout << "\nOrder Books With Zero Updates:\n";
    int zero_count = 0;
    for (const auto& [key, book] : order_books_) {
        if (book.update_count.load() == 0) {
            if (zero_count < 20) {
                std::cout << "  " << key.exchange << "/" << key.symbol << "\n";
            }
            zero_count++;
        }
    }
    std::cout << "Total with zero updates: " << zero_count << " / " << order_books_.size() << "\n";
    
    std::cout << "======================\n\n";
}
