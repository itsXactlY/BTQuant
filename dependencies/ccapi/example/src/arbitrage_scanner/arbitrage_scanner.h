#pragma once

#include "ccapi_cpp/ccapi_session.h"
#include "arbitrage_types.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <functional>
#include <thread>
#include <mutex>

class ArbitrageScanner : public ccapi::EventHandler {
public:
    struct Config {
        std::vector<std::string> exchanges;
        std::vector<std::string> symbols;  // normalized symbols like "BTC-USDT"
        std::string market_type = "spot";
        double min_profit_bps = 30.0;
        int64_t max_age_us = 100'000;
        size_t buffer_size = 1000;
        int flush_interval_ms = 1000;
        int stats_report_interval_s = 10;
    };
    
    using OpportunityCallback = std::function<void(const Arbitrage::Opportunity&)>;

    explicit ArbitrageScanner(const Config& cfg);
    ~ArbitrageScanner() override = default;

    void start();
    void stop();
    void waitForShutdown();
    
    void setOpportunityCallback(OpportunityCallback cb) {
        opportunity_callback_ = std::move(cb);
    }
    
    void processEvent(const ccapi::Event& event, ccapi::Session* session) override;

private:
    struct OrderBookKey {
        std::string exchange;
        std::string symbol;
        
        bool operator==(const OrderBookKey& o) const {
            return exchange == o.exchange && symbol == o.symbol;
        }
    };
    
    struct OrderBookKeyHash {
        std::size_t operator()(const OrderBookKey& k) const {
            return std::hash<std::string>{}(k.exchange + ":" + k.symbol);
        }
    };
    
    struct alignas(128) OrderBook {
        Arbitrage::OrderBookLevel bid;
        Arbitrage::OrderBookLevel ask;
        std::atomic<uint64_t> update_count{0};
        std::atomic<int64_t> last_update_us{0};
    };

    std::string getExchangeSymbol(const std::string& exchange, const std::string& normalized_symbol);
    std::string normalizeSymbol(const std::string& exchange_symbol, const std::string& exchange);
    
    void processMarketDepth(const std::string& exchange,
                             const std::string& symbol,
                             const ccapi::Element& element,
                             int64_t timestamp_us);
    
    void scanArbitrage(const std::string& symbol, int64_t now_us);
    
    void checkArbitrageDirection(
        const std::string& buy_exchange,
        const std::string& sell_exchange,
        const std::string& symbol,
        const Arbitrage::OrderBookLevel::Snapshot& buy_ask,
        const Arbitrage::OrderBookLevel::Snapshot& sell_bid,
        int64_t now_us);
    
    void onOpportunity(const Arbitrage::Opportunity& opp);
    void printStats();
    void printDetailedStats();
    void runStatsReporter();

    Config config_;
    std::unordered_map<OrderBookKey, OrderBook, OrderBookKeyHash> order_books_;
    
    // symbol mapping: exchange -> (normalized_symbol -> exchange_symbol)
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> symbol_map_;
    
    std::unique_ptr<ccapi::Session> session_;
    OpportunityCallback opportunity_callback_;
    
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> updates_processed_{0};
    std::atomic<uint64_t> opportunities_found_{0};
    std::atomic<uint64_t> opportunities_valid_{0};
    std::atomic<uint64_t> stale_data_rejections_{0};
    std::atomic<uint64_t> zero_price_rejections_{0};
    std::atomic<uint64_t> insufficient_profit_rejections_{0};
    std::atomic<uint64_t> subscription_data_events_{0};
    std::atomic<uint64_t> other_events_{0};
    
    std::thread stats_thread_;
    std::mutex stats_mutex_;
    std::unordered_map<std::string, uint64_t> exchange_update_counts_;
    std::unordered_map<std::string, uint64_t> symbol_update_counts_;
    
    std::atomic<double> best_spread_bps_{-999999.0};
    std::string best_spread_info_;
};
