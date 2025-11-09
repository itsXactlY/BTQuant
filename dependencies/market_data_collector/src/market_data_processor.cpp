#include "market_data_processor.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace {

int64_t nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        steady_clock::now().time_since_epoch()).count();
}

} // anonymous

// ccapi logger definition (must exist in exactly one TU)
namespace ccapi {
Logger* Logger::logger = nullptr;
}

using namespace MarketData;

MarketDataProcessor::MarketDataProcessor(
    std::shared_ptr<MSSQLBulkInserter> db,
    std::shared_ptr<CandleAggregator> candle_agg)
    : db_(std::move(db)), candle_agg_(std::move(candle_agg)) {}

std::vector<std::string> MarketDataProcessor::split(
    const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        parts.push_back(item);
    }
    return parts;
}

bool MarketDataProcessor::processEvent(
    const ccapi::Event& event,
    ccapi::Session* /*session*/) {

    using Type = ccapi::Event::Type;
    const auto type = event.getType();

    try {
        if (type == Type::SESSION_STATUS ||
            type == Type::SUBSCRIPTION_STATUS) {
            std::cout << event.toPrettyString(2, 2) << std::endl;
            return true;
        }

        if (type == Type::SUBSCRIPTION_DATA) {
            for (const auto& msg : event.getMessageList()) {
                auto mtype = msg.getType();
                using MType = ccapi::Message::Type;

                if (mtype == MType::MARKET_DATA_EVENTS_TRADE) {
                    handleTradeMessage(msg);
                } else if (mtype == MType::MARKET_DATA_EVENTS_MARKET_DEPTH) {
                    handleOrderbookMessage(msg);
                }
            }
        }
    } catch (const std::exception& e) {
        ++errors_;
        std::cerr << "Error in processEvent: " << e.what() << std::endl;
    }

    return true;
}

void MarketDataProcessor::setBufferLimits(
    std::size_t max_trades,
    std::size_t max_candles,
    std::size_t max_orderbooks) {

    max_trade_buffer_size_     = max_trades;
    max_candle_buffer_size_    = max_candles;
    max_orderbook_buffer_size_ = max_orderbooks;
}

void MarketDataProcessor::handleTradeMessage(const ccapi::Message& msg) {
    const auto& cid_list = msg.getCorrelationIdList();
    const std::string cid = cid_list.empty() ? "" : cid_list[0];
    auto parts = split(cid, ':');

    std::string exchange    = parts.size() > 0 ? parts[0] : msg.getExchange();
    std::string symbol      = parts.size() > 1 ? parts[1] : msg.getInstrument();
    std::string market_type = parts.size() > 2 ? parts[2] : "spot";

    const auto& elements = msg.getElementList();

    int64_t recv_time_ms = nowMs();

    for (const auto& el : elements) {
        Trade t;
        t.timestamp_ms = std::stoll(el.getValue("TIME"));
        t.price        = std::stod(el.getValue("PRICE"));
        t.quantity     = std::stod(el.getValue("SIZE"));

        // IS_BUYER_MAKER semantics: taker side
        std::string is_bm = el.getValue("IS_BUYER_MAKER");
        bool buyer_maker  = (is_bm == "1" || is_bm == "true");
        t.is_buyer_maker  = buyer_maker;
        t.side            = buyer_maker ? "sell" : "buy";

        t.exchange    = exchange;
        t.symbol      = symbol;
        t.market_type = market_type;
        if (el.has("TRADE_ID")) {
            t.trade_id = el.getValue("TRADE_ID");
        }

        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            trade_buffer_.push_back(t);
            active_pairs_.insert(exchange + ":" + symbol + ":" + market_type);
            ++trades_received_;
        }

        candle_agg_->processTrade(t);

        double latency =
            static_cast<double>(recv_time_ms - t.timestamp_ms);
        // naive moving avg
        stats_.avg_latency_ms =
            0.99 * stats_.avg_latency_ms + 0.01 * latency;
    }

    flushTradesIfNeeded();
    // candles are pulled in flushBuffers()
}

void MarketDataProcessor::handleOrderbookMessage(
    const ccapi::Message& msg) {
    // Minimal: best bid/ask snapshot → JSON arrays [[price,qty]]
    const auto& cid_list = msg.getCorrelationIdList();
    const std::string cid = cid_list.empty() ? "" : cid_list[0];
    auto parts = split(cid, ':');

    std::string exchange    = parts.size() > 0 ? parts[0] : msg.getExchange();
    std::string symbol      = parts.size() > 1 ? parts[1] : msg.getInstrument();
    std::string market_type = parts.size() > 2 ? parts[2] : "spot";

    const auto& elements = msg.getElementList();
    if (elements.empty()) return;

    // For production, you’d rebuild full depth here.
    const auto& el = elements.front();
    std::string bid_p = el.getValue("BEST_BID_PRICE");
    std::string bid_q = el.getValue("BEST_BID_SIZE");
    std::string ask_p = el.getValue("BEST_ASK_PRICE");
    std::string ask_q = el.getValue("BEST_ASK_SIZE");

    OrderbookSnapshot ob;
    ob.timestamp_ms = std::stoll(el.getValue("TIME"));
    ob.exchange     = exchange;
    ob.symbol       = symbol;
    ob.market_type  = market_type;
    ob.bids_json    = "[[" + bid_p + "," + bid_q + "]]";
    ob.asks_json    = "[[" + ask_p + "," + ask_q + "]]";
    ob.checksum     = ""; // TODO: optional checksum

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        orderbook_buffer_.push_back(std::move(ob));
        ++stats_.orderbooks_received;
    }

    flushOrderbooksIfNeeded();
}

void MarketDataProcessor::flushTradesIfNeeded(bool force) {
    std::vector<Trade> batch;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (!force && trade_buffer_.size() < max_trade_buffer_size_) {
            return;
        }
        batch.swap(trade_buffer_);
    }
    if (batch.empty()) return;

    try {
        db_->bulkInsertTrades(batch);
        stats_.trades_inserted += batch.size();
    } catch (const std::exception& e) {
        ++errors_;
        std::cerr << "flushTradesIfNeeded error: " << e.what() << std::endl;
    }
}

void MarketDataProcessor::flushCandlesIfNeeded(bool force) {
    // Pull newly completed candles from aggregator
    auto newly_completed = candle_agg_->getAllCompletedCandles();
    stats_.candles_generated += newly_completed.size();

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        candle_buffer_.insert(candle_buffer_.end(),
                              newly_completed.begin(),
                              newly_completed.end());
        if (!force &&
            candle_buffer_.size() < max_candle_buffer_size_) {
            return;
        }
        if (candle_buffer_.empty()) return;

        // group per table
        std::unordered_map<std::string,
                           std::vector<OHLCV>> by_table;
        for (const auto& c : candle_buffer_) {
            by_table[c.getTableName()].push_back(c);
        }
        candle_buffer_.clear();

        for (auto& kv : by_table) {
            try {
                db_->bulkInsertOHLCV(kv.first, kv.second);
                stats_.candles_inserted += kv.second.size();
            } catch (const std::exception& e) {
                ++errors_;
                std::cerr << "flushCandlesIfNeeded error: "
                          << e.what() << std::endl;
            }
        }
    }
}

void MarketDataProcessor::flushOrderbooksIfNeeded(bool force) {
    std::vector<OrderbookSnapshot> batch;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (!force &&
            orderbook_buffer_.size() < max_orderbook_buffer_size_) {
            return;
        }
        batch.swap(orderbook_buffer_);
    }
    if (batch.empty()) return;

    try {
        db_->bulkInsertOrderbooks(batch);
        stats_.orderbooks_inserted += batch.size();
    } catch (const std::exception& e) {
        ++errors_;
        std::cerr << "flushOrderbooksIfNeeded error: "
                  << e.what() << std::endl;
    }
}

void MarketDataProcessor::flushBuffers() {
    flushTradesIfNeeded(true);
    flushCandlesIfNeeded(true);
    flushOrderbooksIfNeeded(true);
}

MarketDataProcessor::Stats MarketDataProcessor::getStats() const {
    Stats out;
    out = stats_;
    out.trades_received = trades_received_.load();
    out.errors          = errors_.load();
    return out;
}
