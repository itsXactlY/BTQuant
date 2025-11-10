#include "market_data_processor.h"

#include <chrono>
#include <cctype>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string_view>

namespace {

// wall-clock “now” in ms
int64_t nowMs() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               system_clock::now().time_since_epoch())
        .count();
}

// robust double parser with logging
double safeParseDouble(const std::string& label,
                       const std::string& s,
                       bool& ok) {
    ok = false;
    if (s.empty()) {
        return 0.0;
    }
    try {
        std::size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) {
            std::cerr << "safeParseDouble(" << label
                      << "): trailing chars in '" << s << "'\n";
            return 0.0;
        }
        ok = true;
        return v;
    } catch (const std::exception& e) {
        std::cerr << "safeParseDouble(" << label
                  << "): exception for '" << s << "': "
                  << e.what() << "\n";
        return 0.0;
    }
}

// try multiple possible keys on an Element
std::string getAny(const ccapi::Element& el,
                   std::initializer_list<const char*> keys) {
    const auto& m =
        el.getNameValueMap();  // std::map<std::string_view, std::string>
    for (const char* k : keys) {
        auto it = m.find(std::string_view(k));
        if (it != m.end()) return it->second;
    }
    return {};
}

} // anonymous namespace

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

void MarketDataProcessor::processEvent(const ccapi::Event& event,
                                       ccapi::Session* /*session*/) {
    using Type = ccapi::Event::Type;
    const auto type = event.getType();

    try {
        if (type == Type::SESSION_STATUS ||
            type == Type::SUBSCRIPTION_STATUS) {
            std::cout << event.toPrettyString(2, 2) << std::endl;
            return;
        }

        if (type == Type::SUBSCRIPTION_DATA) {
            for (const auto& msg : event.getMessageList()) {
                using MType = ccapi::Message::Type;
                auto mtype = msg.getType();

                if (mtype == MType::MARKET_DATA_EVENTS_TRADE) {
                    handleTradeMessage(msg);
                } else if (mtype ==
                           MType::MARKET_DATA_EVENTS_MARKET_DEPTH) {
                    handleOrderbookMessage(msg);
                }
            }
        }
    } catch (const std::exception& e) {
        ++errors_;
        std::cerr << "Error in processEvent: " << e.what() << std::endl;
    }
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

    std::string exchange    = parts.size() > 0 ? parts[0] : "";
    std::string symbol      = parts.size() > 1 ? parts[1] : "";
    std::string market_type = parts.size() > 2 ? parts[2] : "spot";

    if (exchange.empty() || symbol.empty()) {
        std::cerr << "handleTradeMessage: empty exchange/symbol in CID: "
                  << cid << std::endl;
        return;
    }

    const auto& elements = msg.getElementList();
    if (elements.empty()) return;

    // exchange timestamp from Message (µs)
    auto tp = msg.getTime();
    int64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        tp.time_since_epoch())
                        .count();

    // receive time (µs)
    int64_t recv_time_us = nowMs(); // your nowMs() currently returns microseconds

    for (const auto& el : elements) {
        const auto& m = el.getNameValueMap();

        std::string price_s   = getAny(el, {"LAST_PRICE", "PRICE"});
        std::string qty_s     = getAny(el, {"LAST_SIZE", "SIZE"});
        std::string is_bm_s   = getAny(el, {"IS_BUYER_MAKER"});
        std::string trade_id  = getAny(el, {"TRADE_ID"});

        bool ok_price = false, ok_qty = false;
        double price = safeParseDouble("trade.price", price_s, ok_price);
        double qty   = safeParseDouble("trade.size",  qty_s,   ok_qty);

        if (!ok_price || !ok_qty) {
            std::cerr << "handleTradeMessage: bad PRICE/SIZE, element = "
                      << ccapi::toString(m) << std::endl;
            continue;
        }

        Trade t;
        t.timestamp_ms = ts_us;      // stores microseconds despite the name
        t.exchange     = exchange;
        t.symbol       = symbol;
        t.market_type  = market_type;
        t.price        = price;
        t.quantity     = qty;
        t.trade_id     = trade_id;

        bool buyer_maker = (is_bm_s == "1" || is_bm_s == "true");
        t.is_buyer_maker = buyer_maker;
        t.side           = buyer_maker ? "sell" : "buy";

        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            trade_buffer_.push_back(t);
            active_pairs_.insert(exchange + ":" + symbol + ":" + market_type);
            ++trades_received_;
        }

        candle_agg_->processTrade(t);

        // keep stats as milliseconds
        double latency_ms =
            static_cast<double>(recv_time_us - t.timestamp_ms) / 1000.0;
        stats_.avg_latency_ms = 0.99 * stats_.avg_latency_ms + 0.01 * latency_ms;
    }

    flushTradesIfNeeded();
}

void MarketDataProcessor::handleOrderbookMessage(
    const ccapi::Message& msg) {

    const auto& cid_list = msg.getCorrelationIdList();
    const std::string cid = cid_list.empty() ? "" : cid_list[0];
    auto parts = split(cid, ':');

    std::string exchange    = parts.size() > 0 ? parts[0] : "";
    std::string symbol      = parts.size() > 1 ? parts[1] : "";
    std::string market_type = parts.size() > 2 ? parts[2] : "spot";

    if (exchange.empty() || symbol.empty()) {
        std::cerr << "handleOrderbookMessage: empty exchange/symbol in CID: "
                  << cid << std::endl;
        return;
    }

    const auto& elements = msg.getElementList();
    if (elements.empty()) return;

    // Collect best bid / ask across all elements in this message
    std::string bid_p_s, bid_q_s, ask_p_s, ask_q_s;

    for (const auto& el : elements) {
        auto bp = getAny(el, {"BID_PRICE", "BEST_BID_PRICE"});
        if (!bp.empty()) bid_p_s = bp;

        auto bq = getAny(el, {"BID_SIZE", "BEST_BID_SIZE"});
        if (!bq.empty()) bid_q_s = bq;

        auto ap = getAny(el, {"ASK_PRICE", "BEST_ASK_PRICE"});
        if (!ap.empty()) ask_p_s = ap;

        auto aq = getAny(el, {"ASK_SIZE", "BEST_ASK_SIZE"});
        if (!aq.empty()) ask_q_s = aq;
    }

    bool have_bid = !bid_p_s.empty() && !bid_q_s.empty();
    bool have_ask = !ask_p_s.empty() && !ask_q_s.empty();

    // If we got neither side, *then* log and bail
    if (!have_bid && !have_ask) {
        const auto& firstMap = elements.front().getNameValueMap();
        std::cerr << "handleOrderbookMessage: no BID/ASK in message, element = "
                  << ccapi::toString(firstMap) << std::endl;
        return;
    }

    bool ok_bp = true, ok_bq = true, ok_ap = true, ok_aq = true;
    double bid_p = 0.0, bid_q = 0.0, ask_p = 0.0, ask_q = 0.0;

    if (have_bid) {
        bid_p = safeParseDouble("bid_price", bid_p_s, ok_bp);
        bid_q = safeParseDouble("bid_size",  bid_q_s, ok_bq);
    }
    if (have_ask) {
        ask_p = safeParseDouble("ask_price", ask_p_s, ok_ap);
        ask_q = safeParseDouble("ask_size",  ask_q_s, ok_aq);
    }

    if ((have_bid && (!ok_bp || !ok_bq)) ||
        (have_ask && (!ok_ap || !ok_aq))) {
        const auto& firstMap = elements.front().getNameValueMap();
        std::cerr << "handleOrderbookMessage: parse error, element = "
                  << ccapi::toString(firstMap) << std::endl;
        return;
    }

    // timestamp from Message (stored internally as microseconds)
    auto tp = msg.getTime();
    int64_t ts_us = std::chrono::duration_cast<
                        std::chrono::microseconds>(
                        tp.time_since_epoch())
                        .count();

    OrderbookSnapshot ob;
    ob.timestamp_ms = ts_us;
    ob.exchange     = exchange;
    ob.symbol       = symbol;
    ob.market_type  = market_type;

    // Build JSON; allow one-sided updates
    std::ostringstream bids, asks;
    bids << "[";
    if (have_bid) {
        bids << "[" << bid_p << "," << bid_q << "]";
    }
    bids << "]";

    asks << "[";
    if (have_ask) {
        asks << "[" << ask_p << "," << ask_q << "]";
    }
    asks << "]";

    ob.bids_json = bids.str();
    ob.asks_json = asks.str();
    ob.checksum.clear();

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
