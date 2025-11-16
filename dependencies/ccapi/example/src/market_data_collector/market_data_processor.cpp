#include "market_data_processor.h"

#include <chrono>
#include <cctype>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string_view>

namespace {

// wall-clock “now” in ms
int64_t nowMicros() {
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

    const std::string key = exchange + ":" + symbol + ":" + market_type;

    const auto& elements = msg.getElementList();
    if (elements.empty()) return;

    // exchange timestamp from Message (µs)
    auto tp = msg.getTime();
    int64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        tp.time_since_epoch())
                        .count();

    // receive time (µs)
    int64_t recv_time_us = nowMicros();

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
        t.timestamp_us = ts_us;      // stores microseconds despite the name
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
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            pair_stats_[key].trades++;
        }
        candle_agg_->processTrade(t);

        // keep stats as milliseconds
        double latency_ms =
            static_cast<double>(recv_time_us - t.timestamp_us) / 1000.0;
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

    const std::string key = exchange + ":" + symbol + ":" + market_type;

    const auto& elements = msg.getElementList();
    if (elements.empty()) return;

    // exchange timestamp from Message (µs)
    auto tp = msg.getTime();
    int64_t ts_us = std::chrono::duration_cast<
                        std::chrono::microseconds>(
                        tp.time_since_epoch()).count();

    // Collect *all* levels across all elements in this message
    struct Level { double price; double qty; };
    std::vector<Level> bids;
    std::vector<Level> asks;

    bids.reserve(elements.size());
    asks.reserve(elements.size());

    for (const auto& el : elements) {
        // One ccapi Element may contain both a bid and an ask
        const auto& m = el.getNameValueMap();

        auto bid_p_s = getAny(el, {"BID_PRICE", "BEST_BID_PRICE"});
        auto bid_q_s = getAny(el, {"BID_SIZE",  "BEST_BID_SIZE"});
        auto ask_p_s = getAny(el, {"ASK_PRICE", "BEST_ASK_PRICE"});
        auto ask_q_s = getAny(el, {"ASK_SIZE",  "BEST_ASK_SIZE"});

        bool ok = true;

        if (!bid_p_s.empty() && !bid_q_s.empty()) {
            bool okp = false, okq = false;
            double p = safeParseDouble("bid_price", bid_p_s, okp);
            double q = safeParseDouble("bid_size",  bid_q_s, okq);
            if (okp && okq && q > 0.0) {
                bids.push_back({p, q});
            } else if (!okp || !okq) {
                std::cerr << "handleOrderbookMessage: bad BID level, element = "
                          << ccapi::toString(m) << std::endl;
            }
        }

        if (!ask_p_s.empty() && !ask_q_s.empty()) {
            bool okp = false, okq = false;
            double p = safeParseDouble("ask_price", ask_p_s, okp);
            double q = safeParseDouble("ask_size",  ask_q_s, okq);
            if (okp && okq && q > 0.0) {
                asks.push_back({p, q});
            } else if (!okp || !okq) {
                std::cerr << "handleOrderbookMessage: bad ASK level, element = "
                          << ccapi::toString(m) << std::endl;
            }
        }
    }

    if (bids.empty() && asks.empty()) {
        const auto& firstMap = elements.front().getNameValueMap();
        std::cerr << "handleOrderbookMessage: no BID/ASK levels, element = "
                  << ccapi::toString(firstMap) << std::endl;
        return;
    }

    MarketData::OrderbookSnapshot ob;
    ob.timestamp_us = ts_us;
    ob.exchange     = exchange;
    ob.symbol       = symbol;
    ob.market_type  = market_type;

    // Build JSON: [[price, qty], ...]
    auto build_side_json = [](const std::vector<Level>& side) {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < side.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "[" << side[i].price << "," << side[i].qty << "]";
        }
        oss << "]";
        return oss.str();
    };

    ob.bids_json = build_side_json(bids);
    ob.asks_json = build_side_json(asks);
    ob.checksum.clear(); // can be wired later if exchange supports it

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        orderbook_buffer_.push_back(std::move(ob));
        ++stats_.orderbooks_received;
        active_pairs_.insert(exchange + ":" + symbol + ":" + market_type);
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        pair_stats_[key].orderbooks++;
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
        std::lock_guard<std::mutex> db_lock(db_mutex_);
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
                std::lock_guard<std::mutex> db_lock(db_mutex_);
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
        std::lock_guard<std::mutex> db_lock(db_mutex_);
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
    {
        out = stats_;
        out.trades_received = trades_received_.load();
        out.errors          = errors_.load();
    }

    const int64_t now_us = nowMicros();
    int64_t last_us = last_stats_ts_us_.exchange(now_us);
    if (last_us > 0) {
        double dt_sec = static_cast<double>(now_us - last_us) / 1'000'000.0;
        if (dt_sec > 0.1) {
            uint64_t tr = trades_received_.load();
            uint64_t ob = stats_.orderbooks_received;

            uint64_t tr_prev =
                trades_last_window_.exchange(tr);
            uint64_t ob_prev =
                orderbooks_last_window_.exchange(ob);

            double tr_rate = static_cast<double>(tr - tr_prev) / dt_sec;
            double ob_rate = static_cast<double>(ob - ob_prev) / dt_sec;

            out.trades_per_sec     = tr_rate;
            out.orderbooks_per_sec = ob_rate;
        }
    }

    return out;
}

std::unordered_map<std::string, MarketDataProcessor::PairStats>
MarketDataProcessor::getPairStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return pair_stats_; // copy
}

std::string MarketDataProcessor::getStatsJson() const {
    auto st   = getStats();
    auto pmap = getPairStats();

    std::ostringstream oss;
    oss << "{";
    oss << "\"trades_received\":"    << st.trades_received    << ",";
    oss << "\"trades_inserted\":"    << st.trades_inserted    << ",";
    oss << "\"candles_generated\":"  << st.candles_generated  << ",";
    oss << "\"candles_inserted\":"   << st.candles_inserted   << ",";
    oss << "\"orderbooks_received\":"<< st.orderbooks_received<< ",";
    oss << "\"orderbooks_inserted\":"<< st.orderbooks_inserted<< ",";
    oss << "\"errors\":"             << st.errors             << ",";
    oss << "\"avg_latency_ms\":"     << st.avg_latency_ms     << ",";
    oss << "\"trades_per_sec\":"     << st.trades_per_sec     << ",";
    oss << "\"orderbooks_per_sec\":" << st.orderbooks_per_sec << ",";

    oss << "\"pairs\":{";
    bool first = true;
    for (const auto& kv : pmap) {
        if (!first) oss << ",";
        first = false;
        oss << "\"" << kv.first << "\":{"
            << "\"trades\":"     << kv.second.trades     << ","
            << "\"orderbooks\":" << kv.second.orderbooks
            << "}";
    }
    oss << "}}";
    return oss.str();
}

