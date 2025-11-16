#include "market_data_processor.h"
#include <algorithm>
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

    // ── 1) Decode correlation id → exchange, symbol, market_type ─────────────
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
    if (elements.empty()) {
        return;
    }

    struct DepthLevel {
        double price{};
        double size{};
    };

    std::vector<double> bid_prices;
    std::vector<double> bid_sizes;
    std::vector<double> ask_prices;
    std::vector<double> ask_sizes;

    // ── 2) Collect all BID/ASK prices & sizes from the snapshot ──────────────
    for (const auto& el : elements) {
        const auto& m = el.getNameValueMap();  // map<string_view,string>
        for (const auto& kv : m) {
            std::string_view name = kv.first;
            const std::string& val = kv.second;

            bool ok = true;
            if (name == "BID_PRICE") {
                double p = safeParseDouble("BID_PRICE", val, ok);
                if (ok) bid_prices.push_back(p);
            } else if (name == "BID_SIZE") {
                double q = safeParseDouble("BID_SIZE", val, ok);
                if (ok) bid_sizes.push_back(q);
            } else if (name == "ASK_PRICE") {
                double p = safeParseDouble("ASK_PRICE", val, ok);
                if (ok) ask_prices.push_back(p);
            } else if (name == "ASK_SIZE") {
                double q = safeParseDouble("ASK_SIZE", val, ok);
                if (ok) ask_sizes.push_back(q);
            }
        }
    }

    if (bid_prices.empty() && ask_prices.empty()) {
        const auto& firstMap = elements.front().getNameValueMap();
        std::cerr << "handleOrderbookMessage: no BID/ASK in message, element = "
                  << ccapi::toString(firstMap) << std::endl;
        return;
    }

    // Pair prices and sizes by index, tolerate tiny mismatches in count
    std::vector<DepthLevel> bids;
    std::vector<DepthLevel> asks;

    {
        std::size_t n = std::min(bid_prices.size(), bid_sizes.size());
        bids.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            bids.push_back(DepthLevel{bid_prices[i], bid_sizes[i]});
        }
        if (bid_prices.size() != bid_sizes.size()) {
            std::cerr << "handleOrderbookMessage: BID price/size count mismatch: "
                      << bid_prices.size() << " prices vs "
                      << bid_sizes.size()  << " sizes for "
                      << exchange << " " << symbol << std::endl;
        }
    }
    {
        std::size_t n = std::min(ask_prices.size(), ask_sizes.size());
        asks.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            asks.push_back(DepthLevel{ask_prices[i], ask_sizes[i]});
        }
        if (ask_prices.size() != ask_sizes.size()) {
            std::cerr << "handleOrderbookMessage: ASK price/size count mismatch: "
                      << ask_prices.size() << " prices vs "
                      << ask_sizes.size()  << " sizes for "
                      << exchange << " " << symbol << std::endl;
        }
    }

    if (bids.empty() && asks.empty()) {
        return;
    }

    // ── 3) Sort and trim to 20 levels each side ──────────────────────────────
    constexpr std::size_t MAX_LEVELS = 20;

    std::sort(
        bids.begin(), bids.end(),
        [](const DepthLevel& a, const DepthLevel& b) {
            return a.price > b.price;  // best bid first
        });
    std::sort(
        asks.begin(), asks.end(),
        [](const DepthLevel& a, const DepthLevel& b) {
            return a.price < b.price;  // best ask first
        });

    if (bids.size() > MAX_LEVELS) bids.resize(MAX_LEVELS);
    if (asks.size() > MAX_LEVELS) asks.resize(MAX_LEVELS);

    // ── 4) Timestamp: internal = µs since epoch ──────────────────────────────
    auto tp = msg.getTime();
    int64_t ts_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            tp.time_since_epoch())
            .count();

    OrderbookSnapshot ob;
    ob.timestamp_us = ts_us;   // µs since epoch (UTC)
    ob.exchange     = exchange;
    ob.symbol       = symbol;
    ob.market_type  = market_type;

    auto buildSideJson = [](const std::vector<DepthLevel>& side) {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < side.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "[" << side[i].price << "," << side[i].size << "]";
        }
        oss << "]";
        return oss.str();
    };

    ob.bids_json = buildSideJson(bids);
    ob.asks_json = buildSideJson(asks);
    ob.checksum.clear();  // reserved for later (e.g. exchange checksum)

    // ── 5) Buffer + stats ────────────────────────────────────────────────────
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

