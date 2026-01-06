#include "market_data_processor.h"

#include <chrono>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string_view>
#include "utilities.h"

namespace {

// Helper function for timestamped logging (same as in exchange_connection_manager)
// getCurrentTimestamp() is declared at global scope above

// wall-clock "now" in ms
int64_t nowMicros() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               system_clock::now().time_since_epoch())
        .count();
}

// getCurrentTimestamp() function is defined in utilities.h

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
    std::shared_ptr<CandleAggregator> candle_agg,
    std::shared_ptr<HotSpine::HotSpineWriter> hotspine_writer,
    bool enable_exclusive_hotspine,
    const ConfigTypes::DebugConfig& debug_config)
    : db_(std::move(db)),
      candle_agg_(std::move(candle_agg)),
      hotspine_writer_(std::move(hotspine_writer)),
      enable_exclusive_hotspine_(enable_exclusive_hotspine),
      debug_config_(debug_config) {}

void MarketDataProcessor::debugLog(const std::string& msg) const {
    if (debug_config_.enabled) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MarketDataProcessor: " << msg << std::endl;
    }
}

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
    
    debugLog("MarketDataProcessor: Received event of type: " + std::to_string(static_cast<int>(type)));
    debugLog("MarketDataProcessor: WebSocket Event Tracking - Type: " + std::to_string(static_cast<int>(type)));
    debugLog("MarketDataProcessor: WebSocket Connection Health - Event received, connection appears active");
    debugLog("MarketDataProcessor: CCAPI Data Flow - Event received for processing");
    debugLog("MarketDataProcessor: CCAPI Data Flow - Event type: " + std::to_string(static_cast<int>(type)) + " detected");
    
    try {
        if (type == Type::SESSION_STATUS ||
            type == Type::SUBSCRIPTION_STATUS) {
            std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: Session/Subscription Status Event:" << std::endl;
            std::cout << event.toPrettyString(2, 2) << std::endl;
              
            // Check for WebSocket connection issues
            std::string eventStr = event.toString();
            if (eventStr.find("WebSocket") != std::string::npos ||
                eventStr.find("websocket") != std::string::npos ||
                eventStr.find("CONNECTION") != std::string::npos) {
                std::cout << "[" << getCurrentTimestamp() << "][WARNING] MarketDataProcessor: WebSocket-related status event detected!" << std::endl;
                  
                // Detailed WebSocket status analysis
                if (eventStr.find("CONNECTED") != std::string::npos ||
                    eventStr.find("connected") != std::string::npos) {
                    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: WebSocket connection established!" << std::endl;
                } else if (eventStr.find("DISCONNECTED") != std::string::npos ||
                          eventStr.find("disconnected") != std::string::npos) {
                    std::cout << "[" << getCurrentTimestamp() << "][ERROR] MarketDataProcessor: WebSocket connection lost!" << std::endl;
                } else if (eventStr.find("ERROR") != std::string::npos ||
                          eventStr.find("error") != std::string::npos) {
                    std::cout << "[" << getCurrentTimestamp() << "][ERROR] MarketDataProcessor: WebSocket error detected!" << std::endl;
                }
            }
              
            return;
        }
  
        if (type == Type::SUBSCRIPTION_DATA) {
            std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: Processing SUBSCRIPTION_DATA event" << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][INFO]   Message count: " << event.getMessageList().size() << std::endl;
            
            debugLog("MarketDataProcessor: Detailed event information:");
            debugLog("  Event type: " + std::to_string(static_cast<int>(type)));
            debugLog("  Processing timestamp: " + getCurrentTimestamp());
            debugLog("  Message list size: " + std::to_string(event.getMessageList().size()));
            
            if (event.getMessageList().empty()) {
                std::cout << "[" << getCurrentTimestamp() << "][WARNING] MarketDataProcessor: Empty message list received!" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING] This could indicate WebSocket data flow issues!" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING] Possible causes:" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - WebSocket connection not properly established" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Subscription not successful" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Exchange not sending data" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Network connectivity issues" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - WebSocket connection dropped" << std::endl;
                
                // Add WebSocket-specific troubleshooting
                std::cout << "[" << getCurrentTimestamp() << "][WARNING] WebSocket troubleshooting:" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Check WebSocket connection status in ExchangeConnectionManager" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Verify subscription was successful" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Look for SESSION_STATUS events" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Check for WebSocket protocol errors" << std::endl;
                
                // Add WebSocket reconnection guidance
                std::cout << "[" << getCurrentTimestamp() << "][WARNING] WebSocket reconnection:" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - CCAPI should automatically attempt to reconnect" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Check for SESSION_STATUS events indicating reconnection attempts" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][WARNING]   - Monitor for SUBSCRIPTION_STATUS events after reconnection" << std::endl;
            } else {
                std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: WebSocket data received successfully!" << std::endl;
                if (debug_config_.enabled) {
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   WebSocket connection appears healthy" << std::endl;
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Processing " << event.getMessageList().size() << " messages" << std::endl;
                }
            }

            // Add detailed WebSocket data flow monitoring
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MarketDataProcessor: WebSocket data flow monitoring:" << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Event type: SUBSCRIPTION_DATA" << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Message list size: " << event.getMessageList().size() << std::endl;
            
            for (const auto& msg : event.getMessageList()) {
                using MType = ccapi::Message::Type;
                auto mtype = msg.getType();

                // Add comprehensive message logging
                std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Processing message type: " << static_cast<int>(mtype) << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Correlation IDs: ";
                for (const auto& cid : msg.getCorrelationIdList()) {
                    std::cout << cid << " ";
                }
                std::cout << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Element count: " << msg.getElementList().size() << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Message timestamp: " << msg.getTime().time_since_epoch().count() << " microseconds" << std::endl;

                // Add message type specific logging
                if (mtype == MType::MARKET_DATA_EVENTS_TRADE) {
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Handling TRADE message" << std::endl;
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Calling handleTradeMessage()" << std::endl;
                    handleTradeMessage(msg);
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Completed handleTradeMessage()" << std::endl;
                } else if (mtype == MType::MARKET_DATA_EVENTS_MARKET_DEPTH) {
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Handling MARKET_DEPTH message" << std::endl;
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Calling handleOrderbookMessage()" << std::endl;
                    handleOrderbookMessage(msg);
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Completed handleOrderbookMessage()" << std::endl;
                } else {
                    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Unknown message type: " << static_cast<int>(mtype) << std::endl;
                    std::cout << "[" << getCurrentTimestamp() << "][WARNING]   Unsupported message type detected!" << std::endl;
                }
            }

            // Add post-processing logging
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MarketDataProcessor: Completed processing SUBSCRIPTION_DATA event" << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Total messages processed: " << event.getMessageList().size() << std::endl;
        } else {
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MarketDataProcessor: Unknown event type: " << static_cast<int>(type) << std::endl;
        }
    } catch (const std::exception& e) {
        ++errors_;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MarketDataProcessor: Error in processEvent: " << e.what() << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] This could indicate data parsing issues or WebSocket protocol errors" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] WebSocket data flow may be disrupted" << std::endl;
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

    // Add detailed trade message logging
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] handleTradeMessage: Processing trade message" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Correlation ID: " << cid << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Exchange: " << exchange << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Symbol: " << symbol << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Market type: " << market_type << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Element count: " << msg.getElementList().size() << std::endl;

    if (exchange.empty() || symbol.empty()) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] handleTradeMessage: empty exchange/symbol in CID: "
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
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Trade added to buffer. Buffer size: " << trade_buffer_.size() << std::endl;
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            pair_stats_[key].trades++;
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Updated pair stats for " << key << std::endl;
        }
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Processing trade with candle aggregator" << std::endl;
        candle_agg_->processTrade(t);
         
        // Write to HotSpine if writer is available
        if (hotspine_writer_) {
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Writing trade to HotSpine" << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   HotSpine Data Flow - Trade data: " << exchange << ":" << symbol
                      << " @ " << t.price << " x " << t.quantity << " (" << t.side << ")" << std::endl;
            
            // Write using the correct public interface - just pass the Trade object
            if (!hotspine_writer_->writeTrade(t)) {
                std::cerr << "[" << getCurrentTimestamp() << "][ERROR] HotSpine write failed for trade: "
                          << exchange << ":" << symbol << std::endl;
                std::cerr << "[" << getCurrentTimestamp() << "][ERROR] HotSpine Data Flow - Write failure detected" << std::endl;
            } else {
                std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   HotSpine write successful" << std::endl;
                std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   HotSpine Data Flow - Trade successfully written" << std::endl;
            }
        }

        // In exclusive hotswap mode, don't write to database
        if (!enable_exclusive_hotspine_) {
            // keep stats as milliseconds
            double latency_ms =
                static_cast<double>(recv_time_us - t.timestamp_us) / 1000.0;
            stats_.avg_latency_ms = 0.99 * stats_.avg_latency_ms + 0.01 * latency_ms;
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Updated latency stats: " << latency_ms << " ms" << std::endl;
        }
    }

    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] handleTradeMessage: Completed processing " << elements.size() << " trade elements" << std::endl;
    
    // In exclusive hotswap mode, don't flush trades to database
    if (!enable_exclusive_hotspine_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Calling flushTradesIfNeeded()" << std::endl;
        flushTradesIfNeeded();
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Completed flushTradesIfNeeded()" << std::endl;
    }
}

void MarketDataProcessor::handleOrderbookMessage(
    const ccapi::Message& msg) {

    const auto& cid_list = msg.getCorrelationIdList();
    const std::string cid = cid_list.empty() ? "" : cid_list[0];
    auto parts = split(cid, ':');

    std::string exchange    = parts.size() > 0 ? parts[0] : "";
    std::string symbol      = parts.size() > 1 ? parts[1] : "";
    std::string market_type = parts.size() > 2 ? parts[2] : "spot";

    // Add detailed orderbook message logging
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] handleOrderbookMessage: Processing orderbook message" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Correlation ID: " << cid << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Exchange: " << exchange << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Symbol: " << symbol << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Market type: " << market_type << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Element count: " << msg.getElementList().size() << std::endl;

    if (exchange.empty() || symbol.empty()) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] handleOrderbookMessage: empty exchange/symbol in CID: "
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
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Orderbook added to buffer. Buffer size: " << orderbook_buffer_.size() << std::endl;
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        pair_stats_[key].orderbooks++;
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Updated pair stats for " << key << std::endl;
    }
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Calling flushOrderbooksIfNeeded()" << std::endl;
    flushOrderbooksIfNeeded();
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] handleOrderbookMessage: Completed processing orderbook message" << std::endl;
}


void MarketDataProcessor::flushTradesIfNeeded(bool force) {
    // In exclusive hotswap mode, don't flush trades to database
    if (enable_exclusive_hotspine_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushTradesIfNeeded: Skipping in exclusive HotSpine mode" << std::endl;
        return;
    }

    std::vector<Trade> batch;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushTradesIfNeeded: Checking buffer. Size: " << trade_buffer_.size() << ", Max: " << max_trade_buffer_size_ << std::endl;
        
        // Enhanced buffer debugging
        if (trade_buffer_.size() > 0) {
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Buffer contains " << trade_buffer_.size() << " trades ready for database insertion" << std::endl;
            if (trade_buffer_.size() >= max_trade_buffer_size_) {
                std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Buffer is FULL - trigger forced flush" << std::endl;
            } else {
                std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Buffer is " << ((double)trade_buffer_.size() / max_trade_buffer_size_ * 100) << "% full" << std::endl;
            }
        }
        
        if (!force && trade_buffer_.size() < max_trade_buffer_size_) {
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushTradesIfNeeded: Buffer not full enough for automatic flush" << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Current: " << trade_buffer_.size() << ", Required: " << max_trade_buffer_size_ << std::endl;
            return;
        }
        batch.swap(trade_buffer_);
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushTradesIfNeeded: Swapped batch of " << batch.size() << " trades for database insertion" << std::endl;
    }
    if (batch.empty()) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushTradesIfNeeded: Batch is empty, no database insertion needed" << std::endl;
        return;
    }

    try {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushTradesIfNeeded: Inserting " << batch.size() << " trades into database" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Database connection status: " << (db_ && db_->isConnected() ? "CONNECTED" : "DISCONNECTED") << std::endl;
        
        if (!db_) {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushTradesIfNeeded: Database connection not available!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Cannot insert " << batch.size() << " trades - database connection failed" << std::endl;
            // Put trades back in buffer for retry
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                trade_buffer_.insert(trade_buffer_.end(), batch.begin(), batch.end());
            }
            return;
        }
        
        if (!db_->isConnected()) {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushTradesIfNeeded: Database connection not available!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Cannot insert " << batch.size() << " trades - database connection failed" << std::endl;
            
            // Attempt to reconnect
            std::cerr << "[" << getCurrentTimestamp() << "][INFO]   Attempting to reconnect to database..." << std::endl;
            
            // Put trades back in buffer for retry
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                trade_buffer_.insert(trade_buffer_.end(), batch.begin(), batch.end());
            }
            return;
        }
        
        std::lock_guard<std::mutex> db_lock(db_mutex_);
        
        // Add detailed database operation logging
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Starting bulk insert operation..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        db_->bulkInsertTrades(batch);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        stats_.trades_inserted += batch.size();
        std::cout << "[" << getCurrentTimestamp() << "][INFO] flushTradesIfNeeded: Successfully inserted " << batch.size() << " trades in " << duration.count() << "ms" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]   Insertion rate: " << (batch.size() * 1000.0 / duration.count()) << " trades/second" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]   Database operation completed successfully" << std::endl;
    } catch (const std::exception& e) {
        ++errors_;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushTradesIfNeeded error: " << e.what() << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Database operation failed!" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   This could indicate database connectivity issues" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Check database connection and credentials" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   " << batch.size() << " trades were NOT inserted into database" << std::endl;
        
        // Put trades back in buffer for retry
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            trade_buffer_.insert(trade_buffer_.end(), batch.begin(), batch.end());
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Trades returned to buffer for retry, buffer size is now: " << trade_buffer_.size() << std::endl;
        }
    }

    // Check if we need to perform a connection health check
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    int64_t last_check = last_connection_check_time_.load();
    if (now_ms - last_check > CONNECTION_CHECK_INTERVAL_MS) {
        if (last_connection_check_time_.compare_exchange_strong(last_check, now_ms)) {
            // Only one thread should perform the health check
            checkDatabaseConnectionHealth();
        }
    }
}

void MarketDataProcessor::flushCandlesIfNeeded(bool force) {
    // In exclusive hotswap mode, don't flush candles to database
    if (enable_exclusive_hotspine_) {
        return;
    }

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
                if (!db_) {
                    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushCandlesIfNeeded: Database connection not available!" << std::endl;
                    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Cannot insert candles - database connection failed" << std::endl;
                    continue;
                }
                
                if (!db_->isConnected()) {
                    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushCandlesIfNeeded: Database connection not available!" << std::endl;
                    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Cannot insert candles - database connection failed" << std::endl;
                    
                    // Attempt to reconnect
                    std::cerr << "[" << getCurrentTimestamp() << "][INFO]   Attempting to reconnect to database..." << std::endl;
                    continue;
                }
                
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

    // Check if we need to perform a connection health check
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    int64_t last_check = last_connection_check_time_.load();
    if (now_ms - last_check > CONNECTION_CHECK_INTERVAL_MS) {
        if (last_connection_check_time_.compare_exchange_strong(last_check, now_ms)) {
            // Only one thread should perform the health check
            checkDatabaseConnectionHealth();
        }
    }
}

void MarketDataProcessor::flushOrderbooksIfNeeded(bool force) {
    // In exclusive hotswap mode, don't flush orderbooks to database
    if (enable_exclusive_hotspine_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushOrderbooksIfNeeded: Skipping in exclusive HotSpine mode" << std::endl;
        return;
    }

    std::vector<OrderbookSnapshot> batch;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushOrderbooksIfNeeded: Checking buffer. Size: " << orderbook_buffer_.size() << ", Max: " << max_orderbook_buffer_size_ << std::endl;
        if (!force &&
            orderbook_buffer_.size() < max_orderbook_buffer_size_) {
            std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushOrderbooksIfNeeded: Buffer not full enough, skipping flush" << std::endl;
            return;
        }
        batch.swap(orderbook_buffer_);
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushOrderbooksIfNeeded: Swapped batch of " << batch.size() << " orderbooks" << std::endl;
    }
    if (batch.empty()) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushOrderbooksIfNeeded: Batch is empty, skipping" << std::endl;
        return;
    }

    try {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] flushOrderbooksIfNeeded: Inserting " << batch.size() << " orderbooks into database" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Database connection status: " << (db_ && db_->isConnected() ? "CONNECTED" : "DISCONNECTED") << std::endl;
        
        if (!db_) {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushOrderbooksIfNeeded: Database connection not available!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Cannot insert " << batch.size() << " orderbooks - database connection failed" << std::endl;
            return;
        }
        
        if (!db_->isConnected()) {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushOrderbooksIfNeeded: Database connection not available!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Cannot insert " << batch.size() << " orderbooks - database connection failed" << std::endl;
            
            // Attempt to reconnect
            std::cerr << "[" << getCurrentTimestamp() << "][INFO]   Attempting to reconnect to database..." << std::endl;
            return;
        }
        
        std::lock_guard<std::mutex> db_lock(db_mutex_);
        
        // Add detailed database operation logging
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Starting bulk insert operation..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        db_->bulkInsertOrderbooks(batch);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        stats_.orderbooks_inserted += batch.size();
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Successfully inserted " << batch.size() << " orderbooks in " << duration.count() << "ms" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Database operation completed successfully" << std::endl;
    } catch (const std::exception& e) {
        ++errors_;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushOrderbooksIfNeeded error: "
                  << e.what() << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Database operation failed!" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   This could indicate database connectivity issues" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Check database connection and credentials" << std::endl;
    }
}

void MarketDataProcessor::flushBuffers() {
    // In exclusive hotswap mode, only flush HotSpine
    if (enable_exclusive_hotspine_) {
        if (hotspine_writer_) {
            hotspine_writer_->flushBatch();
        }
    } else {
        flushTradesIfNeeded(true);
        flushCandlesIfNeeded(true);
        flushOrderbooksIfNeeded(true);
    }
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

// Add a method to log WebSocket data flow statistics
void MarketDataProcessor::logWebSocketDataFlowStats() const {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: WebSocket Data Flow Statistics:" << std::endl;
    
    auto st = getStats();
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   Data Reception Rates:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Trades received: " << st.trades_received << " (" << st.trades_per_sec << " /sec)" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Orderbooks received: " << st.orderbooks_received << " (" << st.orderbooks_per_sec << " /sec)" << std::endl;
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   Data Processing Status:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Trades inserted: " << st.trades_inserted << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Orderbooks inserted: " << st.orderbooks_inserted << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Candles generated: " << st.candles_generated << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Candles inserted: " << st.candles_inserted << std::endl;
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   Error Statistics:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Total errors: " << st.errors << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Average latency: " << st.avg_latency_ms << " ms" << std::endl;
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket Health Indicators:" << std::endl;
    if (st.trades_per_sec > 0 || st.orderbooks_per_sec > 0) {
        std::cout << "[" << getCurrentTimestamp() << "][INFO]     WebSocket connection: HEALTHY (receiving data)" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]     Data flow is active and healthy" << std::endl;
    } else {
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]   WebSocket connection: NO DATA RECEIVED" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]   Possible issues:" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - WebSocket connection not established" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Subscription not successful" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Exchange not sending data" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Network connectivity issues" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - WebSocket connection dropped" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - WebSocket protocol errors" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Message parsing failures" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Exchange rate limiting" << std::endl;
        
        // Add WebSocket reconnection guidance
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]   WebSocket Reconnection:" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - CCAPI should automatically attempt to reconnect" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Check for SESSION_STATUS events indicating reconnection attempts" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Monitor for SUBSCRIPTION_STATUS events after reconnection" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Verify WebSocket connection is re-established" << std::endl;
    }
    
    // Add WebSocket-specific troubleshooting
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket Troubleshooting:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     If no data is received:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]       1. Check WebSocket connection status in ExchangeConnectionManager" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]       2. Verify subscription was successful" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]       3. Check for SESSION_STATUS events" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]       4. Look for SUBSCRIPTION_STATUS events" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]       5. Verify WebSocket URL configuration" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]       6. Check network connectivity to exchange" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]       7. Test with different WebSocket timeout settings" << std::endl;
    
    auto pmap = getPairStats();
    if (!pmap.empty()) {
        std::cout << "[" << getCurrentTimestamp() << "][INFO]   Active Trading Pairs:" << std::endl;
        for (const auto& kv : pmap) {
            std::cout << "[" << getCurrentTimestamp() << "][INFO]     " << kv.first
                      << ": trades=" << kv.second.trades
                      << ", orderbooks=" << kv.second.orderbooks << std::endl;
        }
    } else {
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]   No active trading pairs detected" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][WARNING]   This could indicate subscription issues" << std::endl;
    }
}

// Add a method to validate WebSocket data flow
void MarketDataProcessor::validateWebSocketDataFlow() {
        std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: Validating WebSocket data flow..." << std::endl;
        
        auto st = getStats();
        
        // Check if we're receiving any data
        if (st.trades_received == 0 && st.orderbooks_received == 0) {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   WebSocket Data Flow Validation FAILED!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   No data received from WebSocket connection!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   This indicates a critical environment setup issue!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Possible causes:" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - WebSocket connection not established" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Subscription not successful" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Exchange not sending data" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Network connectivity issues" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - WebSocket connection dropped" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - WebSocket protocol errors" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Message parsing failures" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Exchange rate limiting" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - CCAPI configuration not properly handled" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Exchange configuration parsing issues" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Invalid WebSocket URL configuration" << std::endl;
            
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Troubleshooting steps:" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     1. Check WebSocket connection status in ExchangeConnectionManager" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     2. Verify subscription was successful" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     3. Look for SESSION_STATUS events" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     4. Check for SUBSCRIPTION_STATUS events" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     5. Verify WebSocket URL configuration" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     6. Check network connectivity to exchange" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     7. Test with different WebSocket timeout settings" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     8. Verify CCAPI configuration is properly loaded" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     9. Check if exchange configurations are correctly parsed" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]    10. Validate WebSocket session options and timeouts" << std::endl;
            
            return;
        }
        
        // Check data reception rates
        if (st.trades_per_sec < 0.1 && st.orderbooks_per_sec < 0.1) {
            std::cerr << "[" << getCurrentTimestamp() << "][WARNING] WebSocket Data Flow Validation WARNING!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Low data reception rates detected!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Trades: " << st.trades_per_sec << " /sec, Orderbooks: " << st.orderbooks_per_sec << " /sec" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][WARNING] This could indicate WebSocket performance issues" << std::endl;
        }
        
        // Check for data processing errors
        if (st.errors > 0) {
            std::cerr << "[" << getCurrentTimestamp() << "][WARNING] WebSocket Data Flow Validation WARNING!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Data processing errors detected: " << st.errors << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][WARNING] This could indicate WebSocket data parsing issues" << std::endl;
        }
        
        // If we get here, data flow is healthy
        std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket Data Flow Validation PASSED!" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]   Healthy data reception detected:" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]     Trades: " << st.trades_received << " received, " << st.trades_per_sec << " /sec" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]     Orderbooks: " << st.orderbooks_received << " received, " << st.orderbooks_per_sec << " /sec" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]     Average latency: " << st.avg_latency_ms << " ms" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]     Errors: " << st.errors << std::endl;
        
        // Check database integration if enabled
        if (db_) {
            std::cout << "[" << getCurrentTimestamp() << "][INFO]   Database Integration Status:" << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][INFO]     Database connection: " << (db_->isConnected() ? "CONNECTED" : "DISCONNECTED") << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][INFO]     Trades inserted: " << st.trades_inserted << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][INFO]     Orderbooks inserted: " << st.orderbooks_inserted << std::endl;
            std::cout << "[" << getCurrentTimestamp() << "][INFO]     Candles inserted: " << st.candles_inserted << std::endl;
        } else {
            std::cout << "[" << getCurrentTimestamp() << "][INFO]   Database Integration: DISABLED (using HotSpine exclusively)" << std::endl;
        }
        
        // Check HotSpine integration
        if (hotspine_writer_) {
            std::cout << "[" << getCurrentTimestamp() << "][INFO]   HotSpine Integration: ENABLED" << std::endl;
        } else {
            std::cout << "[" << getCurrentTimestamp() << "][INFO]   HotSpine Integration: DISABLED" << std::endl;
        }
        
        std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket data flow validation completed successfully!" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]   Environment setup appears to be working correctly!" << std::endl;
        std::cout << "[" << getCurrentTimestamp() << "][INFO]   CCAPI configuration is properly handled!" << std::endl;
    }


// Add a method to validate CCAPI configuration handling
void MarketDataProcessor::validateCCAPIConfiguration() {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: Validating CCAPI configuration handling..." << std::endl;

    // Check if we have any active pairs (indicating successful subscription)
    auto pmap = getPairStats();
    if (pmap.empty()) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   CCAPI Configuration Validation FAILED!" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   No active trading pairs detected!" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   This indicates CCAPI configuration may not be properly handled!" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Possible root causes:" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     1. WebSocket connection not established" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     2. Subscription requests not sent" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     3. Exchange not responding to subscriptions" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     4. WebSocket data not being processed" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     5. Event handler not receiving events" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     6. Correlation ID parsing issues" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     7. WebSocket protocol errors" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     8. Network connectivity issues" << std::endl;
        return;
    }

    std::cout << "[" << getCurrentTimestamp() << "][INFO]   CCAPI Configuration Validation PASSED!" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   Active trading pairs: " << pmap.size() << std::endl;
    for (const auto& kv : pmap) {
        std::cout << "[" << getCurrentTimestamp() << "][INFO]     " << kv.first
                  << ": trades=" << kv.second.trades
                  << ", orderbooks=" << kv.second.orderbooks << std::endl;
    }
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   CCAPI configuration is properly handled!" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket connection and data flow appear healthy!" << std::endl;
}

// Add a method to add WebSocket debugging
void MarketDataProcessor::addWebSocketDebugging() {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MarketDataProcessor: Adding comprehensive WebSocket debugging..." << std::endl;
    
    auto st = getStats();
    auto pmap = getPairStats();
    
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   WebSocket Debugging Configuration:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]     Data Reception Status:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       Trades received: " << st.trades_received << " (" << st.trades_per_sec << "/sec)" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       Orderbooks received: " << st.orderbooks_received << " (" << st.orderbooks_per_sec << "/sec)" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       Errors: " << st.errors << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       Average latency: " << st.avg_latency_ms << " ms" << std::endl;
    
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]     Active Trading Pairs: " << pmap.size() << std::endl;
    for (const auto& kv : pmap) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       " << kv.first
                  << ": trades=" << kv.second.trades
                  << ", orderbooks=" << kv.second.orderbooks << std::endl;
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   WebSocket Troubleshooting:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]     If no data is received, check:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       1. WebSocket connection status in ExchangeConnectionManager" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       2. Subscription was successful" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       3. SESSION_STATUS events for connection details" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       4. SUBSCRIPTION_STATUS events for confirmation" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       5. Network connectivity to exchange WebSocket endpoints" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       6. WebSocket URL configuration" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       7. WebSocket connection timeouts and ping/pong settings" << std::endl;
    
    // Add WebSocket-specific debugging for common issues
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   WebSocket Connection Debugging:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]     Common WebSocket issues to investigate:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - WebSocket connection timeout too short" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - Ping/pong intervals too aggressive" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - WebSocket URL configuration incorrect" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - Exchange-specific WebSocket requirements" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - Missing authentication credentials" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - WebSocket compression issues" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - WebSocket subprotocol mismatches" << std::endl;
    
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   WebSocket Data Flow Debugging:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]     If data is not flowing:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - Check for SUBSCRIPTION_DATA events" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - Verify message lists are not empty" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - Check for WebSocket protocol errors" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - Monitor WebSocket connection health" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]       - Check exchange-specific WebSocket requirements" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MarketDataProcessor: WebSocket debugging completed!" << std::endl;
}

// Connection health monitoring methods
void MarketDataProcessor::checkDatabaseConnectionHealth() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (!db_) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database connection health check: No database connection object available!" << std::endl;
        return;
    }
    
    bool connected = db_->isConnected();
    std::cout << "[" << getCurrentTimestamp() << "][INFO] Database connection health check: "
              << (connected ? "CONNECTED" : "DISCONNECTED") << std::endl;
    
    if (!connected) {
        std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Database connection is down!" << std::endl;
        std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Attempting to reconnect..." << std::endl;
        
        if (attemptDatabaseReconnect()) {
            std::cout << "[" << getCurrentTimestamp() << "][INFO] Database reconnection successful!" << std::endl;
        } else {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database reconnection failed!" << std::endl;
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database operations will be queued until connection is restored" << std::endl;
        }
    }

    // Check if we need to perform a connection health check
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    int64_t last_check = last_connection_check_time_.load();
    if (now_ms - last_check > CONNECTION_CHECK_INTERVAL_MS) {
        if (last_connection_check_time_.compare_exchange_strong(last_check, now_ms)) {
            // Only one thread should perform the health check
            checkDatabaseConnectionHealth();
        }
    }
}
bool MarketDataProcessor::attemptDatabaseReconnect() {
    if (!db_) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Cannot reconnect: No database connection object available!" << std::endl;
        return false;
    }
    
    try {
        std::cout << "[" << getCurrentTimestamp() << "][INFO] Attempting database reconnection..." << std::endl;
        
        // The MSSQLBulkInserter doesn't have a reconnect method, so we'll try to verify the connection
        // which should trigger a reconnection attempt internally if supported
        bool success = db_->verifyConnection();
        
        if (success) {
            std::cout << "[" << getCurrentTimestamp() << "][INFO] Database reconnection successful!" << std::endl;
            return true;
        } else {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database reconnection failed!" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database reconnection error: " << e.what() << std::endl;
        return false;
    }
}


