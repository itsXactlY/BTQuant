#include "hotspine_extended_reader.hpp"
#include "utils/dynamic_logger.hpp"
#include <sys/time.h>
#include <algorithm>

namespace BTQuant {

HotSpineExtendedReader::HotSpineExtendedReader(const std::string& shm_name)
    : registry_(SymbolRegistry::instance()) {
    reader_ = std::make_unique<HotSpineReader>(shm_name);
    
    // Log initialization with SHM name for debugging
    auto& logger = Logging::DynamicLogger::instance();
    logger.info("HotSpineExtendedReader", "[HotSpineExtendedReader] Initialized with SHM: " + shm_name);
}

HotSpineExtendedReader::~HotSpineExtendedReader() = default;

bool HotSpineExtendedReader::load_symbol_mappings(const std::string& filepath) {
    return registry_.load_from_file(filepath);
}

std::string HotSpineExtendedReader::make_cache_key(const std::string& exchange,
                                                   const std::string& symbol) const {
    return exchange + ":" + symbol;
}

TradeData HotSpineExtendedReader::resolve_trade(const HotTrade& trade) {
    TradeData data;
    data.ts_exchange = trade.ts_exchange;
    data.ts_local = trade.ts_local;
    data.price = trade.price;
    data.size = trade.size;
    data.symbol_id = trade.symbol_id;
    data.side = trade.side;
    
    if (auto info = registry_.get_symbol_info(trade.symbol_id)) {
        data.exchange = info->exchange;
        data.symbol = info->symbol;
    } else {
        data.exchange = "unknown";
        data.symbol = "symbol_" + std::to_string(trade.symbol_id);
    }
    
    return data;
}

OrderbookData HotSpineExtendedReader::resolve_orderbook(const HotOrderbookSnapshot& snapshot) {
    OrderbookData data;
    data.ts_exchange = snapshot.ts_exchange;
    data.ts_local = snapshot.ts_local;
    data.symbol_id = snapshot.symbol_id;
    
    if (auto info = registry_.get_symbol_info(snapshot.symbol_id)) {
        data.exchange = info->exchange;
        data.symbol = info->symbol;
    } else {
        data.exchange = "unknown";
        data.symbol = "symbol_" + std::to_string(snapshot.symbol_id);
    }
    
    // Copy bids
    for (uint8_t i = 0; i < snapshot.bids_count; ++i) {
        data.bids.emplace_back(snapshot.bids[i].price, snapshot.bids[i].size);
    }
    
    // Copy asks
    for (uint8_t i = 0; i < snapshot.asks_count; ++i) {
        data.asks.emplace_back(snapshot.asks[i].price, snapshot.asks[i].size);
    }
    
    return data;
}

std::optional<TradeData> HotSpineExtendedReader::poll_trade() {
    if (!reader_) {
        return std::nullopt;
    }
    
    HotTrade trade;
    if (!reader_->pollTrade(trade)) {
        return std::nullopt;
    }
    
    TradeData data = resolve_trade(trade);
    update_caches(data);
    
    // Increment stats counter
    trades_read_++;
    
    // Log first few trades for debugging
    if (trades_read_ <= 5) {
        auto& logger = Logging::DynamicLogger::instance();
        logger.info("HotSpineExtendedReader", "[HotSpineExtendedReader] Trade #" + std::to_string(trades_read_) + 
                    " " + data.exchange + ":" + data.symbol + 
                    " @ " + std::to_string(data.price) + 
                    " (size: " + std::to_string(data.size) + ")");
    }
    
    return data;
}

std::vector<TradeData> HotSpineExtendedReader::read_all_trades() {
    std::vector<TradeData> trades;
    
    while (auto trade = poll_trade()) {
        trades.push_back(*trade);
    }
    
    return trades;
}

std::optional<OrderbookData> HotSpineExtendedReader::poll_orderbook() {
    if (!reader_) return std::nullopt;
    
    HotOrderbookSnapshot snapshot;
    if (!reader_->pollOrderbook(snapshot)) {
        return std::nullopt;
    }
    
    OrderbookData data = resolve_orderbook(snapshot);
    update_caches(data);
    
    return data;
}

std::vector<OrderbookData> HotSpineExtendedReader::read_all_orderbooks() {
    std::vector<OrderbookData> orderbooks;
    
    while (auto orderbook = poll_orderbook()) {
        orderbooks.push_back(*orderbook);
    }
    
    return orderbooks;
}

std::optional<double> HotSpineExtendedReader::get_latest_price(const std::string& exchange,
                                                               const std::string& symbol) {
    std::string key = make_cache_key(exchange, symbol);
    auto it = price_cache_.find(key);
    
    if (it != price_cache_.end()) {
        uint64_t now = get_current_time_us();
        if (!it->second.is_stale(now)) {
            return it->second.price;
        }
    }
    
    return std::nullopt;
}

std::optional<OrderbookData> HotSpineExtendedReader::get_latest_orderbook(const std::string& exchange,
                                                                           const std::string& symbol) {
    std::string key = make_cache_key(exchange, symbol);
    auto it = orderbook_cache_.find(key);
    
    if (it != orderbook_cache_.end()) {
        uint64_t now = get_current_time_us();
        if (!it->second.is_stale(now)) {
            return it->second.data;
        }
    }
    
    return std::nullopt;
}

std::vector<TradeData> HotSpineExtendedReader::get_recent_trades(const std::string& exchange,
                                                                 const std::string& symbol,
                                                                 size_t limit) {
    std::string key = make_cache_key(exchange, symbol);
    auto it = trade_history_.find(key);
    
    if (it == trade_history_.end()) {
        return {};
    }
    
    std::vector<TradeData> result;
    const auto& trades = it->second;
    
    size_t start = (trades.size() > limit) ? (trades.size() - limit) : 0;
    for (size_t i = start; i < trades.size(); ++i) {
        result.push_back(trades[i]);
    }
    
    return result;
}

std::map<std::string, double> HotSpineExtendedReader::get_all_exchange_prices(const std::string& symbol) {
    std::map<std::string, double> prices;
    
    for (const auto& [key, cache] : price_cache_) {
        // Extract exchange from key
        size_t colon_pos = key.find(':');
        if (colon_pos != std::string::npos) {
            std::string exchange = key.substr(0, colon_pos);
            std::string sym = key.substr(colon_pos + 1);
            
            if (sym == symbol) {
                uint64_t now = get_current_time_us();
                if (!cache.is_stale(now)) {
                    prices[exchange] = cache.price;
                }
            }
        }
    }
    
    return prices;
}

std::map<std::string, OrderbookData> HotSpineExtendedReader::get_all_exchange_orderbooks(const std::string& symbol) {
    std::map<std::string, OrderbookData> orderbooks;
    
    for (const auto& [key, cache] : orderbook_cache_) {
        // Extract exchange from key
        size_t colon_pos = key.find(':');
        if (colon_pos != std::string::npos) {
            std::string exchange = key.substr(0, colon_pos);
            std::string sym = key.substr(colon_pos + 1);
            
            if (sym == symbol) {
                uint64_t now = get_current_time_us();
                if (!cache.is_stale(now)) {
                    orderbooks[exchange] = cache.data;
                }
            }
        }
    }
    
    return orderbooks;
}

uint64_t HotSpineExtendedReader::get_current_time_us() const {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000ULL + tv.tv_usec;
}

std::pair<uint64_t, uint64_t> HotSpineExtendedReader::get_buffer_utilization() const {
    if (!reader_ || !reader_->isAttached()) {
        return {0, 0};
    }
    
    auto status = reader_->get_buffer_status();
    auto& logger = Logging::DynamicLogger::instance();
    
    // Log buffer status for monitoring
    if (status.second > 0) {
        double usage_pct = (static_cast<double>(status.first) / status.second) * 100.0;
        logger.info("HotSpineExtendedReader", "[HotSpineExtendedReader] Buffer: " +
                    std::to_string(status.first) + "/" + std::to_string(status.second) +
                    " (" + std::to_string(usage_pct) + "%)");
    }
    
    return status;
}

uint64_t HotSpineExtendedReader::get_lost_count() const {
    // The real lost count would come from shared memory stats
    // For now, estimate based on buffer wrap-around detection
    // In production, this should be queried from shared memory header
    if (!reader_ || !reader_->isAttached()) {
        return 0;
    }
    
    auto status = reader_->get_buffer_status();
    
    // Estimate: if buffer is nearly full and wrapping, trades may be lost
    if (status.second > 0 && status.first >= status.second * 0.95) {
        auto& logger = Logging::DynamicLogger::instance();
        logger.warning("HotSpineExtendedReader", "[HotSpineExtendedReader] Buffer near capacity - potential trade loss");
    }
    
    return 0; // Placeholder - would need shared memory counter
}

void HotSpineExtendedReader::update_caches(const TradeData& trade) {
    std::string key = make_cache_key(trade.exchange, trade.symbol);
    
    price_cache_[key] = {trade.price, trade.ts_local};
    
    // Add to trade history
    auto& history = trade_history_[key];
    history.push_back(trade);
    if (history.size() > max_trade_history_) {
        history.pop_front();
    }
}

void HotSpineExtendedReader::update_caches(const OrderbookData& orderbook) {
    std::string key = make_cache_key(orderbook.exchange, orderbook.symbol);
    
    orderbook_cache_[key] = {orderbook, orderbook.ts_local};
}

void HotSpineExtendedReader::cleanup_stale_caches() {
    uint64_t now = get_current_time_us();
    
    // Cleanup price cache
    for (auto it = price_cache_.begin(); it != price_cache_.end(); ) {
        if (it->second.is_stale(now, cache_max_age_us_)) {
            it = price_cache_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Cleanup orderbook cache
    for (auto it = orderbook_cache_.begin(); it != orderbook_cache_.end(); ) {
        if (it->second.is_stale(now, cache_max_age_us_)) {
            it = orderbook_cache_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace BTQuant
