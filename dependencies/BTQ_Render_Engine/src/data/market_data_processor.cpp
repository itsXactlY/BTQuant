#include "market_data_processor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace BTQuant {
namespace RenderEngine {

MarketDataProcessor::MarketDataProcessor()
    : vwap_window_size_(100)
    , momentum_window_size_(20)
    , volatility_window_size_(50)
    , spread_analysis_window_(10)
{
    std::cout << "[MarketDataProcessor] Initialized with default parameters" << std::endl;
}

MarketDataProcessor::~MarketDataProcessor() = default;

void MarketDataProcessor::processTradeUpdate(const MarketDataUpdate& update) {
    if (update.type != MarketDataType::TRADE) {
        return;
    }
    
    std::lock_guard lock(data_mutex_);
    
    auto& symbol_data = symbol_analytics_[update.symbol_id];
    
    // Update basic trade data
    TradeData trade;
    trade.timestamp_us = update.timestamp_us;
    trade.price = update.price;
    trade.size = update.size;
    trade.is_buy = (update.side == "buy");
    
    symbol_data.recent_trades.push_back(trade);
    
    // Maintain window size
    if (symbol_data.recent_trades.size() > vwap_window_size_ * 2) {
        symbol_data.recent_trades.erase(
            symbol_data.recent_trades.begin(),
            symbol_data.recent_trades.begin() + vwap_window_size_
        );
    }
    
    // Update analytics
    updateVWAP(symbol_data);
    updateMomentum(symbol_data);
    updateVolatility(symbol_data);
    updateTradingMetrics(symbol_data, trade);
    
    // Update performance metrics
    performance_metrics_.total_trades_processed++;
    performance_metrics_.last_update_time = std::chrono::high_resolution_clock::now();
}

void MarketDataProcessor::processOrderbookUpdate(const MarketDataUpdate& update) {
    if (update.type != MarketDataType::ORDERBOOK) {
        return;
    }
    
    std::lock_guard lock(data_mutex_);
    
    auto& symbol_data = symbol_analytics_[update.symbol_id];
    
    // Update orderbook data
    OrderbookData orderbook;
    orderbook.timestamp_us = update.timestamp_us;
    orderbook.bids = update.bids;
    orderbook.asks = update.asks;
    
    // Calculate spread and depth
    if (!orderbook.bids.empty() && !orderbook.asks.empty()) {
        orderbook.spread = orderbook.asks[0].price - orderbook.bids[0].price;
        orderbook.spread_percent = (orderbook.spread / orderbook.bids[0].price) * 100.0;
        
        // Calculate market depth
        orderbook.bid_depth = calculateMarketDepth(orderbook.bids);
        orderbook.ask_depth = calculateMarketDepth(orderbook.asks);
        orderbook.total_depth = orderbook.bid_depth + orderbook.ask_depth;
        
        // Calculate imbalance
        orderbook.imbalance = (orderbook.bid_depth - orderbook.ask_depth) / 
                             std::max(orderbook.total_depth, 0.001);
    }
    
    symbol_data.recent_orderbooks.push_back(orderbook);
    
    // Maintain window size
    if (symbol_data.recent_orderbooks.size() > spread_analysis_window_ * 2) {
        symbol_data.recent_orderbooks.erase(
            symbol_data.recent_orderbooks.begin(),
            symbol_data.recent_orderbooks.begin() + spread_analysis_window_
        );
    }
    
    // Update spread analytics
    updateSpreadAnalysis(symbol_data);
    
    // Update performance metrics
    performance_metrics_.total_orderbooks_processed++;
}

SymbolAnalytics MarketDataProcessor::getSymbolAnalytics(uint32_t symbol_id) const {
    std::lock_guard lock(data_mutex_);
    
    auto it = symbol_analytics_.find(symbol_id);
    if (it != symbol_analytics_.end()) {
        return it->second;
    }
    
    return SymbolAnalytics{};  // Return empty analytics if not found
}

std::vector<uint32_t> MarketDataProcessor::getActiveSymbols() const {
    std::lock_guard lock(data_mutex_);
    
    std::vector<uint32_t> symbols;
    symbols.reserve(symbol_analytics_.size());
    
    for (const auto& [symbol_id, _] : symbol_analytics_) {
        symbols.push_back(symbol_id);
    }
    
    return symbols;
}

ProcessorPerformanceMetrics MarketDataProcessor::getPerformanceMetrics() const {
    std::lock_guard lock(data_mutex_);
    return performance_metrics_;
}

void MarketDataProcessor::clearSymbolData(uint32_t symbol_id) {
    std::lock_guard lock(data_mutex_);
    symbol_analytics_.erase(symbol_id);
}

void MarketDataProcessor::clearAllData() {
    std::lock_guard lock(data_mutex_);
    symbol_analytics_.clear();
    performance_metrics_ = ProcessorPerformanceMetrics{};
}

void MarketDataProcessor::setVWAPWindow(size_t window_size) {
    vwap_window_size_ = std::max(size_t(10), std::min(window_size, size_t(1000)));
}

void MarketDataProcessor::setMomentumWindow(size_t window_size) {
    momentum_window_size_ = std::max(size_t(5), std::min(window_size, size_t(100)));
}

void MarketDataProcessor::setVolatilityWindow(size_t window_size) {
    volatility_window_size_ = std::max(size_t(10), std::min(window_size, size_t(200)));
}

void MarketDataProcessor::updateVWAP(SymbolAnalytics& symbol_data) {
    if (symbol_data.recent_trades.empty()) {
        return;
    }
    
    // Calculate VWAP over the window
    double total_volume = 0.0;
    double total_value = 0.0;
    
    size_t start_idx = symbol_data.recent_trades.size() > vwap_window_size_ ?
                      symbol_data.recent_trades.size() - vwap_window_size_ : 0;
    
    for (size_t i = start_idx; i < symbol_data.recent_trades.size(); ++i) {
        const auto& trade = symbol_data.recent_trades[i];
        total_volume += trade.size;
        total_value += trade.price * trade.size;
    }
    
    if (total_volume > 0) {
        symbol_data.vwap = total_value / total_volume;
        
        // Calculate VWAP deviation
        double current_price = symbol_data.recent_trades.back().price;
        symbol_data.vwap_deviation = ((current_price - symbol_data.vwap) / symbol_data.vwap) * 100.0;
    }
    
    // Update volume metrics
    symbol_data.volume_1m = calculateVolumeInWindow(symbol_data.recent_trades, 60 * 1000000ULL);  // 1 minute
    symbol_data.volume_5m = calculateVolumeInWindow(symbol_data.recent_trades, 5 * 60 * 1000000ULL);  // 5 minutes
    symbol_data.volume_15m = calculateVolumeInWindow(symbol_data.recent_trades, 15 * 60 * 1000000ULL);  // 15 minutes
}

void MarketDataProcessor::updateMomentum(SymbolAnalytics& symbol_data) {
    if (symbol_data.recent_trades.size() < 2) {
        return;
    }
    
    // Calculate price changes over the momentum window
    std::vector<double> price_changes;
    size_t start_idx = symbol_data.recent_trades.size() > momentum_window_size_ ?
                      symbol_data.recent_trades.size() - momentum_window_size_ : 0;
    
    for (size_t i = start_idx + 1; i < symbol_data.recent_trades.size(); ++i) {
        double prev_price = symbol_data.recent_trades[i-1].price;
        double curr_price = symbol_data.recent_trades[i].price;
        
        if (prev_price > 0) {
            double change = ((curr_price - prev_price) / prev_price) * 100.0;
            price_changes.push_back(change);
        }
    }
    
    if (!price_changes.empty()) {
        // Calculate momentum as average price change
        symbol_data.momentum = std::accumulate(price_changes.begin(), price_changes.end(), 0.0) / price_changes.size();
        
        // Calculate momentum strength (standard deviation of changes)
        double mean = symbol_data.momentum;
        double variance = 0.0;
        for (double change : price_changes) {
            variance += (change - mean) * (change - mean);
        }
        symbol_data.momentum_strength = std::sqrt(variance / price_changes.size());
    }
    
    // Update price extremes
    auto minmax = std::minmax_element(
        symbol_data.recent_trades.begin() + start_idx,
        symbol_data.recent_trades.end(),
        [](const TradeData& a, const TradeData& b) { return a.price < b.price; }
    );
    
    if (minmax.first != symbol_data.recent_trades.end()) {
        symbol_data.price_min = minmax.first->price;
        symbol_data.price_max = minmax.second->price;
        
        double current_price = symbol_data.recent_trades.back().price;
        if (symbol_data.price_max > symbol_data.price_min) {
            symbol_data.price_position = ((current_price - symbol_data.price_min) / 
                                        (symbol_data.price_max - symbol_data.price_min)) * 100.0;
        }
    }
}

void MarketDataProcessor::updateVolatility(SymbolAnalytics& symbol_data) {
    if (symbol_data.recent_trades.size() < volatility_window_size_) {
        return;
    }
    
    // Calculate returns over the volatility window
    std::vector<double> returns;
    size_t start_idx = symbol_data.recent_trades.size() - volatility_window_size_;
    
    for (size_t i = start_idx + 1; i < symbol_data.recent_trades.size(); ++i) {
        double prev_price = symbol_data.recent_trades[i-1].price;
        double curr_price = symbol_data.recent_trades[i].price;
        
        if (prev_price > 0) {
            double return_val = std::log(curr_price / prev_price);
            returns.push_back(return_val);
        }
    }
    
    if (returns.size() >= 2) {
        // Calculate mean return
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        
        // Calculate variance
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= (returns.size() - 1);
        
        // Annualized volatility (assuming trades are roughly evenly spaced)
        symbol_data.volatility = std::sqrt(variance) * std::sqrt(252 * 24 * 60);  // Rough annualization
        
        // Calculate Sharpe-like ratio (return/volatility)
        if (symbol_data.volatility > 0) {
            symbol_data.sharpe_ratio = mean_return / symbol_data.volatility;
        }
    }
}

void MarketDataProcessor::updateTradingMetrics(SymbolAnalytics& symbol_data, const TradeData& trade) {
    // Update trade counts
    if (trade.is_buy) {
        symbol_data.buy_volume += trade.size;
        symbol_data.buy_count++;
    } else {
        symbol_data.sell_volume += trade.size;
        symbol_data.sell_count++;
    }
    
    // Calculate buy/sell ratio
    double total_volume = symbol_data.buy_volume + symbol_data.sell_volume;
    if (total_volume > 0) {
        symbol_data.buy_sell_ratio = symbol_data.buy_volume / total_volume;
    }
    
    // Update trade size statistics
    symbol_data.avg_trade_size = (symbol_data.avg_trade_size * (symbol_data.trade_count - 1) + trade.size) / symbol_data.trade_count;
    symbol_data.trade_count++;
    
    // Track large trades (> 2x average)
    if (trade.size > symbol_data.avg_trade_size * 2.0) {
        symbol_data.large_trade_count++;
    }
    
    // Update last trade info
    symbol_data.last_trade_price = trade.price;
    symbol_data.last_trade_size = trade.size;
    symbol_data.last_trade_time = trade.timestamp_us;
}

void MarketDataProcessor::updateSpreadAnalysis(SymbolAnalytics& symbol_data) {
    if (symbol_data.recent_orderbooks.empty()) {
        return;
    }
    
    // Calculate average spread over window
    double total_spread = 0.0;
    double total_spread_percent = 0.0;
    double total_imbalance = 0.0;
    size_t valid_books = 0;
    
    size_t start_idx = symbol_data.recent_orderbooks.size() > spread_analysis_window_ ?
                      symbol_data.recent_orderbooks.size() - spread_analysis_window_ : 0;
    
    for (size_t i = start_idx; i < symbol_data.recent_orderbooks.size(); ++i) {
        const auto& book = symbol_data.recent_orderbooks[i];
        if (book.spread > 0) {
            total_spread += book.spread;
            total_spread_percent += book.spread_percent;
            total_imbalance += book.imbalance;
            valid_books++;
        }
    }
    
    if (valid_books > 0) {
        symbol_data.avg_spread = total_spread / valid_books;
        symbol_data.avg_spread_percent = total_spread_percent / valid_books;
        symbol_data.avg_imbalance = total_imbalance / valid_books;
        
        // Current spread from latest orderbook
        const auto& latest = symbol_data.recent_orderbooks.back();
        symbol_data.current_spread = latest.spread;
        symbol_data.current_spread_percent = latest.spread_percent;
        symbol_data.current_imbalance = latest.imbalance;
        
        // Market depth
        symbol_data.market_depth = latest.total_depth;
    }
}

double MarketDataProcessor::calculateMarketDepth(const std::vector<PriceLevel>& levels) const {
    double total_depth = 0.0;
    
    // Calculate depth for first 5 levels (or all if less than 5)
    size_t max_levels = std::min(levels.size(), size_t(5));
    for (size_t i = 0; i < max_levels; ++i) {
        total_depth += levels[i].price * levels[i].size;
    }
    
    return total_depth;
}

double MarketDataProcessor::calculateVolumeInWindow(const std::vector<TradeData>& trades, uint64_t window_us) const {
    if (trades.empty()) {
        return 0.0;
    }
    
    uint64_t current_time = trades.back().timestamp_us;
    uint64_t window_start = current_time - window_us;
    
    double volume = 0.0;
    for (auto it = trades.rbegin(); it != trades.rend(); ++it) {
        if (it->timestamp_us < window_start) {
            break;
        }
        volume += it->size;
    }
    
    return volume;
}

std::vector<SymbolRanking> MarketDataProcessor::getRankings(RankingCriteria criteria, size_t limit) const {
    std::lock_guard lock(data_mutex_);
    
    std::vector<SymbolRanking> rankings;
    rankings.reserve(symbol_analytics_.size());
    
    // Create rankings based on criteria
    for (const auto& [symbol_id, analytics] : symbol_analytics_) {
        SymbolRanking ranking;
        ranking.symbol_id = symbol_id;
        
        switch (criteria) {
            case RankingCriteria::VOLUME:
                ranking.value = analytics.volume_1m;
                ranking.label = "Volume (1m)";
                break;
            case RankingCriteria::MOMENTUM:
                ranking.value = std::abs(analytics.momentum);
                ranking.label = "Momentum";
                break;
            case RankingCriteria::VOLATILITY:
                ranking.value = analytics.volatility;
                ranking.label = "Volatility";
                break;
            case RankingCriteria::SPREAD:
                ranking.value = analytics.avg_spread_percent;
                ranking.label = "Spread %";
                break;
            case RankingCriteria::IMBALANCE:
                ranking.value = std::abs(analytics.current_imbalance);
                ranking.label = "Imbalance";
                break;
        }
        
        rankings.push_back(ranking);
    }
    
    // Sort by value (descending)
    std::sort(rankings.begin(), rankings.end(),
              [](const SymbolRanking& a, const SymbolRanking& b) {
                  return a.value > b.value;
              });
    
    // Limit results
    if (limit > 0 && rankings.size() > limit) {
        rankings.resize(limit);
    }
    
    return rankings;
}

MarketSummary MarketDataProcessor::getMarketSummary() const {
    std::lock_guard lock(data_mutex_);
    
    MarketSummary summary;
    summary.total_symbols = symbol_analytics_.size();
    
    if (symbol_analytics_.empty()) {
        return summary;
    }
    
    // Aggregate statistics
    double total_volume = 0.0;
    double total_momentum = 0.0;
    double total_volatility = 0.0;
    size_t active_symbols = 0;
    
    for (const auto& [symbol_id, analytics] : symbol_analytics_) {
        if (analytics.trade_count > 0) {
            active_symbols++;
            total_volume += analytics.volume_1m;
            total_momentum += analytics.momentum;
            total_volatility += analytics.volatility;
            
            // Count trending symbols
            if (std::abs(analytics.momentum) > 1.0) {  // > 1% momentum
                if (analytics.momentum > 0) {
                    summary.trending_up++;
                } else {
                    summary.trending_down++;
                }
            }
        }
    }
    
    summary.active_symbols = active_symbols;
    
    if (active_symbols > 0) {
        summary.avg_volume = total_volume / active_symbols;
        summary.avg_momentum = total_momentum / active_symbols;
        summary.avg_volatility = total_volatility / active_symbols;
    }
    
    summary.last_update = performance_metrics_.last_update_time;
    
    return summary;
}

} // namespace RenderEngine
} // namespace BTQuant