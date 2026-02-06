#include "../include/data/data_types.hpp"
#include "../include/data/TradeData.h"
#include "../include/data/VolumeDataTypes.h"
#include <vector>
#include <random>
#include <chrono>
#include <string>

namespace BTQuant {
namespace RenderEngine {
namespace TestData {

// Random number generator for mock data
static std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

// Generate mock OHLCVCandle data
std::vector<OHLCVCandle> generateMockCandles(size_t count, uint64_t start_timestamp = 0) {
    std::vector<OHLCVCandle> candles;
    candles.reserve(count);
    
    if (start_timestamp == 0) {
        start_timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    }
    
    std::uniform_real_distribution<double> price_dist(100.0, 200.0);  // Base price range
    std::uniform_real_distribution<double> variation_dist(-2.0, 2.0); // Price variation
    std::uniform_real_distribution<double> volume_dist(100.0, 1000.0); // Volume range
    
    double base_price = price_dist(rng);
    
    for (size_t i = 0; i < count; ++i) {
        double open = base_price;
        double high = open + std::abs(variation_dist(rng));
        double low = open - std::abs(variation_dist(rng));
        double close = (high + low) / 2.0 + variation_dist(rng); // Ensure close is within high/low
        
        // Ensure proper OHLC relationships
        high = std::max({open, close, high});
        low = std::min({open, close, low});
        
        OHLCVCandle candle;
        candle.timestamp = start_timestamp + (i * 60000000); // 1-minute intervals in microseconds
        candle.open = open;
        candle.high = high;
        candle.low = low;
        candle.close = close;
        candle.volume = volume_dist(rng);
        candle.trade_count = static_cast<uint64_t>(std::uniform_int_distribution<int>(10, 100)(rng));
        
        candles.push_back(candle);
        base_price = close; // Use close as next open's base
    }
    
    return candles;
}

// Generate mock TradeData
std::vector<BTQuant::Data::TradeData> generateMockTrades(size_t count) {
    std::vector<BTQuant::Data::TradeData> trades;
    trades.reserve(count);
    
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    std::uniform_real_distribution<double> price_dist(100.0, 200.0);
    std::uniform_real_distribution<float> volume_dist(0.1f, 10.0f);
    std::uniform_int_distribution<uint8_t> exchange_dist(0, 5); // 6 exchanges
    std::uniform_int_distribution<int> side_dist(0, 1); // BUY/SELL
    
    for (size_t i = 0; i < count; ++i) {
        uint64_t timestamp = static_cast<uint64_t>(now + (i * 100)); // 100ms intervals
        double price = price_dist(rng);
        float volume = volume_dist(rng);
        BTQuant::Data::TradeSide side = side_dist(rng) ? BTQuant::Data::TradeSide::BUY : BTQuant::Data::TradeSide::SELL;
        uint8_t exchange_id = exchange_dist(rng);
        
        // Randomly assign flags
        uint8_t flags = 0;
        if (std::uniform_int_distribution<int>(0, 1)(rng)) {
            BTQuant::Data::set_flag(flags, BTQuant::Data::TradeFlags::AGGRESSIVE_ORDER);
        }
        if (std::uniform_int_distribution<int>(0, 3)(rng) == 0) { // 25% chance
            BTQuant::Data::set_flag(flags, BTQuant::Data::TradeFlags::MARKET_ORDER);
        }
        
        trades.emplace_back(timestamp, price, volume, side, exchange_id, flags);
    }
    
    return trades;
}

// Generate mock PriceLevel data
std::vector<PriceLevel> generateMockPriceLevels(size_t count, double base_price = 150.0) {
    std::vector<PriceLevel> levels;
    levels.reserve(count);
    
    std::uniform_real_distribution<double> price_offset_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> size_dist(1.0, 100.0);
    
    for (size_t i = 0; i < count; ++i) {
        PriceLevel level;
        level.price = base_price + price_offset_dist(rng);
        level.size = size_dist(rng);
        levels.push_back(level);
    }
    
    return levels;
}

// Generate mock VolumeProfileLevel data
std::vector<VolumeProfileLevel> generateMockVolumeProfileLevels(size_t count, double base_price = 150.0) {
    std::vector<VolumeProfileLevel> levels;
    levels.reserve(count);
    
    std::uniform_real_distribution<double> price_offset_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> total_vol_dist(100.0, 1000.0);
    std::uniform_real_distribution<double> buy_sell_ratio(0.3, 0.7); // 30-70% split
    
    for (size_t i = 0; i < count; ++i) {
        VolumeProfileLevel level;
        level.price = base_price + price_offset_dist(rng);
        level.total_volume = total_vol_dist(rng);
        double ratio = buy_sell_ratio(rng);
        level.buy_volume = level.total_volume * ratio;
        level.sell_volume = level.total_volume * (1.0 - ratio);
        levels.push_back(level);
    }
    
    return levels;
}

// Generate mock GpuOrderBookLevel data
std::vector<GpuOrderBookLevel> generateMockGpuOrderBookLevels(size_t count, float base_price = 150.0f) {
    std::vector<GpuOrderBookLevel> levels;
    levels.reserve(count);
    
    std::uniform_real_distribution<float> price_offset_dist(-5.0f, 5.0f);
    std::uniform_int_distribution<uint32_t> quantity_dist(10, 1000);
    std::uniform_int_distribution<uint32_t> orders_dist(1, 20);
    
    for (size_t i = 0; i < count; ++i) {
        GpuOrderBookLevel level;
        level.price = base_price + price_offset_dist(rng);
        level.askQuantity = quantity_dist(rng);
        level.bidQuantity = quantity_dist(rng);
        level.numOrders = orders_dist(rng);
        levels.push_back(level);
    }
    
    return levels;
}

// Generate mock CandleCluster data
std::vector<CandleCluster> generateMockCandleClusters(size_t count) {
    std::vector<CandleCluster> clusters;
    clusters.reserve(count);
    
    std::uniform_real_distribution<float> coord_dist(0.0f, 1000.0f);
    std::uniform_real_distribution<float> dimension_dist(1.0f, 20.0f);
    std::uniform_int_distribution<uint32_t> volume_dist(100, 10000);
    std::uniform_int_distribution<uint32_t> count_dist(1, 100);
    std::uniform_real_distribution<float> vwap_dist(100.0f, 200.0f);
    
    for (size_t i = 0; i < count; ++i) {
        CandleCluster cluster(
            coord_dist(rng),           // centerX
            coord_dist(rng),           // centerY  
            dimension_dist(rng),       // width
            dimension_dist(rng),       // height
            volume_dist(rng),          // bidVolume
            volume_dist(rng),          // askVolume
            count_dist(rng),           // tradeCount
            vwap_dist(rng),            // vwap
            true                       // hasTrades
        );
        
        cluster.buyTradeCount = count_dist(rng);
        cluster.sellTradeCount = count_dist(rng);
        cluster.maxSingleTradeVolume = std::uniform_real_distribution<float>(0.1f, 10.0f)(rng);
        cluster.startTimeNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) + (i * 1000000000ULL); // 1 sec intervals
        cluster.endTimeNs = cluster.startTimeNs + 500000000ULL; // 0.5 sec duration
        
        clusters.push_back(cluster);
    }
    
    return clusters;
}


// Generate mock RendererStats
RendererStats generateMockRendererStats() {
    RendererStats stats;
    stats.framesRendered = std::uniform_int_distribution<uint32_t>(1000, 10000)(rng);
    stats.lobUpdates = std::uniform_int_distribution<uint32_t>(100, 1000)(rng);
    stats.tradeUpdates = std::uniform_int_distribution<uint32_t>(500, 5000)(rng);
    stats.footprintCellsRendered = std::uniform_int_distribution<uint32_t>(100, 2000)(rng);
    stats.averageFrameTimeMs = std::uniform_real_distribution<double>(1.0, 16.0)(rng); // 1-16ms avg frame time
    stats.lastUpdateTimeNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    
    return stats;
}

// Generate mock RendererConfig
RendererConfig generateMockRendererConfig() {
    RendererConfig config;
    
    // LOB Heatmap config
    config.lobHeatmap.width = std::uniform_int_distribution<uint32_t>(512, 2048)(rng);
    config.lobHeatmap.height = std::uniform_int_distribution<uint32_t>(256, 1024)(rng);
    config.lobHeatmap.maxLiquidity = std::uniform_real_distribution<float>(50000.0f, 500000.0f)(rng);
    config.lobHeatmap.invertYAxis = std::uniform_int_distribution<int>(0, 1)(rng) == 1;
    
    // Footprint Chart config
    config.footprintChart.maxClusters = std::uniform_int_distribution<uint32_t>(1024, 8192)(rng);
    config.footprintChart.cellMinSize = std::uniform_real_distribution<float>(1.0f, 5.0f)(rng);
    config.footprintChart.cellMaxSize = std::uniform_real_distribution<float>(10.0f, 30.0f)(rng);
    config.footprintChart.showLabels = std::uniform_int_distribution<int>(0, 1)(rng) == 1;
    
    // TPO Profile config
    config.tpoProfile.bucketCount = std::uniform_int_distribution<uint32_t>(128, 512)(rng);
    config.tpoProfile.priceResolution = std::uniform_real_distribution<float>(0.01f, 1.0f)(rng);
    config.tpoProfile.timeWindowMs = std::uniform_int_distribution<uint32_t>(10000, 60000)(rng); // 10-60 sec
    config.tpoProfile.resetOnUpdate = std::uniform_int_distribution<int>(0, 1)(rng) == 1;
    
    return config;
}

// Generate comprehensive mock dataset containing all data types
struct ComprehensiveMockDataset {
    std::vector<OHLCVCandle> candles;
    std::vector<BTQuant::Data::TradeData> trades;
    std::vector<PriceLevel> price_levels;
    std::vector<VolumeProfileLevel> volume_profile_levels;
    std::vector<GpuOrderBookLevel> gpu_order_book_levels;
    std::vector<CandleCluster> candle_clusters;
    std::vector<HotspineTradeTick> hotspine_trade_ticks;
    RendererStats renderer_stats;
    RendererConfig renderer_config;
};

ComprehensiveMockDataset generateComprehensiveMockDataset() {
    ComprehensiveMockDataset dataset;
    
    dataset.candles = generateMockCandles(100);
    dataset.trades = generateMockTrades(500);
    dataset.price_levels = generateMockPriceLevels(20);
    dataset.volume_profile_levels = generateMockVolumeProfileLevels(50);
    dataset.gpu_order_book_levels = generateMockGpuOrderBookLevels(30);
    dataset.candle_clusters = generateMockCandleClusters(25);
    dataset.hotspine_trade_ticks = generateMockHotspineTradeTicks(1000);
    dataset.renderer_stats = generateMockRendererStats();
    dataset.renderer_config = generateMockRendererConfig();
    
    return dataset;
}

} // namespace TestData
} // namespace RenderEngine
} // namespace BTQuant