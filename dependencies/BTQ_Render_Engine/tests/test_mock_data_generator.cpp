#include "mock_data_generator.cpp"
#include <iostream>
#include <cassert>

int main() {
    // Test OHLCVCandle generation
    auto candles = BTQuant::RenderEngine::TestData::generateMockCandles(10);
    assert(candles.size() == 10);
    std::cout << "Generated " << candles.size() << " OHLCV candles\n";
    
    // Test TradeData generation
    auto trades = BTQuant::RenderEngine::TestData::generateMockTrades(20);
    assert(trades.size() == 20);
    std::cout << "Generated " << trades.size() << " trade records\n";
    
    // Test PriceLevel generation
    auto price_levels = BTQuant::RenderEngine::TestData::generateMockPriceLevels(15);
    assert(price_levels.size() == 15);
    std::cout << "Generated " << price_levels.size() << " price levels\n";
    
    // Test VolumeProfileLevel generation
    auto volume_profiles = BTQuant::RenderEngine::TestData::generateMockVolumeProfileLevels(12);
    assert(volume_profiles.size() == 12);
    std::cout << "Generated " << volume_profiles.size() << " volume profile levels\n";
    
    // Test GpuOrderBookLevel generation
    auto gpu_levels = BTQuant::RenderEngine::TestData::generateMockGpuOrderBookLevels(8);
    assert(gpu_levels.size() == 8);
    std::cout << "Generated " << gpu_levels.size() << " GPU order book levels\n";
    
    // Test CandleCluster generation
    auto clusters = BTQuant::RenderEngine::TestData::generateMockCandleClusters(5);
    assert(clusters.size() == 5);
    std::cout << "Generated " << clusters.size() << " candle clusters\n";
    
    // Test HotspineTradeTick generation
    auto ticks = BTQuant::RenderEngine::TestData::generateMockHotspineTradeTicks(30);
    assert(ticks.size() == 30);
    std::cout << "Generated " << ticks.size() << " hotspine trade ticks\n";
    
    // Test RendererStats generation
    auto stats = BTQuant::RenderEngine::TestData::generateMockRendererStats();
    std::cout << "Generated renderer stats with " << stats.framesRendered << " frames rendered\n";
    
    // Test RendererConfig generation
    auto config = BTQuant::RenderEngine::TestData::generateMockRendererConfig();
    std::cout << "Generated renderer config with heatmap size " 
              << config.lobHeatmap.width << "x" << config.lobHeatmap.height << "\n";
    
    // Test comprehensive dataset generation
    auto dataset = BTQuant::RenderEngine::TestData::generateComprehensiveMockDataset();
    std::cout << "Generated comprehensive dataset with " 
              << dataset.candles.size() << " candles, "
              << dataset.trades.size() << " trades, "
              << dataset.price_levels.size() << " price levels\n";
    
    std::cout << "All tests passed!\n";
    return 0;
}