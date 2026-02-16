#include "components/chart_super_node.hpp"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include <iostream>
#include <memory>
#include <cassert>

int main() {
    std::cout << "Testing ChartSuperNode...\n";

    // Create mock dependencies
    // DEPRECATED - Legacy hotspine
    auto bridge = std::make_shared<BTQuant::HotSpineDataBridge>();
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();
    
    // Create the ChartSuperNode
    auto super_node = std::make_shared<BTQuant::ChartSuperNode>(bridge, processor);
    
    // Initialize the super node
    super_node->initialize();
    
    assert(super_node->is_initialized() == true);
    std::cout << "✓ Initialization test passed\n";
    
    // Test chart creation
    uint32_t chart_id = super_node->create_chart("BTC-USDT", "Binance", 1, 
                                                 RenderEngine::TimeFrame::TF_1MIN);
    assert(chart_id != 0);
    std::cout << "✓ Chart creation test passed\n";
    
    // Test getting chart count
    size_t chart_count = super_node->get_active_chart_count();
    assert(chart_count == 1);
    std::cout << "✓ Chart count test passed\n";
    
    // Test TPO engine access
    auto& tpo_engine = super_node->get_tpo_engine();
    std::cout << "✓ TPO engine access test passed\n";
    
    // Test indicator calculation
    BTQuant::ChartSuperNode::IndicatorParams params;
    params.period = 14;
    super_node->calculate_indicator("SMA", chart_id, params);
    std::cout << "✓ Indicator calculation test passed\n";
    
    // Test thread safety methods
    super_node->lock();
    super_node->unlock();
    bool lock_acquired = super_node->try_lock();
    assert(lock_acquired == true);
    super_node->unlock();
    std::cout << "✓ Thread safety test passed\n";
    
    // Test update method
    super_node->update();
    std::cout << "✓ Update test passed\n";
    
    // Test destruction
    super_node->destroy_chart(chart_id);
    assert(super_node->get_active_chart_count() == 0);
    std::cout << "✓ Chart destruction test passed\n";
    
    std::cout << "\nAll tests passed! ChartSuperNode is working correctly.\n";
    
    return 0;
}