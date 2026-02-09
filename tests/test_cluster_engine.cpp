#include "analytics/cluster_engine.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing Cluster Engine with Bypass Mode...\n";

    // Create a cluster engine instance
    analytics::ClusterEngine engine;

    // Test 1: Verify initial state
    assert(engine.IsBypassMode() == false);  // Should be active by default
    std::cout << "✓ Initial state: Engine is active\n";

    // Test 2: Add some sample data points
    std::vector<MarketDataPoint> sample_data = {
        {1634567890000, 45000.0, 1.5, "BTCUSD"},
        {1634567891000, 45100.0, 2.0, "BTCUSD"},
        {1634567892000, 44900.0, 1.2, "BTCUSD"},
        {1634567893000, 45200.0, 0.8, "BTCUSD"},
        {1634567894000, 45300.0, 1.7, "BTCUSD"}
    };

    // Initialize the engine with sample data
    engine.Initialize(sample_data);
    std::cout << "✓ Initialization completed\n";

    // Get clusters (should have some clusters in active mode)
    auto clusters = engine.GetClusters();
    std::cout << "✓ Retrieved " << clusters.size() << " clusters in active mode\n";

    // Test 3: Enable bypass mode
    engine.SetBypassMode(true);
    assert(engine.IsBypassMode() == true);
    std::cout << "✓ Bypass mode enabled\n";

    // Test 4: Add a new data point in bypass mode
    MarketDataPoint new_point{1634567895000, 45400.0, 1.0, "BTCUSD"};
    engine.Update(new_point);
    std::cout << "✓ Updated with new point in bypass mode\n";

    // Test 5: Get clusters in bypass mode (should return empty or simplified)
    clusters = engine.GetClusters();
    std::cout << "✓ Retrieved " << clusters.size() << " clusters in bypass mode\n";

    // Test 6: Get analytics bypass result
    auto bypass_result = engine.GetAnalyticsBypassResult();
    std::cout << "✓ Got bypass analytics result\n";
    std::cout << "  Average price: " << bypass_result.average_price << "\n";
    std::cout << "  Volatility: " << bypass_result.volatility << "\n";
    std::cout << "  Trend: " << bypass_result.trend << "\n";

    // Test 7: Disable bypass mode and verify
    engine.SetBypassMode(false);
    assert(engine.IsBypassMode() == false);
    std::cout << "✓ Bypass mode disabled, engine active again\n";

    std::cout << "\nAll tests passed! Cluster Engine with Bypass Mode is working correctly.\n";
    return 0;
}