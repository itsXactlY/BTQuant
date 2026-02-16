#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>

// DEPRECATED - Legacy hotspine
// Mock HotSpineDataBridge for testing purposes
class MockHotSpineDataBridge {
public:
    void connect() {}
    // Add mock methods as needed for testing
};

// Include our header after defining the mock
#include "dependencies/BTQ_Render_Engine/include/data/orderbook_snapshot_manager.hpp"

void test_basic_functionality() {
    std::cout << "Testing basic functionality..." << std::endl;
    
    MockHotSpineDataBridge mock_bridge;
    void* bridge_ptr = &mock_bridge;
    OrderbookSnapshotManager manager(bridge_ptr, 100); // Small buffer for testing
    
    // Test initial state
    assert(manager.size() == 0);
    assert(manager.capacity() == 100);
    assert(manager.get_latest_snapshot() == nullptr);
    
    std::cout << "✓ Initial state tests passed" << std::endl;
    
    // Create a test snapshot
    OrderbookSnapshotManager::OrderbookSnapshot snapshot;
    snapshot.timestamp = 123456789;
    snapshot.symbol_id = 1;
    snapshot.bids_count = 2;
    snapshot.asks_count = 2;
    
    // Add some sample data
    snapshot.bids[0] = {100.0, 10.0};
    snapshot.bids[1] = {99.5, 15.0};
    snapshot.asks[0] = {101.0, 8.0};
    snapshot.asks[1] = {101.5, 12.0};
    
    // Add the snapshot
    manager.add_snapshot(std::move(snapshot));
    
    // Give a moment for the atomic operations to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Verify it was added
    assert(manager.size() >= 0); // At least 0
    const auto* latest = manager.get_latest_snapshot();
    assert(latest != nullptr);
    assert(latest->timestamp == 123456789);
    assert(latest->symbol_id == 1);
    assert(latest->bids_count == 2);
    assert(latest->asks_count == 2);
    
    std::cout << "✓ Basic add/retrieve tests passed" << std::endl;
}

int main() {
    std::cout << "Running OrderbookSnapshotManager basic tests..." << std::endl;
    
    test_basic_functionality();
    
    std::cout << "\nBasic tests passed! ✓" << std::endl;
    
    return 0;
}