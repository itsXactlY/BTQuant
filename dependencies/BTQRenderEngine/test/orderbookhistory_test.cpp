#include "../src/data/orderbookhistory.h"
#include <iostream>
#include <cassert>
#include <vector>

void testBasicFunctionality() {
    std::cout << "Testing basic OrderBookHistory functionality..." << std::endl;
    
    OrderBookHistory history(5); // Small buffer for testing
    
    // Create and add some test snapshots
    auto snapshot1 = std::make_unique<L2Snapshot>();
    snapshot1->timestamp = 1000;
    snapshot1->bids.push_back(OrderLevel(99.5, 100));
    snapshot1->asks.push_back(OrderLevel(100.5, 100));
    
    auto snapshot2 = std::make_unique<L2Snapshot>();
    snapshot2->timestamp = 2000;
    snapshot2->bids.push_back(OrderLevel(99.0, 150));
    snapshot2->asks.push_back(OrderLevel(101.0, 150));
    
    history.addSnapshot(std::move(snapshot1));
    history.addSnapshot(std::move(snapshot2));
    
    assert(history.size() == 2);
    assert(!history.empty());
    
    auto retrieved = history.getSnapshotByIndex(0);
    assert(retrieved != nullptr);
    assert(retrieved->timestamp == 1000);
    assert(retrieved->bids.size() == 1);
    assert(retrieved->asks.size() == 1);
    assert(retrieved->bids[0].price == 99.5);
    assert(retrieved->asks[0].price == 100.5);
    
    retrieved = history.getSnapshotByIndex(1);
    assert(retrieved != nullptr);
    assert(retrieved->timestamp == 2000);
    
    std::cout << "Basic functionality test passed!" << std::endl;
}

void testCircularBuffer() {
    std::cout << "Testing circular buffer behavior..." << std::endl;
    
    OrderBookHistory history(3); // Small buffer to test wrapping
    
    // Fill the buffer
    for (int i = 0; i < 5; ++i) {
        auto snapshot = std::make_unique<L2Snapshot>();
        snapshot->timestamp = 1000 + i * 100;
        snapshot->bids.push_back(OrderLevel(99.0 + i, 100 + i));
        snapshot->asks.push_back(OrderLevel(100.0 + i, 100 + i));
        
        history.addSnapshot(std::move(snapshot));
    }
    
    // Buffer capacity is 3, so we should have 3 snapshots
    assert(history.size() == 3);
    
    // After adding 5 snapshots to a size-3 buffer:
    // Internal buffer: [1300, 1400, 1200], head_=2
    // Logical order (oldest to newest): [1200, 1300, 1400]
    // So getSnapshotByIndex(0) should return 1200 (oldest)
    auto oldest = history.getSnapshotByIndex(0);
    assert(oldest != nullptr);
    assert(oldest->timestamp == 1200);
    
    // The middle should be timestamp 1300
    auto middle = history.getSnapshotByIndex(1);
    assert(middle != nullptr);
    assert(middle->timestamp == 1300);
    
    // The newest should be timestamp 1400
    auto newest = history.getSnapshotByIndex(2);
    assert(newest != nullptr);
    assert(newest->timestamp == 1400);
    
    std::cout << "Circular buffer test passed!" << std::endl;
}

void testGetByTime() {
    std::cout << "Testing getSnapshotByTime functionality..." << std::endl;
    
    OrderBookHistory history(10);
    
    // Add snapshots at different times
    for (int i = 0; i < 5; ++i) {
        auto snapshot = std::make_unique<L2Snapshot>();
        snapshot->timestamp = 1000 + i * 100;
        snapshot->bids.push_back(OrderLevel(99.0 + i, 100 + i));
        snapshot->asks.push_back(OrderLevel(100.0 + i, 100 + i));
        
        history.addSnapshot(std::move(snapshot));
    }
    
    // Test retrieving by exact timestamp
    auto snapshot = history.getSnapshotByTime(1200);
    assert(snapshot != nullptr);
    assert(snapshot->timestamp == 1200);
    
    // Test retrieving by close timestamp (should get closest)
    snapshot = history.getSnapshotByTime(1250); // Closest to 1200 or 1300? Should be 1200 if equal diff
    assert(snapshot != nullptr);
    // Since 1250 is equidistant from 1200 and 1300, it could be either, so we just check it's one of them
    assert(snapshot->timestamp == 1200 || snapshot->timestamp == 1300);
    
    std::cout << "Get by time test passed!" << std::endl;
}

void testGetRange() {
    std::cout << "Testing getSnapshotsInRange functionality..." << std::endl;
    
    OrderBookHistory history(10);
    
    // Add snapshots at different times
    for (int i = 0; i < 5; ++i) {
        auto snapshot = std::make_unique<L2Snapshot>();
        snapshot->timestamp = 1000 + i * 100;
        snapshot->bids.push_back(OrderLevel(99.0 + i, 100 + i));
        snapshot->asks.push_back(OrderLevel(100.0 + i, 100 + i));
        
        history.addSnapshot(std::move(snapshot));
    }
    
    // Get snapshots in range [1100, 1300]
    auto range = history.getSnapshotsInRange(1100, 1300);
    assert(range.size() == 3); // Should get timestamps 1100, 1200, 1300
    
    // Verify timestamps are in range
    for (const auto& snap : range) {
        assert(snap->timestamp >= 1100 && snap->timestamp <= 1300);
    }
    
    std::cout << "Get range test passed!" << std::endl;
}

void testClear() {
    std::cout << "Testing clear functionality..." << std::endl;
    
    OrderBookHistory history(10);
    
    // Add some snapshots
    auto snapshot = std::make_unique<L2Snapshot>();
    snapshot->timestamp = 1000;
    history.addSnapshot(std::move(snapshot));
    
    assert(history.size() == 1);
    
    history.clear();
    
    assert(history.size() == 0);
    assert(history.empty());
    
    std::cout << "Clear test passed!" << std::endl;
}

int main() {
    std::cout << "Running OrderBookHistory tests..." << std::endl;
    
    testBasicFunctionality();
    testCircularBuffer();
    testGetByTime();
    testGetRange();
    testClear();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}