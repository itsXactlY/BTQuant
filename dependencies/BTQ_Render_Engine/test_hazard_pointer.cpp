#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <cassert>

#include "include/threading/hazard_pointer.hpp"

using namespace btq::threading;

// Test structure to be managed by hazard pointers
struct TestNode {
    int value;
    std::atomic<int> ref_count{1};
    
    TestNode(int v) : value(v) {}
    
    void inc_ref() { ref_count.fetch_add(1, std::memory_order_relaxed); }
    void dec_ref() { 
        if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }
    
    ~TestNode() = default;
};

void test_basic_hazard_pointer() {
    std::cout << "Testing basic hazard pointer functionality...\n";
    
    HazardPointerManager& manager = HazardPointerManager::instance();
    
    // Create a test node
    TestNode* node = new TestNode(42);
    
    // Acquire a hazard pointer to protect the node
    {
        auto guard = manager.acquire_hazard_pointer(node);
        
        // Access the node through the guard
        assert(guard.get()->value == 42);
        assert(guard->value == 42);
        assert((*guard).value == 42);
        
        // Retire the node (should not be deleted immediately due to hazard pointer)
        manager.retire(node, [node]() { node->dec_ref(); });
    }
    
    // After the guard goes out of scope, the node should be safe to delete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    manager.force_cleanup();
    
    std::cout << "Basic hazard pointer test passed.\n";
}

void test_concurrent_access() {
    std::cout << "Testing concurrent hazard pointer access...\n";
    
    HazardPointerManager& manager = HazardPointerManager::instance();
    
    // Shared pointer to be accessed by multiple threads
    TestNode* shared_node = new TestNode(100);
    
    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;
    
    // Reader threads that will protect the node with hazard pointers
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            
            // Protect the node with a hazard pointer
            auto guard = manager.acquire_hazard_pointer(shared_node);
            
            // Do some work with the protected node
            int val = guard->value;
            (void)val; // Suppress unused variable warning
            
            // Sleep a bit to allow other threads to run
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
    
    // Writer thread that tries to retire the node
    std::thread writer([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        // Wait a bit to ensure readers have acquired hazard pointers
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        // Try to retire the node (should not be deleted while readers hold hazard pointers)
        manager.retire(shared_node, [shared_node]() { 
            std::cout << "Node with value " << shared_node->value << " deleted safely\n"; 
            delete shared_node; 
        });
        
        // Force cleanup - should not delete the node yet because of active hazard pointers
        manager.force_cleanup();
    });
    
    // Start all threads
    start_flag.store(true, std::memory_order_release);
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    writer.join();
    
    // Now that all readers are done, force cleanup again to delete the node
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.force_cleanup();
    
    std::cout << "Concurrent access test passed.\n";
}

void test_multiple_nodes() {
    std::cout << "Testing multiple nodes with hazard pointers...\n";
    
    HazardPointerManager& manager = HazardPointerManager::instance();
    
    // Create multiple nodes
    std::vector<TestNode*> nodes;
    for (int i = 0; i < 10; ++i) {
        nodes.push_back(new TestNode(i * 10));
    }
    
    // Protect some nodes with hazard pointers
    std::vector<decltype(manager.acquire_hazard_pointer(nodes[0]))> guards;
    for (size_t i = 0; i < nodes.size(); i += 2) {  // Protect every other node
        guards.push_back(manager.acquire_hazard_pointer(nodes[i]));
    }
    
    // Retire all nodes (some should be protected, others not)
    for (auto* node : nodes) {
        manager.retire(node, [node]() { 
            std::cout << "Retired node with value: " << node->value << std::endl;
            delete node; 
        });
    }
    
    // Force cleanup - should only delete unprotected nodes
    manager.force_cleanup();
    
    // Release even-indexed guards (the protected nodes)
    guards.clear();
    
    // Force cleanup again - now should delete previously protected nodes
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.force_cleanup();
    
    std::cout << "Multiple nodes test passed.\n";
}

void test_move_semantics() {
    std::cout << "Testing move semantics for hazard pointer guards...\n";
    
    HazardPointerManager& manager = HazardPointerManager::instance();
    
    TestNode* node1 = new TestNode(1);
    TestNode* node2 = new TestNode(2);
    
    // Create a guard and move it
    auto guard1 = manager.acquire_hazard_pointer(node1);
    assert(guard1.get() == node1);
    
    // Move the guard to another variable
    auto guard2 = std::move(guard1);
    assert(guard2.get() == node1);
    // guard1 should be in a valid but unspecified state after move
    
    // Move assign to an existing guard
    auto guard3 = manager.acquire_hazard_pointer(node2);
    guard3 = std::move(guard2);
    assert(guard3.get() == node1);
    
    // Clean up
    manager.retire(node1, [node1]() { delete node1; });
    manager.retire(node2, [node2]() { delete node2; });
    manager.force_cleanup();
    
    std::cout << "Move semantics test passed.\n";
}

int main() {
    std::cout << "Starting hazard pointer tests...\n";
    
    test_basic_hazard_pointer();
    test_concurrent_access();
    test_multiple_nodes();
    test_move_semantics();
    
    std::cout << "All hazard pointer tests passed!\n";
    
    return 0;
}