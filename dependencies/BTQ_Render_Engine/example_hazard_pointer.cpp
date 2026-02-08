#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>

#include "include/threading/hazard_pointer.hpp"

using namespace btq::threading;

// Example structure that will be shared between threads
struct SharedNode {
    int data;
    std::atomic<int> ref_count{1};
    
    SharedNode(int val) : data(val) {}
    
    void inc_ref() { ref_count.fetch_add(1, std::memory_order_relaxed); }
    void dec_ref() { 
        if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::cout << "Deleting node with data: " << data << std::endl;
            delete this;
        }
    }
    
    ~SharedNode() = default;
};

int main() {
    std::cout << "Hazard Pointer Example - Safe Memory Reclamation\n";
    std::cout << "================================================\n\n";
    
    HazardPointerManager& hpm = HazardPointerManager::instance();
    
    // Create a shared node
    SharedNode* shared_node = new SharedNode(42);
    std::cout << "Created shared node with data: " << shared_node->data << std::endl;
    
    // Thread 1: Reads the node using hazard pointer
    std::thread reader([&hpm, shared_node]() {
        // Protect the node with a hazard pointer
        auto guard = hpm.acquire_hazard_pointer(shared_node);
        
        std::cout << "Reader thread: Accessing node data = " << guard->data << std::endl;
        
        // Simulate some work while holding the hazard pointer
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        std::cout << "Reader thread: Done accessing node\n";
        // Hazard pointer automatically released when guard goes out of scope
    });
    
    // Give the reader thread a chance to acquire the hazard pointer
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Main thread tries to retire the node (but it won't be deleted while reader holds hazard pointer)
    std::cout << "Main thread: Attempting to retire the node\n";
    hpm.retire(shared_node, [shared_node]() { 
        std::cout << "Deleter called for node with data: " << shared_node->data << std::endl;
        shared_node->dec_ref(); 
    });
    
    // Wait for reader to finish
    reader.join();
    
    // Now that the reader is done, force cleanup to delete the retired node
    std::cout << "Main thread: Forcing cleanup\n";
    hpm.force_cleanup();
    
    std::cout << "\nExample completed successfully!\n";
    
    return 0;
}