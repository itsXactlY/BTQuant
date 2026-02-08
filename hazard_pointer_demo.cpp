/**
 * @file hazard_pointer_demo.cpp
 * @brief Demonstration of the C++26 hazard pointer implementation
 */

#include "dependencies/BTQ_Render_Engine/include/threading/hazard_pointer_cxx26.hpp"
#include <iostream>
#include <thread>
#include <atomic>

using namespace btq;

// Example class that inherits from hazard_pointer_obj_base
struct DataNode : hazard_pointer_obj_base<DataNode> {
    int value;
    DataNode(int v) : value(v) {}
    
    ~DataNode() {
        std::cout << "DataNode with value " << value << " destroyed" << std::endl;
    }
};

// Shared atomic pointer
std::atomic<DataNode*> shared_data{nullptr};

void reader(int id) {
    // Create a hazard pointer for this thread
    hazard_pointer hp = make_hazard_pointer();
    
    for (int i = 0; i < 5; ++i) {
        // Protect the shared data pointer
        DataNode* current = hp.protect(shared_data);
        
        if (current) {
            std::cout << "Reader " << id << " accessing value: " << current->value << std::endl;
        } else {
            std::cout << "Reader " << id << " sees null" << std::endl;
        }
        
        // Reset protection before next access
        hp.reset_protection();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void writer() {
    for (int i = 0; i < 5; ++i) {
        DataNode* old_data = shared_data.load();
        DataNode* new_data = new DataNode(i * 10);
        
        // Atomically replace the pointer
        DataNode* expected = old_data;
        while (!shared_data.compare_exchange_weak(expected, new_data)) {
            expected = old_data;
        }
        
        // Retire the old data if it exists
        if (old_data) {
            old_data->retire();  // Safely defer deletion
        }
        
        std::cout << "Writer created new node with value: " << new_data->value << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

int main() {
    std::cout << "=== C++26 Hazard Pointer Demo ===" << std::endl;
    
    // Start reader threads
    std::thread r1(reader, 1);
    std::thread r2(reader, 2);
    
    // Start writer thread
    std::thread w(writer);
    
    // Wait for all threads to complete
    r1.join();
    r2.join();
    w.join();
    
    // Force cleanup of any remaining retired objects
    detail::hazard_pointer_manager::instance().cleanup_retired();
    
    std::cout << "Demo completed successfully!" << std::endl;
    
    return 0;
}