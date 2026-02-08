#include "dependencies/BTQ_Render_Engine/include/threading/hazard_pointer_cxx26.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>

using namespace btq;

// Example class that can be protected by hazard pointers
struct ProtectedNode : hazard_pointer_obj_base<ProtectedNode> {
    int data;
    ProtectedNode(int val) : data(val) {}
    
    ~ProtectedNode() {
        std::cout << "Node with data " << data << " deleted" << std::endl;
    }
};

// Shared atomic pointer to a protected node
std::atomic<ProtectedNode*> shared_ptr{nullptr};

void reader_thread(int id) {
    hazard_pointer hp = make_hazard_pointer();
    
    for (int i = 0; i < 100; ++i) {
        ProtectedNode* current = hp.protect(shared_ptr);
        if (current) {
            // Simulate some work with the protected pointer
            volatile int temp = current->data;  // Prevent optimization
            std::cout << "Reader " << id << " sees data: " << current->data << std::endl;
        } else {
            std::cout << "Reader " << id << " sees null" << std::endl;
        }
        
        // Reset protection before next iteration
        hp.reset_protection();
        
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void writer_thread() {
    for (int i = 0; i < 50; ++i) {
        // Create a new node
        ProtectedNode* old_node = shared_ptr.load();
        ProtectedNode* new_node = new ProtectedNode(i * 10);
        
        // Atomically replace the pointer
        ProtectedNode* expected = old_node;
        while (!shared_ptr.compare_exchange_weak(expected, new_node)) {
            expected = old_node;
        }
        
        // Retire the old node if it existed
        if (old_node) {
            old_node->retire();  // This will defer deletion safely
        }
        
        std::cout << "Writer created node with data: " << new_node->data << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

int main() {
    std::cout << "Testing C++26-style hazard pointers..." << std::endl;
    
    // Start reader threads
    std::vector<std::thread> readers;
    for (int i = 0; i < 3; ++i) {
        readers.emplace_back(reader_thread, i);
    }
    
    // Start writer thread
    std::thread writer(writer_thread);
    
    // Wait for all threads to complete
    for (auto& r : readers) {
        r.join();
    }
    writer.join();
    
    // Force cleanup of any remaining retired objects
    detail::hazard_pointer_manager::instance().cleanup_retired();
    
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
}