#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cassert>

#include "../dependencies/BTQ_Render_Engine/include/rcu/rcu_config_wrapper.hpp"

void test_rcu_basic_functionality() {
    std::cout << "Testing basic RCU functionality...\n";
    
    // Test RCU with unordered_set
    BTQ::rcu_unordered_set<int> rcu_set;
    
    // Insert some values
    rcu_set.insert(1);
    rcu_set.insert(2);
    rcu_set.insert(3);
    
    // Verify values can be read
    assert(rcu_set.contains(1));
    assert(rcu_set.contains(2));
    assert(rcu_set.contains(3));
    assert(!rcu_set.contains(4)); // Should not exist
    
    // Erase a value
    rcu_set.erase(2);
    assert(!rcu_set.contains(2)); // Should no longer exist
    
    std::cout << "Basic RCU functionality test passed!\n";
}

void test_rcu_concurrent_access() {
    std::cout << "Testing concurrent RCU access...\n";
    
    BTQ::rcu_unordered_set<int> rcu_set;
    
    // Writer thread - continuously updates the set
    std::atomic<bool> stop_writing{false};
    std::thread writer([&rcu_set, &stop_writing]() {
        int counter = 10;
        while (!stop_writing.load()) {
            rcu_set.insert(counter++);
            if (counter % 10 == 0) {
                rcu_set.erase(counter - 5); // Erase some old values
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    
    // Reader threads - continuously read from the set
    std::atomic<int> read_count{0};
    std::vector<std::thread> readers;
    
    for (int i = 0; i < 3; ++i) {
        readers.emplace_back([&rcu_set, &read_count, &stop_writing]() {
            while (!stop_writing.load()) {
                // Perform read operations
                auto guard = rcu_set.read_lock();
                size_t size = guard->size();
                (void)size; // Suppress unused variable warning
                read_count.fetch_add(1);
                
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }
    
    // Let them run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop_writing.store(true);
    
    writer.join();
    for (auto& r : readers) {
        r.join();
    }
    
    std::cout << "Concurrent RCU access test passed! Performed " 
              << read_count.load() << " read operations.\n";
}

void test_rcu_config_map() {
    std::cout << "Testing RCU with configuration maps...\n";
    
    BTQ::rcu_unordered_map<std::string, std::string> rcu_config;
    
    // Insert some configuration values
    rcu_config.insert_or_assign("database.host", "localhost");
    rcu_config.insert_or_assign("database.port", "5432");
    rcu_config.insert_or_assign("cache.enabled", "true");
    
    // Read values back
    auto host_opt = rcu_config.get("database.host");
    auto port_opt = rcu_config.get("database.port");
    auto cache_opt = rcu_config.get("cache.enabled");
    
    assert(host_opt.has_value() && host_opt.value() == "localhost");
    assert(port_opt.has_value() && port_opt.value() == "5432");
    assert(cache_opt.has_value() && cache_opt.value() == "true");
    
    // Update a value
    rcu_config.insert_or_assign("database.port", "3306");
    auto new_port = rcu_config.get("database.port");
    assert(new_port.has_value() && new_port.value() == "3306");
    
    // Erase a value
    rcu_config.erase("cache.enabled");
    auto missing = rcu_config.get("cache.enabled");
    assert(!missing.has_value());
    
    std::cout << "RCU configuration map test passed!\n";
}

int main() {
    std::cout << "Starting RCU tests...\n";
    
    test_rcu_basic_functionality();
    test_rcu_concurrent_access();
    test_rcu_config_map();
    
    std::cout << "All RCU tests passed!\n";
    return 0;
}