#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cassert>
#include <string>

#include "rcu.hpp"

void test_rcu_basic_functionality() {
    std::cout << "Testing basic C++26 RCU functionality...\n";

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

    std::cout << "Basic C++26 RCU functionality test passed!\n";
}

void test_rcu_concurrent_access() {
    std::cout << "Testing concurrent C++26 RCU access...\n";

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
                // Perform read operations using the new interface
                auto guard = rcu_set.rcu_read_lock();
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

    std::cout << "Concurrent C++26 RCU access test passed! Performed "
              << read_count.load() << " read operations.\n";
}

void test_rcu_config_map() {
    std::cout << "Testing C++26 RCU with configuration maps...\n";

    BTQ::rcu_unordered_map<std::string, std::string> rcu_config;

    // Insert some configuration values
    rcu_config.insert_or_assign("database.host", "localhost");
    rcu_config.insert_or_assign("database.port", "5432");
    rcu_config.insert_or_assign("cache.enabled", "true");

    // Read values back using the new interface
    auto guard = rcu_config.rcu_read_lock();
    auto host_it = rcu_config.find("database.host");
    auto port_it = rcu_config.find("database.port");
    auto cache_it = rcu_config.find("cache.enabled");

    assert(host_it != guard->end() && host_it->second == "localhost");
    assert(port_it != guard->end() && port_it->second == "5432");
    assert(cache_it != guard->end() && cache_it->second == "true");

    // Also test the get method
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

    std::cout << "C++26 RCU configuration map test passed!\n";
}

void test_rcu_complex_update() {
    std::cout << "Testing C++26 RCU complex update operations...\n";

    BTQ::rcu_unordered_map<std::string, int> rcu_config;
    
    // Insert initial values
    rcu_config.insert_or_assign("counter1", 10);
    rcu_config.insert_or_assign("counter2", 20);
    
    // Perform a complex update operation
    rcu_config.update([](auto& map) {
        map["counter1"] += 5;
        map["counter2"] *= 2;
        map["counter3"] = 100;
        map.erase("nonexistent"); // Safe to erase non-existent keys
    });
    
    // Verify the updates
    auto counter1 = rcu_config.get("counter1");
    auto counter2 = rcu_config.get("counter2");
    auto counter3 = rcu_config.get("counter3");
    
    assert(counter1.has_value() && counter1.value() == 15);
    assert(counter2.has_value() && counter2.value() == 40);
    assert(counter3.has_value() && counter3.value() == 100);
    
    std::cout << "C++26 RCU complex update test passed!\n";
}

void test_rcu_active_pairs_simulation() {
    std::cout << "Testing C++26 RCU with active pairs simulation...\n";

    // Simulate the active_pairs_ functionality from MarketDataProcessor
    BTQ::rcu_unordered_set<std::string> active_pairs;

    // Add some trading pairs
    active_pairs.insert("BINANCE:BTC:USD:spot");
    active_pairs.insert("COINBASE:ETH:USD:spot");
    active_pairs.insert("FTX:SOL:USD:spot");

    // Verify pairs exist
    assert(active_pairs.contains("BINANCE:BTC:USD:spot"));
    assert(active_pairs.contains("COINBASE:ETH:USD:spot"));
    assert(active_pairs.contains("FTX:SOL:USD:spot"));
    assert(!active_pairs.contains("NONEXISTENT:PAIR"));

    // Remove a pair
    active_pairs.erase("FTX:SOL:USD:spot");
    assert(!active_pairs.contains("FTX:SOL:USD:spot"));

    // Test concurrent access pattern similar to MarketDataProcessor
    std::atomic<bool> stop{false};
    std::atomic<int> checks{0};

    std::thread checker([&]() {
        while (!stop.load()) {
            // This simulates the reader thread checking for active pairs
            auto guard = active_pairs.rcu_read_lock();
            if (guard->size() > 0) {
                checks.fetch_add(1);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    // Simulate adding new pairs (like in handleTradeMessage)
    for (int i = 0; i < 10; ++i) {
        std::string pair = "EXCHANGE:SYM" + std::to_string(i) + ":USD:spot";
        active_pairs.insert(pair);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true);
    checker.join();

    std::cout << "C++26 RCU active pairs simulation test passed! Performed "
              << checks.load() << " checks.\n";
}

int main() {
    std::cout << "Starting C++26 RCU tests...\n";

    test_rcu_basic_functionality();
    test_rcu_concurrent_access();
    test_rcu_config_map();
    test_rcu_complex_update();
    test_rcu_active_pairs_simulation();

    std::cout << "All C++26 RCU tests passed!\n";
    return 0;
}