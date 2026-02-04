#include "../include/threading/lockfree_queue.hpp"
#include "../include/task_scheduler.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

using namespace btq::threading;

// Example demonstrating thread-safe communication between calculation threads and UI thread
int main() {
    std::cout << "Demonstrating thread-safe communication between calculation and UI threads..." << std::endl;

    // Using UIUpdateQueue which is optimized for calculation thread to UI thread communication
    UIUpdateQueue<btq::Trade> trade_queue(5000); // Max size of 5000 items to prevent memory buildup
    
    // Simulate calculation threads producing data
    std::vector<std::thread> calc_threads;
    
    // Start 3 calculation threads that generate trade data
    for (int i = 0; i < 3; ++i) {
        calc_threads.emplace_back([&, i]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> price_dist(95.0, 105.0);
            std::uniform_real_distribution<> vol_dist(1.0, 100.0);
            
            for (int j = 0; j < 1000; ++j) {  // Each thread produces 1000 trades
                btq::Trade trade;
                trade.price = price_dist(gen);
                trade.volume = vol_dist(gen);
                trade.timestamp = std::chrono::system_clock::now();
                
                // Push to queue - this is thread-safe
                bool success = trade_queue.push(trade);
                if (!success) {
                    std::cout << "Warning: Trade dropped due to queue overflow!" << std::endl;
                }
                
                // Small delay to simulate calculation work
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            std::cout << "Calculation thread " << i << " completed." << std::endl;
        });
    }
    
    // Simulate UI thread consuming data
    std::thread ui_thread([&trade_queue]() {
        int processed_count = 0;
        int batch_count = 0;
        
        auto start_time = std::chrono::steady_clock::now();
        auto timeout = start_time + std::chrono::seconds(10); // 10 second timeout
        
        while (processed_count < 3000 && std::chrono::steady_clock::now() < timeout) {
            // Process data in batches for efficiency - typical UI pattern
            auto batch = trade_queue.pop_batch(50); // Process up to 50 items at once
            
            if (!batch.empty()) {
                batch_count++;
                processed_count += batch.size();
                
                // Simulate UI processing (e.g., updating charts, order books, etc.)
                for (const auto& trade : batch) {
                    // In real UI code, this might update chart displays, order book, etc.
                    // For demo, just show first and last of each batch
                    if (&trade == &batch.front() || &trade == &batch.back()) {
                        std::cout << "UI processed trade: Price=" << trade.price 
                                  << ", Volume=" << trade.volume << std::endl;
                    }
                }
                
                std::cout << "UI processed batch " << batch_count << " with " 
                          << batch.size() << " items. Total: " << processed_count << std::endl;
            } else {
                // No data available, yield to allow other threads to run
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
            }
        }
        
        std::cout << "UI thread processed " << processed_count << " trades in " 
                  << batch_count << " batches." << std::endl;
        
        // Drain any remaining items
        auto remaining = trade_queue.drain_all();
        if (!remaining.empty()) {
            std::cout << "Drained " << remaining.size() << " remaining items." << std::endl;
        }
    });
    
    // Wait for all calculation threads to complete
    for (auto& t : calc_threads) {
        t.join();
    }
    
    // Wait for UI thread to complete
    ui_thread.join();
    
    // Print final statistics
    std::cout << "\nFinal queue statistics:" << std::endl;
    std::cout << "Total pushed: " << trade_queue.total_pushed() << std::endl;
    std::cout << "Total popped: " << trade_queue.total_popped() << std::endl;
    std::cout << "Dropped count: " << trade_queue.dropped_count() << std::endl;
    std::cout << "Current size: " << trade_queue.size_approx() << std::endl;
    
    std::cout << "\nThread-safe communication demo completed successfully!" << std::endl;
    
    return 0;
}