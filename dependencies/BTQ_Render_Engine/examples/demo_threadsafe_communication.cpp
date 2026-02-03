#include "../include/threading/lockfree_queue.hpp"
#include "../include/task_scheduler.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

using namespace btq::threading;

// Example demonstrating usage of lock-free data structures for calculation and UI thread communication
int main() {
    std::cout << "Demonstrating thread-safe data structures for calculation and UI thread communication...\n" << std::endl;

    // Example 1: Using LockFreeQueue for passing trade data from calculation to UI thread
    LockFreeQueue<btq::Trade> trade_queue;
    std::atomic<bool> should_stop{false};

    // Calculation thread (simulating trade data generation)
    auto calc_thread = std::thread([&trade_queue, &should_stop]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(95.0, 105.0);
        std::uniform_real_distribution<> volume_dist(10.0, 100.0);

        int counter = 0;
        while (!should_stop.load() && counter < 100) {
            btq::Trade trade;
            trade.timestamp = std::chrono::system_clock::now();
            trade.price = price_dist(gen);
            trade.volume = volume_dist(gen);

            trade_queue.push(std::move(trade));
            ++counter;
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Simulate processing time
        }
        std::cout << "Calculation thread finished generating " << counter << " trades." << std::endl;
    });

    // UI thread (simulating data consumption)
    auto ui_thread = std::thread([&trade_queue, &should_stop]() {
        int processed = 0;
        while (!should_stop.load() || !trade_queue.empty()) {
            auto trade_opt = trade_queue.try_pop();
            if (trade_opt.has_value()) {
                // Process the trade data in the UI
                std::cout << "UI Thread: Processed trade - Price: " << trade_opt->price 
                          << ", Volume: " << trade_opt->volume << std::endl;
                ++processed;
            } else {
                // No data available, yield to other threads
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        std::cout << "UI Thread finished processing " << processed << " trades." << std::endl;
    });

    // Let the threads run for a bit
    std::this_thread::sleep_for(std::chrono::seconds(2));
    should_stop.store(true);

    calc_thread.join();
    ui_thread.join();

    std::cout << "\nExample 2: Using SPSCRingBuffer for high-frequency data transfer..." << std::endl;

    // Example 2: Using SPSCRingBuffer for high-frequency price updates
    SPSCRingBuffer<double> price_buffer(100); // Ring buffer with capacity of 100
    
    // Producer thread (calculation)
    auto producer = std::thread([&price_buffer, &should_stop]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(90.0, 110.0);

        int counter = 0;
        while (!should_stop.load() && counter < 50) {
            double price = price_dist(gen);
            if (price_buffer.push(price)) {
                ++counter;
            } else {
                // Buffer full, wait a bit
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        std::cout << "Producer thread finished generating " << counter << " prices." << std::endl;
    });

    // Consumer thread (UI)
    auto consumer = std::thread([&price_buffer, &should_stop]() {
        int processed = 0;
        while (processed < 50) {
            auto price_opt = price_buffer.try_pop();
            if (price_opt.has_value()) {
                // Process the price in the UI
                std::cout << "Consumer: Received price " << *price_opt << std::endl;
                ++processed;
            } else {
                // Buffer empty, wait a bit
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        std::cout << "Consumer thread finished processing " << processed << " prices." << std::endl;
    });

    // Let run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    should_stop.store(true);

    producer.join();
    consumer.join();

    std::cout << "\nAll examples completed successfully!" << std::endl;
    return 0;
}