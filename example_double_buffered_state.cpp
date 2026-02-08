#include "threading/double_buffered_state.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>

using namespace btq::threading;

// Example: Trading statistics that can be safely updated and read concurrently
struct TradingStats {
    double total_volume = 0.0;
    double buy_volume = 0.0;
    double sell_volume = 0.0;
    int trade_count = 0;
    double last_price = 0.0;
    std::vector<double> recent_prices;
    
    // Update stats with a new trade
    void update(double price, double size, bool is_buy) {
        last_price = price;
        total_volume += size;
        if (is_buy) {
            buy_volume += size;
        } else {
            sell_volume += size;
        }
        trade_count++;
        
        // Keep only the last 100 prices
        recent_prices.push_back(price);
        if (recent_prices.size() > 100) {
            recent_prices.erase(recent_prices.begin());
        }
    }
    
    // Calculate some derived metrics
    double get_buy_ratio() const {
        return total_volume > 0 ? buy_volume / total_volume : 0.5;
    }
    
    double get_avg_price() const {
        if (recent_prices.empty()) return 0.0;
        double sum = 0.0;
        for (double p : recent_prices) sum += p;
        return sum / recent_prices.size();
    }
};

void demonstrate_usage() {
    std::cout << "=== Double Buffered State Usage Example ===\n\n";
    
    // Create a double buffered state for trading statistics
    DoubleBufferedState<TradingStats> stats_buffer;
    
    // Simulate a writer thread that updates trading stats
    std::thread writer([&stats_buffer]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(95.0, 105.0);
        std::uniform_real_distribution<> size_dist(1.0, 100.0);
        std::bernoulli_distribution buy_dist(0.5);
        
        for (int i = 0; i < 1000; ++i) {
            // Get the back buffer and update it
            auto& back_stats = stats_buffer.write();
            double price = price_dist(gen);
            double size = size_dist(gen);
            bool is_buy = buy_dist(gen);
            
            back_stats.update(price, size, is_buy);
            
            // Atomically swap the buffers so readers see the updated data
            stats_buffer.swap();
            
            std::this_thread::sleep_for(std::chrono::microseconds(100)); // Simulate time between trades
        }
    });
    
    // Simulate a reader thread that displays current stats
    std::thread reader([&stats_buffer]() {
        for (int i = 0; i < 100; ++i) {
            // Safely read the current stats without blocking the writer
            const auto& current_stats = stats_buffer.read();
            
            std::cout << "Trades: " << current_stats.trade_count 
                      << ", Volume: " << current_stats.total_volume
                      << ", Buy Ratio: " << current_stats.get_buy_ratio()
                      << ", Avg Price: " << current_stats.get_avg_price()
                      << ", Last Price: " << current_stats.last_price
                      << ", Updates: " << stats_buffer.get_update_count()
                      << "\n";
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    
    writer.join();
    reader.join();
    
    std::cout << "\nFinal update count: " << stats_buffer.get_update_count() << "\n";
    
    // Demonstrate the modify_and_swap functionality
    std::cout << "\n=== Using modify_and_swap ===\n";
    DoubleBufferedState<int> counter(0);
    
    // Increment the counter using a lambda function
    for (int i = 0; i < 5; ++i) {
        counter.modify_and_swap([](int& value) {
            value += 10;  // Add 10 to current value
        });
        
        std::cout << "Counter value: " << counter.read() << "\n";
    }
    
    // Demonstrate read_with functionality
    std::cout << "\n=== Using read_with ===\n";
    int result = counter.read_with([](const int& value) {
        return value * 2;  // Return double the value
    });
    std::cout << "Doubled value: " << result << "\n";
    
    std::cout << "\nUsage demonstration completed!\n";
}

int main() {
    demonstrate_usage();
    return 0;
}