#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include "task_scheduler.hpp"

using namespace btq;

class TaskSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        scheduler = std::make_unique<TaskScheduler>(4); // Use 4 threads for testing
    }

    void TearDown() override {
        scheduler.reset();
    }

    std::unique_ptr<TaskScheduler> scheduler;
};

TEST_F(TaskSchedulerTest, BasicTaskExecution) {
    bool task_executed = false;
    auto future = std::async(std::launch::async, [&]() {
        scheduler->enqueue_task([&task_executed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            task_executed = true;
        });
    });

    future.wait_for(std::chrono::seconds(1));
    EXPECT_TRUE(task_executed);
}

TEST_F(TaskSchedulerTest, VolumeProfileCalculation) {
    // Create sample trades
    std::vector<Trade> trades;
    for (int i = 0; i < 100; ++i) {
        Trade t;
        t.timestamp = std::chrono::system_clock::now() + std::chrono::seconds(i);
        t.price = 100.0 + (i % 10); // Prices between 100-109
        t.volume = 10.0 + (i % 5);  // Volumes between 10-14
        trades.push_back(t);
    }

    auto future = scheduler->calculate_volume_profile_async(trades, 100.0, 110.0, 10);
    auto result = future.get(); // Wait for result
    
    EXPECT_EQ(result.size(), 10);
    
    // Check that volumes are accumulated properly
    double total_volume = 0.0;
    for (double vol : result) {
        total_volume += vol;
    }
    
    double expected_total = 0.0;
    for (const auto& trade : trades) {
        expected_total += trade.volume;
    }
    
    EXPECT_NEAR(total_volume, expected_total, 0.001);
}

TEST_F(TaskSchedulerTest, VWAPCalculation) {
    // Create sample trades
    std::vector<Trade> trades;
    for (int i = 0; i < 10; ++i) {
        Trade t;
        t.timestamp = std::chrono::system_clock::now() + std::chrono::seconds(i);
        t.price = 100.0 + i;
        t.volume = 10.0 + i;
        trades.push_back(t);
    }

    auto future = scheduler->calculate_volume_weighted_average_price_async(trades);
    auto vwap = future.get(); // Wait for result
    
    // Calculate expected VWAP manually
    double total_value = 0.0;
    double total_volume = 0.0;
    for (const auto& trade : trades) {
        total_value += trade.price * trade.volume;
        total_volume += trade.volume;
    }
    double expected_vwap = total_value / total_volume;
    
    EXPECT_NEAR(vwap, expected_vwap, 0.001);
}

TEST_F(TaskSchedulerTest, SMACalculation) {
    std::vector<double> prices = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    int period = 3;
    
    auto future = scheduler->calculate_sma_async(prices, period);
    auto sma_result = future.get(); // Wait for result
    
    EXPECT_EQ(sma_result.size(), prices.size() - period + 1);
    
    // Verify first SMA value manually
    double expected_first_sma = (100 + 101 + 102) / 3.0;
    EXPECT_NEAR(sma_result[0], expected_first_sma, 0.001);
}

TEST_F(TaskSchedulerTest, EMACalculation) {
    std::vector<double> prices = {100, 101, 102, 103, 104};
    int period = 3;
    
    auto future = scheduler->calculate_ema_async(prices, period);
    auto ema_result = future.get(); // Wait for result
    
    EXPECT_GT(ema_result.size(), 0);
    
    // EMA should have at least one value calculated
    EXPECT_FALSE(std::isnan(ema_result[0]));
    EXPECT_GT(ema_result[0], 0);
}

TEST_F(TaskSchedulerTest, MultipleTasksConcurrent) {
    const int num_tasks = 10;
    std::vector<std::future<bool>> futures;
    
    for (int i = 0; i < num_tasks; ++i) {
        auto promise = std::make_shared<std::promise<bool>>();
        auto future = promise->get_future();
        
        scheduler->enqueue_task([promise, i]() {
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            promise->set_value(true);
        });
        
        futures.push_back(std::move(future));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        EXPECT_TRUE(future.get());
    }
}

TEST_F(TaskSchedulerTest, AggregateCandles) {
    // Create sample trades
    std::vector<Trade> trades;
    auto base_time = std::chrono::system_clock::now();
    for (int i = 0; i < 100; ++i) {
        Trade t;
        t.timestamp = base_time + std::chrono::seconds(i * 30); // Every 30 seconds
        t.price = 100.0 + (i % 5); // Oscillate between 100-104
        t.volume = 10.0;
        trades.push_back(t);
    }

    auto future = scheduler->aggregate_candles_async(trades, std::chrono::seconds(60)); // 1-minute candles
    auto candles = future.get(); // Wait for result
    
    // Should have approximately half the number of candles as trades (since we're grouping 2 trades per candle)
    EXPECT_GT(candles.size(), 0);
    EXPECT_LE(candles.size(), trades.size());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}