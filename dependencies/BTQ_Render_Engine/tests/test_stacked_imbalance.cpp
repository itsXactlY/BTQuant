#include "../include/analytics/cluster_engine.hpp"
#include "../../ccapi/example/src/market_data_collector/market_data_types.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <tuple>

using namespace Analytics;
using namespace MarketData;

void test_stacked_imbalance_detection() {
    std::cout << "Testing Stacked Imbalance Detection...\n";

    // Create a cluster engine with tick size of 1.0
    ClusterEngine engine(1.0);

    // Create test trades at the same price level but different time buckets
    // Time bucket 0
    Trade trade1;  // Price: 100, Quantity: 10, Buy
    trade1.price = 100.0;
    trade1.quantity = 10.0;
    trade1.timestamp_us = 1000000;
    trade1.is_buyer_maker = true;
    trade1.exchange = "TEST";
    trade1.symbol = "TEST";
    trade1.market_type = "spot";
    trade1.trade_id = "1";

    Trade trade2;  // Price: 100, Quantity: 5, Sell
    trade2.price = 100.0;
    trade2.quantity = 5.0;
    trade2.timestamp_us = 1000001;
    trade2.is_buyer_maker = false;
    trade2.exchange = "TEST";
    trade2.symbol = "TEST";
    trade2.market_type = "spot";
    trade2.trade_id = "2";

    // Time bucket 1 (same price, more volume)
    Trade trade3;  // Price: 100, Quantity: 50, Buy - significant increase
    trade3.price = 100.0;
    trade3.quantity = 50.0;
    trade3.timestamp_us = 2000000;
    trade3.is_buyer_maker = true;
    trade3.exchange = "TEST";
    trade3.symbol = "TEST";
    trade3.market_type = "spot";
    trade3.trade_id = "3";

    Trade trade4;  // Price: 100, Quantity: 5, Sell
    trade4.price = 100.0;
    trade4.quantity = 5.0;
    trade4.timestamp_us = 2000001;
    trade4.is_buyer_maker = false;
    trade4.exchange = "TEST";
    trade4.symbol = "TEST";
    trade4.market_type = "spot";
    trade4.trade_id = "4";

    // Process trades in different time buckets
    engine.processTrade(trade1, 0);  // Time bucket 0
    engine.processTrade(trade2, 0);  // Time bucket 0
    engine.processTrade(trade3, 1);  // Time bucket 1
    engine.processTrade(trade4, 1);  // Time bucket 1

    // Detect stacked imbalances with threshold of 3.0
    auto imbalances = engine.detect_stacked_imbalances(3.0);

    std::cout << "Found " << imbalances.size() << " imbalances\n";

    // Check if we detected the significant increase in buy volume at the same price level
    bool found_imbalance = false;
    for (const auto& imbalance : imbalances) {
        auto [price_idx, time_bucket, current_volume, previous_volume, ratio] = imbalance;

        std::cout << "Imbalance found: Price Index=" << price_idx
                  << ", Time Bucket=" << time_bucket
                  << ", Current Vol=" << current_volume
                  << ", Previous Vol=" << previous_volume
                  << ", Ratio=" << ratio << std::endl;

        // Check if we found a buy volume imbalance (current buy volume much higher than previous)
        if (time_bucket == 1 && ratio > 3.0) {
            found_imbalance = true;
        }
    }

    assert(found_imbalance && "Should detect stacked imbalance when buy volume increases significantly");

    std::cout << "Stacked imbalance detection test PASSED!\n\n";
}

void test_cross_stacked_imbalance() {
    std::cout << "Testing Cross Stacked Imbalance Detection...\n";

    // Create a cluster engine with tick size of 1.0
    ClusterEngine engine(1.0);

    // Create trades to test cross-imbalance (current buy vs previous sell)
    // Time bucket 0: More sell volume
    Trade trade1;  // Sell at price 100
    trade1.price = 100.0;
    trade1.quantity = 5.0;
    trade1.timestamp_us = 1000000;
    trade1.is_buyer_maker = false;
    trade1.exchange = "TEST";
    trade1.symbol = "TEST";
    trade1.market_type = "spot";
    trade1.trade_id = "1";

    Trade trade2;  // Buy at price 100
    trade2.price = 100.0;
    trade2.quantity = 1.0;
    trade2.timestamp_us = 1000001;
    trade2.is_buyer_maker = true;
    trade2.exchange = "TEST";
    trade2.symbol = "TEST";
    trade2.market_type = "spot";
    trade2.trade_id = "2";

    // Time bucket 1: More buy volume (potential bullish reversal)
    Trade trade3;  // Sell at price 100
    trade3.price = 100.0;
    trade3.quantity = 1.0;
    trade3.timestamp_us = 2000000;
    trade3.is_buyer_maker = false;
    trade3.exchange = "TEST";
    trade3.symbol = "TEST";
    trade3.market_type = "spot";
    trade3.trade_id = "3";

    Trade trade4;  // Buy at price 100 - much higher
    trade4.price = 100.0;
    trade4.quantity = 20.0;
    trade4.timestamp_us = 2000001;
    trade4.is_buyer_maker = true;
    trade4.exchange = "TEST";
    trade4.symbol = "TEST";
    trade4.market_type = "spot";
    trade4.trade_id = "4";

    // Process trades
    engine.processTrade(trade1, 0);  // Time bucket 0
    engine.processTrade(trade2, 0);  // Time bucket 0
    engine.processTrade(trade3, 1);  // Time bucket 1
    engine.processTrade(trade4, 1);  // Time bucket 1

    // Detect stacked imbalances with threshold of 3.0
    auto imbalances = engine.detect_stacked_imbalances(3.0);

    std::cout << "Found " << imbalances.size() << " cross imbalances\n";

    bool found_cross_imbalance = false;
    for (const auto& imbalance : imbalances) {
        auto [price_idx, time_bucket, current_volume, previous_volume, ratio] = imbalance;

        std::cout << "Cross Imbalance found: Price Index=" << price_idx
                  << ", Time Bucket=" << time_bucket
                  << ", Current Vol=" << current_volume
                  << ", Previous Vol=" << previous_volume
                  << ", Ratio=" << ratio << std::endl;

        // Check if we found a cross imbalance (current buy vs previous sell)
        if (time_bucket == 1 && ratio > 3.0) {
            found_cross_imbalance = true;
        }
    }

    assert(found_cross_imbalance && "Should detect cross stacked imbalance");

    std::cout << "Cross stacked imbalance detection test PASSED!\n\n";
}

int main() {
    test_stacked_imbalance_detection();
    test_cross_stacked_imbalance();
    
    std::cout << "All stacked imbalance tests PASSED!\n";
    return 0;
}