#include "../../../dependencies/BTQ_Render_Engine/include/analytics/cluster_engine.hpp"
#include "../../../dependencies/ccapi/example/src/market_data_collector/market_data_types.h"
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

using namespace Analytics;
using namespace MarketData;

void test_diagonal_imbalance_detection() {
  std::cout << "Testing diagonal imbalance detection..." << std::endl;

  // Create a ClusterEngine with tick size of 1.0 for simplicity
  ClusterEngine engine(1.0);

  // Create some test trades to simulate diagonal imbalances
  // Price level 100: lots of buy volume
  Trade trade1;
  trade1.price = 100.0;
  trade1.quantity = 100.0;
  trade1.is_buyer_maker = false; // buyer taker -> contributes to buy_volume
  trade1.timestamp_us = 1000000;

  // Price level 99: small sell volume
  Trade trade2;
  trade2.price = 99.0;
  trade2.quantity = 20.0;
  trade2.is_buyer_maker = true; // seller taker -> contributes to sell_volume
  trade2.timestamp_us = 1000001;

  // Process the trades
  engine.processTrade(trade1, 0); // time bucket 0
  engine.processTrade(trade2, 0); // time bucket 0

  // Add more buy volume at price 100 to increase the ratio
  Trade trade3 = trade1;
  trade3.timestamp_us = 1000002;
  engine.processTrade(trade3, 0);

  // Now detect diagonal imbalances with threshold of 3.0
  auto imbalances = engine.detect_diagonal_imbalances(3.0);

  std::cout << "Number of detected imbalances: " << imbalances.size()
            << std::endl;

  // We expect to find an imbalance where buy_volume at price 100 is much higher
  // than sell_volume at price 99
  bool found_expected_imbalance = false;
  for (const auto &imbalance : imbalances) {
    auto [price_index, time_bucket, buy_vol, sell_vol, ratio] = imbalance;

    std::cout << "Imbalance found: price_index=" << price_index
              << ", time_bucket=" << time_bucket << ", buy_vol=" << buy_vol
              << ", sell_vol=" << sell_vol << ", ratio=" << ratio << std::endl;

    // Check if this is the expected imbalance (buy_volume at 100 vs sell_volume
    // at 99)
    if (price_index == 100 && time_bucket == 0) {
      // The ratio should be buy_volume_at_100 / sell_volume_at_99
      // We had 200 buy volume at 100 (two trades of 100 each) and 20 sell
      // volume at 99 So ratio should be 200/20 = 10.0
      if (ratio >= 3.0) {
        found_expected_imbalance = true;
      }
    }
  }

  assert(found_expected_imbalance &&
         "Expected diagonal imbalance was not detected!");

  std::cout << "Diagonal imbalance detection test PASSED!" << std::endl;
}

void test_reverse_diagonal_imbalance_detection() {
  std::cout << "Testing reverse diagonal imbalance detection..." << std::endl;

  // Create a ClusterEngine with tick size of 1.0 for simplicity
  ClusterEngine engine(1.0);

  // Create some test trades to simulate reverse diagonal imbalances
  // Price level 100: lots of sell volume
  Trade trade1;
  trade1.price = 100.0;
  trade1.quantity = 150.0;
  trade1.is_buyer_maker = true; // seller taker -> contributes to sell_volume
  trade1.timestamp_us = 1000000;

  // Price level 99: small buy volume
  Trade trade2;
  trade2.price = 99.0;
  trade2.quantity = 25.0;
  trade2.is_buyer_maker = false; // buyer taker -> contributes to buy_volume
  trade2.timestamp_us = 1000001;

  // Process the trades
  engine.processTrade(trade1, 0); // time bucket 0
  engine.processTrade(trade2, 0); // time bucket 0

  // Now detect diagonal imbalances with threshold of 4.0
  auto imbalances = engine.detect_diagonal_imbalances(4.0);

  std::cout << "Number of detected reverse imbalances: " << imbalances.size()
            << std::endl;

  // We expect to find an imbalance where sell_volume at price 100 is much
  // higher than buy_volume at price 99
  bool found_expected_imbalance = false;
  for (const auto &imbalance : imbalances) {
    auto [price_index, time_bucket, sell_vol, buy_vol, ratio] = imbalance;

    std::cout << "Reverse imbalance found: price_index=" << price_index
              << ", time_bucket=" << time_bucket << ", sell_vol=" << sell_vol
              << ", buy_vol=" << buy_vol << ", ratio=" << ratio << std::endl;

    // The sell volume at price 100 should be much higher than buy volume at
    // price 99 Ratio should be 150/25 = 6.0 which is > 4.0 threshold
    if (ratio >= 4.0) {
      found_expected_imbalance = true;
    }
  }

  assert(found_expected_imbalance &&
         "Expected reverse diagonal imbalance was not detected!");

  std::cout << "Reverse diagonal imbalance detection test PASSED!" << std::endl;
}

void test_no_imbalance_below_threshold() {
  std::cout << "Testing that no imbalances are detected below threshold..."
            << std::endl;

  // Create a ClusterEngine with tick size of 1.0 for simplicity
  ClusterEngine engine(1.0);

  // Create some test trades where the ratio is below the threshold
  Trade trade1;
  trade1.price = 100.0;
  trade1.quantity = 50.0;
  trade1.is_buyer_maker = false; // contributes to buy_volume
  trade1.timestamp_us = 1000000;

  Trade trade2;
  trade2.price = 99.0;
  trade2.quantity = 30.0;
  trade2.is_buyer_maker = true; // contributes to sell_volume
  trade2.timestamp_us = 1000001;

  // Process the trades
  engine.processTrade(trade1, 0); // time bucket 0
  engine.processTrade(trade2, 0); // time bucket 0

  // Now detect diagonal imbalances with threshold of 3.0
  // The ratio would be 50/30 ≈ 1.67 which is less than 3.0
  auto imbalances = engine.detect_diagonal_imbalances(3.0);

  assert(imbalances.empty() &&
         "Unexpected imbalances detected below threshold!");

  std::cout << "No imbalance below threshold test PASSED!" << std::endl;
}

void test_stacked_imbalance_detection() {
  std::cout << "Testing stacked imbalance detection..." << std::endl;

  // Create a ClusterEngine with tick size of 1.0 for simplicity
  ClusterEngine engine(1.0);

  // Create some test trades to simulate stacked imbalances
  // Same price level (100), but across different time buckets
  // Time bucket 0: moderate buy volume at price 100
  Trade trade1;
  trade1.price = 100.0;
  trade1.quantity = 50.0;
  trade1.is_buyer_maker = false; // buyer taker -> contributes to buy_volume
  trade1.timestamp_us = 1000000;

  // Time bucket 1: much higher buy volume at same price level (100)
  Trade trade2;
  trade2.price = 100.0;
  trade2.quantity = 200.0;       // Much higher volume
  trade2.is_buyer_maker = false; // buyer taker -> contributes to buy_volume
  trade2.timestamp_us = 2000000;

  // Process the trades in different time buckets
  engine.processTrade(trade1, 0); // time bucket 0
  engine.processTrade(trade2, 1); // time bucket 1

  // Now detect stacked imbalances with threshold of 3.0
  auto imbalances = engine.detect_stacked_imbalances(3.0);

  std::cout << "Number of detected stacked imbalances: " << imbalances.size()
            << std::endl;

  // We expect to find an imbalance where buy_volume at time bucket 1 is much
  // higher than at time bucket 0
  bool found_expected_imbalance = false;
  for (const auto &imbalance : imbalances) {
    auto [price_index, time_bucket, current_vol, previous_vol, ratio] =
        imbalance;

    std::cout << "Stacked imbalance found: price_index=" << price_index
              << ", time_bucket=" << time_bucket
              << ", current_vol=" << current_vol
              << ", previous_vol=" << previous_vol << ", ratio=" << ratio
              << std::endl;

    // Check if this is the expected imbalance (higher volume in current bucket
    // vs previous)
    if (price_index == 100 &&
        time_bucket == 1) { // time bucket 1 compared to time bucket 0
      // The ratio should be current_vol / previous_vol = 200 / 50 = 4.0
      if (ratio >= 3.0) {
        found_expected_imbalance = true;
      }
    }
  }

  assert(found_expected_imbalance &&
         "Expected stacked imbalance was not detected!");

  std::cout << "Stacked imbalance detection test PASSED!" << std::endl;
}

void test_stacked_sell_imbalance_detection() {
  std::cout << "Testing stacked sell imbalance detection..." << std::endl;

  // Create a ClusterEngine with tick size of 1.0 for simplicity
  ClusterEngine engine(1.0);

  // Create some test trades to simulate stacked sell imbalances
  // Time bucket 0: moderate sell volume at price 100
  Trade trade1;
  trade1.price = 100.0;
  trade1.quantity = 30.0;
  trade1.is_buyer_maker = true; // seller taker -> contributes to sell_volume
  trade1.timestamp_us = 1000000;

  // Time bucket 1: much higher sell volume at same price level (100)
  Trade trade2;
  trade2.price = 100.0;
  trade2.quantity = 150.0;      // Much higher volume
  trade2.is_buyer_maker = true; // seller taker -> contributes to sell_volume
  trade2.timestamp_us = 2000000;

  // Process the trades in different time buckets
  engine.processTrade(trade1, 0); // time bucket 0
  engine.processTrade(trade2, 1); // time bucket 1

  // Now detect stacked imbalances with threshold of 4.0
  auto imbalances = engine.detect_stacked_imbalances(4.0);

  std::cout << "Number of detected stacked sell imbalances: "
            << imbalances.size() << std::endl;

  // We expect to find an imbalance where sell_volume at time bucket 1 is much
  // higher than at time bucket 0
  bool found_expected_imbalance = false;
  for (const auto &imbalance : imbalances) {
    auto [price_index, time_bucket, current_vol, previous_vol, ratio] =
        imbalance;

    std::cout << "Stacked sell imbalance found: price_index=" << price_index
              << ", time_bucket=" << time_bucket
              << ", current_vol=" << current_vol
              << ", previous_vol=" << previous_vol << ", ratio=" << ratio
              << std::endl;

    // Check if this is the expected imbalance (higher sell volume in current
    // bucket vs previous)
    if (price_index == 100 &&
        time_bucket == 1) { // time bucket 1 compared to time bucket 0
      // The ratio should be current_vol / previous_vol = 150 / 30 = 5.0
      if (ratio >= 4.0) {
        found_expected_imbalance = true;
      }
    }
  }

  assert(found_expected_imbalance &&
         "Expected stacked sell imbalance was not detected!");

  std::cout << "Stacked sell imbalance detection test PASSED!" << std::endl;
}

int main() {
  std::cout << "Running ClusterEngine diagonal and stacked imbalance detection "
               "tests..."
            << std::endl;

  test_diagonal_imbalance_detection();
  test_reverse_diagonal_imbalance_detection();
  test_no_imbalance_below_threshold();
  test_stacked_imbalance_detection();
  test_stacked_sell_imbalance_detection();

  std::cout << "All tests PASSED!" << std::endl;
  return 0;
}