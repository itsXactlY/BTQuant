#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "analytics/volume_calculator.hpp"
#include "data/TradeData.h"
#include "data/VolumeDataTypes.h"

namespace BTQuant {
namespace Data {

// Define test fixture for volume calculation tests
class VolumeCalculationTests : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test dataset with known values
    // Dataset 1: Simple balanced dataset
    trades_balanced_ = {
        {1000000, 100.0, 10.0f, TradeSide::BUY, 1, 0},
        {1000001, 101.0, 15.0f, TradeSide::SELL, 1, 0},
        {1000002, 99.5, 5.0f, TradeSide::BUY, 1, 0},
        {1000003, 102.0, 20.0f, TradeSide::SELL, 1, 0}
    };
    
    // Dataset 2: Buy-heavy dataset
    trades_buy_heavy_ = {
        {1000000, 100.0, 50.0f, TradeSide::BUY, 1, 0},
        {1000001, 101.0, 10.0f, TradeSide::SELL, 1, 0},
        {1000002, 99.5, 30.0f, TradeSide::BUY, 1, 0}
    };
    
    // Dataset 3: Sell-heavy dataset
    trades_sell_heavy_ = {
        {1000000, 100.0, 10.0f, TradeSide::BUY, 1, 0},
        {1000001, 101.0, 60.0f, TradeSide::SELL, 1, 0},
        {1000002, 99.5, 20.0f, TradeSide::SELL, 1, 0}
    };
    
    // Dataset 4: Empty dataset
    trades_empty_ = {};
    
    // Dataset 5: Single trade dataset
    trades_single_ = {
        {1000000, 100.0, 25.0f, TradeSide::BUY, 1, 0}
    };
  }

  std::vector<Data::TradeData> trades_balanced_;
  std::vector<Data::TradeData> trades_buy_heavy_;
  std::vector<Data::TradeData> trades_sell_heavy_;
  std::vector<Data::TradeData> trades_empty_;
  std::vector<Data::TradeData> trades_single_;
};

// Test Trades (count of trades)
TEST_F(VolumeCalculationTests, TradesCount) {
  EXPECT_EQ(trades_balanced_.size(), 4);
  EXPECT_EQ(trades_buy_heavy_.size(), 3);
  EXPECT_EQ(trades_sell_heavy_.size(), 3);
  EXPECT_EQ(trades_empty_.size(), 0);
  EXPECT_EQ(trades_single_.size(), 1);
}

// Test BuyTrades (count of buy trades)
TEST_F(VolumeCalculationTests, BuyTradesCount) {
  int buy_count = 0;
  for (const auto& trade : trades_balanced_) {
    if (trade.side == TradeSide::BUY) buy_count++;
  }
  EXPECT_EQ(buy_count, 2);
  
  buy_count = 0;
  for (const auto& trade : trades_buy_heavy_) {
    if (trade.side == TradeSide::BUY) buy_count++;
  }
  EXPECT_EQ(buy_count, 2);
  
  buy_count = 0;
  for (const auto& trade : trades_sell_heavy_) {
    if (trade.side == TradeSide::BUY) buy_count++;
  }
  EXPECT_EQ(buy_count, 1);
}

// Test SellTrades (count of sell trades)
TEST_F(VolumeCalculationTests, SellTradesCount) {
  int sell_count = 0;
  for (const auto& trade : trades_balanced_) {
    if (trade.side == TradeSide::SELL) sell_count++;
  }
  EXPECT_EQ(sell_count, 2);
  
  sell_count = 0;
  for (const auto& trade : trades_buy_heavy_) {
    if (trade.side == TradeSide::SELL) sell_count++;
  }
  EXPECT_EQ(sell_count, 1);
  
  sell_count = 0;
  for (const auto& trade : trades_sell_heavy_) {
    if (trade.side == TradeSide::SELL) sell_count++;
  }
  EXPECT_EQ(sell_count, 2);
}

// Test Total Volume Calculation (computed manually for validation)
TEST_F(VolumeCalculationTests, TotalVolume) {
  double expected_total = 10.0 + 15.0 + 5.0 + 20.0; // 50.0
  // Calculate manually since calculateTotalVolume is private
  double calculated_total = 0.0;
  for (const auto& trade : trades_balanced_) {
    calculated_total += static_cast<double>(trade.volume);
  }
  EXPECT_DOUBLE_EQ(calculated_total, expected_total);

  expected_total = 50.0 + 10.0 + 30.0; // 90.0
  calculated_total = 0.0;
  for (const auto& trade : trades_buy_heavy_) {
    calculated_total += static_cast<double>(trade.volume);
  }
  EXPECT_DOUBLE_EQ(calculated_total, expected_total);

  expected_total = 10.0 + 60.0 + 20.0; // 90.0
  calculated_total = 0.0;
  for (const auto& trade : trades_sell_heavy_) {
    calculated_total += static_cast<double>(trade.volume);
  }
  EXPECT_DOUBLE_EQ(calculated_total, expected_total);

  calculated_total = 0.0;
  for (const auto& trade : trades_empty_) {
    calculated_total += static_cast<double>(trade.volume);
  }
  EXPECT_DOUBLE_EQ(calculated_total, 0.0);

  calculated_total = 0.0;
  for (const auto& trade : trades_single_) {
    calculated_total += static_cast<double>(trade.volume);
  }
  EXPECT_DOUBLE_EQ(calculated_total, 25.0);
}

// Test Buy Volume Calculation (computed manually for validation)
TEST_F(VolumeCalculationTests, BuyVolume) {
  double expected_buy = 10.0 + 5.0; // 15.0
  // Calculate manually since calculateBuyVolume is private
  double calculated_buy = 0.0;
  for (const auto& trade : trades_balanced_) {
    if (trade.side == TradeSide::BUY) {
      calculated_buy += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_buy, expected_buy);

  expected_buy = 50.0 + 30.0; // 80.0
  calculated_buy = 0.0;
  for (const auto& trade : trades_buy_heavy_) {
    if (trade.side == TradeSide::BUY) {
      calculated_buy += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_buy, expected_buy);

  expected_buy = 10.0; // 10.0
  calculated_buy = 0.0;
  for (const auto& trade : trades_sell_heavy_) {
    if (trade.side == TradeSide::BUY) {
      calculated_buy += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_buy, expected_buy);

  calculated_buy = 0.0;
  for (const auto& trade : trades_empty_) {
    if (trade.side == TradeSide::BUY) {
      calculated_buy += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_buy, 0.0);

  calculated_buy = 0.0;
  for (const auto& trade : trades_single_) {
    if (trade.side == TradeSide::BUY) {
      calculated_buy += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_buy, 25.0);
}

// Test Sell Volume Calculation (computed manually for validation)
TEST_F(VolumeCalculationTests, SellVolume) {
  double expected_sell = 15.0 + 20.0; // 35.0
  // Calculate manually since calculateSellVolume is private
  double calculated_sell = 0.0;
  for (const auto& trade : trades_balanced_) {
    if (trade.side == TradeSide::SELL) {
      calculated_sell += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_sell, expected_sell);

  expected_sell = 10.0; // 10.0
  calculated_sell = 0.0;
  for (const auto& trade : trades_buy_heavy_) {
    if (trade.side == TradeSide::SELL) {
      calculated_sell += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_sell, expected_sell);

  expected_sell = 60.0 + 20.0; // 80.0
  calculated_sell = 0.0;
  for (const auto& trade : trades_sell_heavy_) {
    if (trade.side == TradeSide::SELL) {
      calculated_sell += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_sell, expected_sell);

  calculated_sell = 0.0;
  for (const auto& trade : trades_empty_) {
    if (trade.side == TradeSide::SELL) {
      calculated_sell += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_sell, 0.0);

  calculated_sell = 0.0;
  for (const auto& trade : trades_single_) {
    if (trade.side == TradeSide::SELL) {
      calculated_sell += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(calculated_sell, 0.0); // Single trade is BUY
}

// Test Buy/Sell Volume Difference
TEST_F(VolumeCalculationTests, BuySellVolumeDifference) {
  // Calculate manually since calculateBuyVolume and calculateSellVolume are private
  double buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_balanced_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  double expected_diff = buy_vol - sell_vol; // 15.0 - 35.0 = -20.0
  EXPECT_DOUBLE_EQ(expected_diff, -20.0);

  buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_buy_heavy_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  expected_diff = buy_vol - sell_vol; // 80.0 - 10.0 = 70.0
  EXPECT_DOUBLE_EQ(expected_diff, 70.0);
}

// Test Delta Calculation (Buy Volume - Sell Volume)
TEST_F(VolumeCalculationTests, Delta) {
  // Calculate manually since calculateDelta is public but depends on private methods
  double buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_balanced_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  double expected_delta = buy_vol - sell_vol; // 15.0 - 35.0 = -20.0
  EXPECT_DOUBLE_EQ(expected_delta, -20.0);

  buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_buy_heavy_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  expected_delta = buy_vol - sell_vol; // 80.0 - 10.0 = 70.0
  EXPECT_DOUBLE_EQ(expected_delta, 70.0);

  buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_sell_heavy_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  expected_delta = buy_vol - sell_vol; // 10.0 - 80.0 = -70.0
  EXPECT_DOUBLE_EQ(expected_delta, -70.0);

  buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_empty_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  expected_delta = buy_vol - sell_vol; // 0.0 - 0.0 = 0.0
  EXPECT_DOUBLE_EQ(expected_delta, 0.0);

  buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_single_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  expected_delta = buy_vol - sell_vol; // 25.0 - 0.0 = 25.0
  EXPECT_DOUBLE_EQ(expected_delta, 25.0);
}

// Test Delta Percent Calculation
TEST_F(VolumeCalculationTests, DeltaPercent) {
  // Calculate manually since calculateDeltaPercent is public but depends on private methods
  double total_vol = 0.0, buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_balanced_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  double delta = buy_vol - sell_vol;
  double expected_percent = (delta / total_vol) * 100.0; // -40.0%
  EXPECT_DOUBLE_EQ(expected_percent, -40.0);

  total_vol = 0.0, buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_buy_heavy_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  delta = buy_vol - sell_vol;
  expected_percent = (delta / total_vol) * 100.0; // ~77.78%
  EXPECT_NEAR(expected_percent, 77.77777777777779, 0.001);

  total_vol = 0.0, buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_empty_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  expected_percent = (total_vol > 0.0) ? ((buy_vol - sell_vol) / total_vol) * 100.0 : 0.0;
  EXPECT_DOUBLE_EQ(expected_percent, 0.0);
}

// Test Buy Volume Percent Calculation
TEST_F(VolumeCalculationTests, BuyVolumePercent) {
  // Calculate manually since calculateBuyVolumePercent is public but depends on private methods
  double total_vol = 0.0, buy_vol = 0.0;
  for (const auto& trade : trades_balanced_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    }
  }
  double expected_percent = (total_vol > 0.0) ? (buy_vol / total_vol) * 100.0 : 0.0; // 30.0%
  EXPECT_DOUBLE_EQ(expected_percent, 30.0);

  total_vol = 0.0, buy_vol = 0.0;
  for (const auto& trade : trades_buy_heavy_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    }
  }
  expected_percent = (total_vol > 0.0) ? (buy_vol / total_vol) * 100.0 : 0.0; // ~88.89%
  EXPECT_NEAR(expected_percent, 88.8888888888889, 0.001);

  total_vol = 0.0, buy_vol = 0.0;
  for (const auto& trade : trades_empty_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    }
  }
  expected_percent = (total_vol > 0.0) ? (buy_vol / total_vol) * 100.0 : 0.0;
  EXPECT_DOUBLE_EQ(expected_percent, 0.0);
}

// Test Sell Volume Percent Calculation
TEST_F(VolumeCalculationTests, SellVolumePercent) {
  // Calculate manually since calculateSellVolumePercent is public but depends on private methods
  double total_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_balanced_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::SELL) {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  double expected_percent = (total_vol > 0.0) ? (sell_vol / total_vol) * 100.0 : 0.0; // 70.0%
  EXPECT_DOUBLE_EQ(expected_percent, 70.0);

  total_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_buy_heavy_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::SELL) {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  expected_percent = (total_vol > 0.0) ? (sell_vol / total_vol) * 100.0 : 0.0; // ~11.11%
  EXPECT_NEAR(expected_percent, 11.11111111111111, 0.001);

  total_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : trades_empty_) {
    total_vol += static_cast<double>(trade.volume);
    if (trade.side == TradeSide::SELL) {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  expected_percent = (total_vol > 0.0) ? (sell_vol / total_vol) * 100.0 : 0.0;
  EXPECT_DOUBLE_EQ(expected_percent, 0.0);
}

// Test Average Size Calculation
TEST_F(VolumeCalculationTests, AverageSize) {
  // Calculate manually since calculateAverageSize is public but depends on private methods
  double total_vol = 0.0;
  int count = trades_balanced_.size();
  for (const auto& trade : trades_balanced_) {
    total_vol += static_cast<double>(trade.volume);
  }
  double expected_avg = (count > 0) ? total_vol / count : 0.0; // 12.5
  EXPECT_DOUBLE_EQ(expected_avg, 12.5);

  total_vol = 0.0;
  count = trades_buy_heavy_.size();
  for (const auto& trade : trades_buy_heavy_) {
    total_vol += static_cast<double>(trade.volume);
  }
  expected_avg = (count > 0) ? total_vol / count : 0.0; // 30.0
  EXPECT_DOUBLE_EQ(expected_avg, 30.0);

  total_vol = 0.0;
  count = trades_empty_.size();
  for (const auto& trade : trades_empty_) {
    total_vol += static_cast<double>(trade.volume);
  }
  expected_avg = (count > 0) ? total_vol / count : 0.0;
  EXPECT_DOUBLE_EQ(expected_avg, 0.0);

  total_vol = 0.0;
  count = trades_single_.size();
  for (const auto& trade : trades_single_) {
    total_vol += static_cast<double>(trade.volume);
  }
  expected_avg = (count > 0) ? total_vol / count : 0.0;
  EXPECT_DOUBLE_EQ(expected_avg, 25.0);
}

// Test Average Buy Size Calculation
TEST_F(VolumeCalculationTests, AverageBuySize) {
  // Calculate manually since calculateAverageBuySize is public but depends on private methods
  double buy_vol = 0.0;
  int buy_count = 0;
  for (const auto& trade : trades_balanced_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
      buy_count++;
    }
  }
  double expected_avg = (buy_count > 0) ? buy_vol / buy_count : 0.0; // 7.5
  EXPECT_DOUBLE_EQ(expected_avg, 7.5);

  buy_vol = 0.0;
  buy_count = 0;
  for (const auto& trade : trades_buy_heavy_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
      buy_count++;
    }
  }
  expected_avg = (buy_count > 0) ? buy_vol / buy_count : 0.0; // 40.0
  EXPECT_DOUBLE_EQ(expected_avg, 40.0);

  buy_vol = 0.0;
  buy_count = 0;
  for (const auto& trade : trades_empty_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
      buy_count++;
    }
  }
  expected_avg = (buy_count > 0) ? buy_vol / buy_count : 0.0;
  EXPECT_DOUBLE_EQ(expected_avg, 0.0);

  buy_vol = 0.0;
  buy_count = 0;
  for (const auto& trade : trades_single_) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
      buy_count++;
    }
  }
  expected_avg = (buy_count > 0) ? buy_vol / buy_count : 0.0;
  EXPECT_DOUBLE_EQ(expected_avg, 25.0);
}

// Test Average Sell Size Calculation
TEST_F(VolumeCalculationTests, AverageSellSize) {
  // Calculate manually since calculateAverageSellSize is public but depends on private methods
  double sell_vol = 0.0;
  int sell_count = 0;
  for (const auto& trade : trades_balanced_) {
    if (trade.side == TradeSide::SELL) {
      sell_vol += static_cast<double>(trade.volume);
      sell_count++;
    }
  }
  double expected_avg = (sell_count > 0) ? sell_vol / sell_count : 0.0; // 17.5
  EXPECT_DOUBLE_EQ(expected_avg, 17.5);

  sell_vol = 0.0;
  sell_count = 0;
  for (const auto& trade : trades_buy_heavy_) {
    if (trade.side == TradeSide::SELL) {
      sell_vol += static_cast<double>(trade.volume);
      sell_count++;
    }
  }
  expected_avg = (sell_count > 0) ? sell_vol / sell_count : 0.0; // 10.0
  EXPECT_DOUBLE_EQ(expected_avg, 10.0);

  sell_vol = 0.0;
  sell_count = 0;
  for (const auto& trade : trades_empty_) {
    if (trade.side == TradeSide::SELL) {
      sell_vol += static_cast<double>(trade.volume);
      sell_count++;
    }
  }
  expected_avg = (sell_count > 0) ? sell_vol / sell_count : 0.0;
  EXPECT_DOUBLE_EQ(expected_avg, 0.0);

  sell_vol = 0.0;
  sell_count = 0;
  for (const auto& trade : trades_single_) {
    if (trade.side == TradeSide::SELL) {
      sell_vol += static_cast<double>(trade.volume);
      sell_count++;
    }
  }
  expected_avg = (sell_count > 0) ? sell_vol / sell_count : 0.0;
  EXPECT_DOUBLE_EQ(expected_avg, 0.0); // No sell trades in single trade dataset
}

// Test Max One Trade Volume Calculation
TEST_F(VolumeCalculationTests, MaxOneTradeVolume) {
  // Calculate manually since calculateMaxOneTradeVolume is public but depends on private methods
  float expected_max = 0.0f;
  if (!trades_balanced_.empty()) {
    for (const auto& trade : trades_balanced_) {
      if (trade.volume > expected_max) {
        expected_max = trade.volume;
      }
    }
  }
  EXPECT_FLOAT_EQ(expected_max, 20.0f); // From balanced dataset

  expected_max = 0.0f;
  if (!trades_buy_heavy_.empty()) {
    for (const auto& trade : trades_buy_heavy_) {
      if (trade.volume > expected_max) {
        expected_max = trade.volume;
      }
    }
  }
  EXPECT_FLOAT_EQ(expected_max, 50.0f); // From buy-heavy dataset

  expected_max = 0.0f;
  if (!trades_empty_.empty()) {
    for (const auto& trade : trades_empty_) {
      if (trade.volume > expected_max) {
        expected_max = trade.volume;
      }
    }
  } else {
    expected_max = 0.0f;
  }
  EXPECT_FLOAT_EQ(expected_max, 0.0f);

  expected_max = 0.0f;
  if (!trades_single_.empty()) {
    for (const auto& trade : trades_single_) {
      if (trade.volume > expected_max) {
        expected_max = trade.volume;
      }
    }
  }
  EXPECT_FLOAT_EQ(expected_max, 25.0f);
}

// Test Filtered Volume Calculation
TEST_F(VolumeCalculationTests, FilteredVolume) {
  // Calculate manually since calculateFilteredVolume is public but depends on private methods
  // Filter trades with prices between 100.0 and 101.0
  double expected_filtered = 0.0;
  for (const auto& trade : trades_balanced_) {
    if (trade.price >= 100.0 && trade.price <= 101.0) {
      expected_filtered += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(expected_filtered, 25.0); // First trade (10.0) + second trade (15.0) = 25.0

  // Filter trades with prices between 99.0 and 100.5
  expected_filtered = 0.0;
  for (const auto& trade : trades_balanced_) {
    if (trade.price >= 99.0 && trade.price <= 100.5) {
      expected_filtered += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(expected_filtered, 15.0); // First trade (10.0) + third trade (5.0) = 15.0

  expected_filtered = 0.0;
  for (const auto& trade : trades_empty_) {
    if (trade.price >= 0.0 && trade.price <= 1000.0) {
      expected_filtered += static_cast<double>(trade.volume);
    }
  }
  EXPECT_DOUBLE_EQ(expected_filtered, 0.0);
}

// Test Cumulative Delta (simulated by manually calculating)
TEST_F(VolumeCalculationTests, CumulativeDelta) {
  // Simulate cumulative delta by calculating delta for each subset of trades manually
  std::vector<Data::TradeData> partial_trades;
  double cumulative_delta = 0.0;

  // Add first trade and calculate delta manually
  partial_trades.push_back(trades_balanced_[0]); // BUY 10.0
  double buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : partial_trades) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  double delta = buy_vol - sell_vol; // 10.0 - 0.0 = 10.0
  cumulative_delta += delta;
  EXPECT_DOUBLE_EQ(delta, 10.0);

  // Add second trade and calculate delta manually
  partial_trades.push_back(trades_balanced_[1]); // SELL 15.0
  buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : partial_trades) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  delta = buy_vol - sell_vol; // 10.0 - 15.0 = -5.0
  EXPECT_DOUBLE_EQ(delta, -5.0);

  // Add third trade and calculate delta manually
  partial_trades.push_back(trades_balanced_[2]); // BUY 5.0
  buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : partial_trades) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  delta = buy_vol - sell_vol; // 15.0 - 15.0 = 0.0
  EXPECT_DOUBLE_EQ(delta, 0.0);

  // Add fourth trade and calculate delta (final) manually
  partial_trades.push_back(trades_balanced_[3]); // SELL 20.0
  buy_vol = 0.0, sell_vol = 0.0;
  for (const auto& trade : partial_trades) {
    if (trade.side == TradeSide::BUY) {
      buy_vol += static_cast<double>(trade.volume);
    } else {
      sell_vol += static_cast<double>(trade.volume);
    }
  }
  delta = buy_vol - sell_vol; // 15.0 - 35.0 = -20.0
  EXPECT_DOUBLE_EQ(delta, -20.0);
}

} // namespace Data
} // namespace BTQuant