#include <gtest/gtest.h>

#include "../include/data/exchange_aggregator.hpp"
#include "../include/market_data_processor.hpp"
#include "../include/symbol_manager.hpp"

using namespace BTQuant;
using namespace BTQuant::Data;

class ExchangeAggregatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create mock/shared objects for testing
    processor_ = std::make_shared<RenderEngine::MarketDataProcessor>();
    symbol_manager_ = std::make_shared<RenderEngine::SymbolManager>();

    aggregator_ = std::make_unique<ExchangeAggregator>(processor_, symbol_manager_);
    aggregator_->initialize();
  }

  void TearDown() override {
    aggregator_.reset();
  }

  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<RenderEngine::SymbolManager> symbol_manager_;
  std::unique_ptr<ExchangeAggregator> aggregator_;
};

TEST_F(ExchangeAggregatorTest, InitializeAndAddExchange) {
  EXPECT_TRUE(aggregator_->initialize());

  ExchangeFeatures features;
  features.exchange_name = "binance";
  features.latency_offset_us = 100.0;
  features.supported_symbols = {"BTCUSDT", "ETHUSDT"};
  features.reliability_score = 0.95;

  aggregator_->addExchange("binance", features);

  auto exchanges = aggregator_->getAvailableExchanges();
  EXPECT_EQ(exchanges.size(), 1);
  EXPECT_EQ(exchanges[0], "binance");
}

TEST_F(ExchangeAggregatorTest, ProcessDataFromMultipleExchanges) {
  // Add multiple exchanges
  ExchangeFeatures binance_features;
  binance_features.exchange_name = "binance";
  binance_features.latency_offset_us = 100.0;
  binance_features.reliability_score = 0.95;

  ExchangeFeatures coinbase_features;
  coinbase_features.exchange_name = "coinbase";
  coinbase_features.latency_offset_us = 150.0;
  coinbase_features.reliability_score = 0.90;

  aggregator_->addExchange("binance", binance_features);
  aggregator_->addExchange("coinbase", coinbase_features);

  // Create mock market data updates
  RenderEngine::MarketDataUpdate binance_update;
  binance_update.type = RenderEngine::MarketDataType::TRADE;
  binance_update.symbol_id = 1;
  binance_update.timestamp = 1704067200000000ULL;  // Jan 1, 2024
  binance_update.price = 40000.0;
  binance_update.size = 1.0;
  binance_update.side = "buy";

  RenderEngine::MarketDataUpdate coinbase_update;
  coinbase_update.type = RenderEngine::MarketDataType::TRADE;
  coinbase_update.symbol_id = 1;
  coinbase_update.timestamp = 1704067200000100ULL;  // Same time + 100us
  coinbase_update.price = 40010.0;
  coinbase_update.size = 0.5;
  coinbase_update.side = "sell";

  // Process data from both exchanges
  aggregator_->processDataUpdate("binance", "BTCUSDT", binance_update);
  aggregator_->processDataUpdate("coinbase", "BTCUSDT", coinbase_update);

  // Verify data was stored
  auto aggregated = aggregator_->getAggregatedData("BTCUSDT");
  ASSERT_TRUE(aggregated.has_value());
  EXPECT_EQ(aggregated->symbol, "BTCUSDT");
  EXPECT_EQ(aggregated->exchange_data.size(), 2);
}

TEST_F(ExchangeAggregatorTest, CalculateWeightedAveragePrice) {
  // Add exchanges with different reliability scores
  ExchangeFeatures binance_features;
  binance_features.exchange_name = "binance";
  binance_features.reliability_score = 0.95;

  ExchangeFeatures coinbase_features;
  coinbase_features.exchange_name = "coinbase";
  coinbase_features.reliability_score = 0.90;

  aggregator_->addExchange("binance", binance_features);
  aggregator_->addExchange("coinbase", coinbase_features);

  // Create mock data with different prices and volumes
  RenderEngine::MarketDataUpdate binance_update;
  binance_update.type = RenderEngine::MarketDataType::TRADE;
  binance_update.symbol_id = 1;
  binance_update.timestamp = 1704067200000000ULL;
  binance_update.price = 40000.0;
  binance_update.size = 2.0;  // Higher volume

  RenderEngine::MarketDataUpdate coinbase_update;
  coinbase_update.type = RenderEngine::MarketDataType::TRADE;
  coinbase_update.symbol_id = 1;
  coinbase_update.timestamp = 1704067200000050ULL;
  coinbase_update.price = 40010.0;
  coinbase_update.size = 1.0;  // Lower volume

  aggregator_->processDataUpdate("binance", "BTCUSDT", binance_update);
  aggregator_->processDataUpdate("coinbase", "BTCUSDT", coinbase_update);

  // Calculate weighted average
  double weighted_avg = aggregator_->calculateWeightedAveragePrice("BTCUSDT");

  // The result should be closer to binance's price due to higher volume
  EXPECT_GT(weighted_avg, 40000.0);
  EXPECT_LT(weighted_avg, 40010.0);
}

TEST_F(ExchangeAggregatorTest, TimeSynchronizationStrategies) {
  // Add exchanges
  ExchangeFeatures binance_features;
  binance_features.exchange_name = "binance";
  binance_features.latency_offset_us = 100.0;

  ExchangeFeatures coinbase_features;
  coinbase_features.exchange_name = "coinbase";
  coinbase_features.latency_offset_us = 200.0;

  aggregator_->addExchange("binance", binance_features);
  aggregator_->addExchange("coinbase", coinbase_features);

  // Create mock data with different timestamps
  RenderEngine::MarketDataUpdate binance_update;
  binance_update.type = RenderEngine::MarketDataType::TRADE;
  binance_update.symbol_id = 1;
  binance_update.timestamp = 1704067200000000ULL;
  binance_update.price = 40000.0;
  binance_update.size = 1.0;

  RenderEngine::MarketDataUpdate coinbase_update;
  coinbase_update.type = RenderEngine::MarketDataType::TRADE;
  coinbase_update.symbol_id = 1;
  coinbase_update.timestamp = 1704067200000100ULL;  // Later timestamp
  coinbase_update.price = 40010.0;
  coinbase_update.size = 1.0;

  aggregator_->processDataUpdate("binance", "BTCUSDT", binance_update);
  aggregator_->processDataUpdate("coinbase", "BTCUSDT", coinbase_update);

  // Test different synchronization strategies
  aggregator_->setTimeSyncStrategy(TimeSyncStrategy::EARLIEST_TIMESTAMP);
  uint64_t earliest_ts = aggregator_->calculateSynchronizedTimestamp("BTCUSDT");
  EXPECT_EQ(earliest_ts, 1704067200000000ULL);

  aggregator_->setTimeSyncStrategy(TimeSyncStrategy::LATEST_TIMESTAMP);
  uint64_t latest_ts = aggregator_->calculateSynchronizedTimestamp("BTCUSDT");
  EXPECT_EQ(latest_ts, 1704067200000100ULL);

  aggregator_->setTimeSyncStrategy(TimeSyncStrategy::AVERAGE_TIMESTAMP);
  uint64_t avg_ts = aggregator_->calculateSynchronizedTimestamp("BTCUSTS");
  EXPECT_GT(avg_ts, 1704067200000000ULL);
  EXPECT_LT(avg_ts, 1704067200000100ULL);

  aggregator_->setTimeSyncStrategy(TimeSyncStrategy::OFFSET_COMPENSATION);
  uint64_t offset_ts = aggregator_->calculateSynchronizedTimestamp("BTCUSDT");
  // With offset compensation, the result should be adjusted based on latency offsets
  EXPECT_GT(offset_ts, 0);
}

TEST_F(ExchangeAggregatorTest, RemoveExchange) {
  ExchangeFeatures features;
  features.exchange_name = "binance";

  aggregator_->addExchange("binance", features);
  EXPECT_EQ(aggregator_->getAvailableExchanges().size(), 1);

  aggregator_->removeExchange("binance");
  EXPECT_EQ(aggregator_->getAvailableExchanges().size(), 0);
}

TEST_F(ExchangeAggregatorTest, GetExchangeFeatures) {
  ExchangeFeatures features;
  features.exchange_name = "binance";
  features.latency_offset_us = 100.0;
  features.reliability_score = 0.95;

  aggregator_->addExchange("binance", features);

  auto retrieved_features = aggregator_->getExchangeFeatures("binance");
  ASSERT_TRUE(retrieved_features.has_value());
  EXPECT_EQ(retrieved_features->exchange_name, "binance");
  EXPECT_DOUBLE_EQ(retrieved_features->latency_offset_us, 100.0);
  EXPECT_DOUBLE_EQ(retrieved_features->reliability_score, 0.95);
}