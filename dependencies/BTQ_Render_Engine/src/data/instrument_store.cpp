#include "../include/data/candle_aggregator.hpp"
#include "../include/hotspine_data_bridge.hpp"
#include "../include/market_data_processor.hpp" // For TimeFrame enum definition
#include <iostream>

namespace BTQuant {

// InstrumentStore implementation
InstrumentStore::InstrumentStore() {
  aggregator = std::make_unique<CandleAggregator>();
}

InstrumentStore::~InstrumentStore() = default;

void InstrumentStore::add_trade(double timestamp, float price, float size) {
  // Add to aggregator (handles all timeframes)
  aggregator->add_trade(timestamp, price, size);

  // Refresh candles for active timeframe
  refresh_candles();
}

void InstrumentStore::set_timeframe(RenderEngine::TimeFrame tf) {
  active_timeframe = tf;
  refresh_candles();
}

void InstrumentStore::refresh_candles() {
  // Get candles for active timeframe
  auto candles = aggregator->get_candles(active_timeframe, 10000);

  // Clear existing arrays
  timestamps.clear();
  opens.clear();
  highs.clear();
  lows.clear();
  closes.clear();
  volumes.clear();

  // Populate from aggregated candles
  timestamps.reserve(candles.size());
  opens.reserve(candles.size());
  highs.reserve(candles.size());
  lows.reserve(candles.size());
  closes.reserve(candles.size());
  volumes.reserve(candles.size());

  for (const auto &candle : candles) {
    timestamps.push_back(candle.timestamp);
    opens.push_back(candle.open);
    highs.push_back(candle.high);
    lows.push_back(candle.low);
    closes.push_back(candle.close);
    volumes.push_back(candle.volume);
  }
}

} // namespace BTQuant
