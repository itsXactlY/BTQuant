#pragma once

#include <cstdint>
#include <vector>

#include "trading/HotspineData.h"

namespace BTQuant {
namespace RenderEngine {

/**
 * RenderSnapshot — flat, pre-computed, render-ready data.
 *
 * Workers build this from SymbolAnalytics under their existing lock,
 * then publish it to a TripleBuffer for lock-free render access.
 *
 * The render thread reads this with ZERO locks and ZERO copies.
 */
struct RenderSnapshot {
  // Identity
  uint32_t symbol_id = 0;
  uint64_t last_update_time = 0;

  // Pre-flattened orderbook (sorted, ready for GPU upload)
  struct OrderBookLevel {
    float price;
    float size;
  };
  std::vector<OrderBookLevel> bids;  // Descending by price
  std::vector<OrderBookLevel> asks;  // Ascending by price
  float ob_min_price = 0.0f;
  float ob_max_price = 0.0f;

  // Recent trades — pre-converted to GPU-ready format
  std::vector<HotspineTradeTick> trade_ticks;

  // Pre-aggregated footprint clusters
  std::vector<CandleCluster> footprint_clusters;

  // Basic price info
  double last_price = 0.0;
  double vwap = 0.0;
  double momentum = 0.0;
  double volatility = 0.0;

  // Clear all data (reuse allocated memory)
  void clear() {
    symbol_id = 0;
    last_update_time = 0;
    bids.clear();
    asks.clear();
    ob_min_price = 0.0f;
    ob_max_price = 0.0f;
    trade_ticks.clear();
    footprint_clusters.clear();
    last_price = 0.0;
    vwap = 0.0;
    momentum = 0.0;
    volatility = 0.0;
  }
};

}  // namespace RenderEngine
}  // namespace BTQuant
