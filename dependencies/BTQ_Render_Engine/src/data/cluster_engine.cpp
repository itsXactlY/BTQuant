#include "data/cluster_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace BTQuant {

// ============================================================================
// Construction
// ============================================================================

ClusterEngine::ClusterEngine(uint64_t candle_duration_us)
    : candle_duration_us_(candle_duration_us) {}

// ============================================================================
// Tick Size Helpers
// ============================================================================

double ClusterEngine::detect_tick_size(double price) const {
  // Auto-detect sensible tick size based on price magnitude
  if (price >= 10000.0) return 1.0;  // BTC → $1 ticks
  if (price >= 1000.0) return 0.10;  // ETH → $0.10 ticks
  if (price >= 100.0) return 0.01;   // SOL → $0.01 ticks
  if (price >= 10.0) return 0.01;    // BNB-range
  if (price >= 1.0) return 0.001;    // Mid-caps
  if (price >= 0.01) return 0.0001;  // DOGE-range
  return 0.00001;                    // Sub-cent tokens
}

int64_t ClusterEngine::price_to_tick(double price, double tick_size) const {
  return static_cast<int64_t>(std::round(price / tick_size));
}

double ClusterEngine::tick_to_price(int64_t tick, double tick_size) const {
  return tick * tick_size;
}

// ============================================================================
// Config
// ============================================================================

void ClusterEngine::set_tick_size(uint32_t symbol_id, double tick_size) {
  if (symbol_id >= MAX_SYMBOLS) return;
  auto& state = symbols_[symbol_id];
  std::lock_guard<std::mutex> lock(state.mu);
  state.tick_size = tick_size;
  state.tick_detected = (tick_size > 0.0);
}

void ClusterEngine::set_candle_duration(uint64_t duration_us) { candle_duration_us_ = duration_us; }

// ============================================================================
// Trade Ingestion
// ============================================================================

void ClusterEngine::ingest_trade(const TradeData& trade) {
  if (trade.symbol_id >= MAX_SYMBOLS) return;
  auto& state = symbols_[trade.symbol_id];
  std::lock_guard<std::mutex> lock(state.mu);

  // Auto-detect tick size from first trade
  if (!state.tick_detected && trade.price > 0.0) {
    state.tick_size = detect_tick_size(trade.price);
    state.tick_detected = true;
  }
  if (state.tick_size <= 0.0) return;

  double vol = static_cast<double>(trade.volume);
  bool is_buy = (trade.side == TradeSide::BUY);

  // ---- Update session volume profile ----
  int64_t tick_idx = price_to_tick(trade.price, state.tick_size);
  auto& profile_entry = state.profile_map[tick_idx];
  profile_entry.price = tick_to_price(tick_idx, state.tick_size);
  if (is_buy) {
    profile_entry.buy_vol += vol;
    profile_entry.buy_count++;
  } else {
    profile_entry.sell_vol += vol;
    profile_entry.sell_count++;
  }

  // ---- Update CVD ----
  state.cvd += is_buy ? vol : -vol;

  // ---- Update footprint column ----
  // Determine which candle time bucket this trade belongs to
  uint64_t candle_start = (trade.timestamp_us / candle_duration_us_) * candle_duration_us_;

  // Start a new column if needed
  if (!state.has_current || candle_start != state.current_column.open_time_us) {
    // Finalize previous column
    if (state.has_current) {
      finalize_column(state);
    }

    // Start new column
    state.current_column = FootprintColumn{};
    state.current_column.open_time_us = candle_start;
    state.current_column.close_time_us = candle_start + candle_duration_us_;
    state.current_column.open = trade.price;
    state.current_column.high = trade.price;
    state.current_column.low = trade.price;
    state.has_current = true;
  }

  auto& col = state.current_column;
  col.close = trade.price;
  col.high = std::max(col.high, trade.price);
  col.low = std::min(col.low, trade.price);
  col.delta += is_buy ? vol : -vol;

  // Find or create tick cluster in current column
  double tick_price = tick_to_price(tick_idx, state.tick_size);
  auto it = std::find_if(col.ticks.begin(), col.ticks.end(), [tick_price](const TickCluster& t) {
    return std::abs(t.price - tick_price) < 1e-12;
  });
  if (it != col.ticks.end()) {
    if (is_buy) {
      it->buy_vol += vol;
      it->buy_count++;
    } else {
      it->sell_vol += vol;
      it->sell_count++;
    }
  } else {
    TickCluster tc{};
    tc.price = tick_price;
    if (is_buy) {
      tc.buy_vol = vol;
      tc.buy_count = 1;
    } else {
      tc.sell_vol = vol;
      tc.sell_count = 1;
    }
    col.ticks.push_back(tc);
  }
}

void ClusterEngine::finalize_column(SymbolState& state) {
  // Sort ticks by price
  auto& col = state.current_column;
  std::sort(col.ticks.begin(), col.ticks.end(),
            [](const TickCluster& a, const TickCluster& b) { return a.price < b.price; });

  // Add to columns, maintain max size
  state.columns.push_back(std::move(col));
  if (state.columns.size() > MAX_COLUMNS) {
    state.columns.erase(state.columns.begin());
  }
}

// ============================================================================
// Read API
// ============================================================================

std::vector<FootprintColumn> ClusterEngine::get_footprint_columns(uint32_t symbol_id,
                                                                  size_t max_columns) const {
  if (symbol_id >= MAX_SYMBOLS) return {};
  const auto& state = symbols_[symbol_id];
  std::lock_guard<std::mutex> lock(state.mu);

  size_t n = std::min(state.columns.size(), max_columns);
  std::vector<FootprintColumn> result;
  result.reserve(n + 1);

  // Return last N finalized columns
  if (n > 0) {
    result.assign(state.columns.end() - n, state.columns.end());
  }

  // Also include the current in-progress column
  if (state.has_current && state.current_column.ticks.size() > 0) {
    FootprintColumn current_copy = state.current_column;
    std::sort(current_copy.ticks.begin(), current_copy.ticks.end(),
              [](const TickCluster& a, const TickCluster& b) { return a.price < b.price; });
    result.push_back(std::move(current_copy));
  }

  return result;
}

VolumeProfile ClusterEngine::get_volume_profile(uint32_t symbol_id) const {
  if (symbol_id >= MAX_SYMBOLS) return {};
  const auto& state = symbols_[symbol_id];
  std::lock_guard<std::mutex> lock(state.mu);

  VolumeProfile profile;
  profile.levels.reserve(state.profile_map.size());

  for (const auto& [tick_idx, cluster] : state.profile_map) {
    profile.levels.push_back(cluster);
  }

  // Sort by price
  std::sort(profile.levels.begin(), profile.levels.end(),
            [](const TickCluster& a, const TickCluster& b) { return a.price < b.price; });

  // Find POC (max total volume)
  profile.max_volume = 0.0;
  for (const auto& lvl : profile.levels) {
    double vol = lvl.total_vol();
    if (vol > profile.max_volume) {
      profile.max_volume = vol;
      profile.poc_price = lvl.price;
    }
  }

  // Compute Value Area (70%)
  compute_value_area(profile);

  return profile;
}

void ClusterEngine::compute_value_area(VolumeProfile& profile) const {
  if (profile.levels.empty()) return;

  double total_vol = 0.0;
  for (const auto& lvl : profile.levels) total_vol += lvl.total_vol();
  if (total_vol <= 0.0) return;

  double target = total_vol * 0.70;

  // Start from POC and expand outward
  int poc_idx = 0;
  for (int i = 0; i < static_cast<int>(profile.levels.size()); ++i) {
    if (std::abs(profile.levels[i].price - profile.poc_price) < 1e-12) {
      poc_idx = i;
      break;
    }
  }

  double accumulated = profile.levels[poc_idx].total_vol();
  int lo = poc_idx, hi = poc_idx;

  while (accumulated < target && (lo > 0 || hi < static_cast<int>(profile.levels.size()) - 1)) {
    double vol_below = (lo > 0) ? profile.levels[lo - 1].total_vol() : 0.0;
    double vol_above = (hi < static_cast<int>(profile.levels.size()) - 1)
                           ? profile.levels[hi + 1].total_vol()
                           : 0.0;

    if (vol_below >= vol_above && lo > 0) {
      lo--;
      accumulated += profile.levels[lo].total_vol();
    } else if (hi < static_cast<int>(profile.levels.size()) - 1) {
      hi++;
      accumulated += profile.levels[hi].total_vol();
    } else if (lo > 0) {
      lo--;
      accumulated += profile.levels[lo].total_vol();
    } else {
      break;
    }
  }

  profile.val = profile.levels[lo].price;
  profile.vah = profile.levels[hi].price;
}

double ClusterEngine::get_cvd(uint32_t symbol_id) const {
  if (symbol_id >= MAX_SYMBOLS) return 0.0;
  const auto& state = symbols_[symbol_id];
  std::lock_guard<std::mutex> lock(state.mu);
  return state.cvd;
}

}  // namespace BTQuant
