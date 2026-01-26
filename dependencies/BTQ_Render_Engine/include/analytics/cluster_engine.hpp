#pragma once

#include "../hotspine_layout_v3.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint> // Fix int64_t
#include <mutex> // Only for internal engine state if accessed by multiple writer threads (though architecture implies single writer pinned)
#include <unordered_map>
#include <vector>

// ============================================================================
// AGENT 2: QUANT ENGINEER
// MISSION: O(1) Analytics Engine & Viewport Rasterizer
// ============================================================================

namespace HotSpine::Analytics {

using namespace HotSpine::V3;

// Helper for fixed-point price keys to avoid float issues in map keys
// Assuming 8 decimal places is sufficient for crypto/fi
constexpr double PRICE_SCALER = 100000000.0;
inline int64_t price_to_key(double price) {
  return static_cast<int64_t>(std::round(price * PRICE_SCALER));
}

class ClusterEngine {
public:
  ClusterEngine() {
    // Pre-allocate to prevent early rehashes
    canvas_.reserve(100000);
  }

  // --------------------------------------------------------------------
  // "Infinite Canvas" Ingestion (O(1))
  // --------------------------------------------------------------------
  void process_trade(double price, double size, bool is_buyer_maker) {
    // Standardizing side: if buyer_maker (aggressor is seller), it's Sell
    // volume. If !buyer_maker (aggressor is buyer), it's Buy volume. But
    // usually "side" in data feeds: "buy" or "sell" refers to aggressor. Let's
    // assume standard viewer semantics: Delta = BuyVol - SellVol

    int64_t key = price_to_key(price);

    // O(1) Lookup & Insert
    VolumeNode &node = canvas_[key];

    // Accumulate
    // We store total volume at this price.
    // Note: VolumeNode in layout has {price, volume}.
    // In a real footprint, we'd want bid_vol/ask_vol split.
    // But directives say "VolumeNode (16 bytes, packed)" -> likely just
    // Price/Vol pair? Agent 1 defined VolumeNode as {price, volume}. So we just
    // aggregate total volume? The ClusterColumn has "delta". If the Node
    // doesn't split bid/ask, we can't recalculate delta perfectly from just the
    // node unless we store it differently in RAM. DECISION: To comply with
    // Agent 1's struct, we accumulate total volume in the Node, but we might
    // need extended state in RAM to track delta if required. Directives:
    // "Database (RAM): A dynamic ClusterEngine ... Viewport (SHM): ... Reader
    // never calculates". If Viewport needs Delta, we calculate it here. But
    // VolumeNode only has `volume`. Wait, ClusterColumn has `double delta`.
    // That's global for the column? "Delta: Buy Vol - Sell Vol" usually applies
    // to the bar/column total. AND per-row? Agent 1's VolumeNode is just
    // price/vol. Okay, we will accumulate global column delta in
    // `current_delta_` and row volume in `canvas_`.

    node.price = price; // Ensure price is set
    node.volume += size;

    if (is_buyer_maker) { // Sell aggression
      current_delta_ -= size;
    } else { // Buy aggression
      current_delta_ += size;
    }

    current_volume_ += size;

    // Track H/L for auto-centering
    if (price > session_high_)
      session_high_ = price;
    if (price < session_low_)
      session_low_ = price;
  }

  // --------------------------------------------------------------------
  // Rasterization: RAM -> SHM Viewport (O(N) relative to Viewport Size)
  // --------------------------------------------------------------------
  void snapshot_to_viewport(ClusterColumn &viewport, double center_price,
                            double tick_size) {
    // Reset Viewport
    // timestamp_us should be set by caller
    viewport.total_volume = current_volume_;
    viewport.delta = current_delta_;
    viewport.symbol_id = 0; // Caller sets

    // We want to render VIEWPORT_ROWS around center_price.
    // Each row represents a bucket of `tick_size`.

    int half_rows = VIEWPORT_ROWS / 2;
    double top_price = center_price + (half_rows * tick_size);
    double bottom_price = center_price - (half_rows * tick_size);

    viewport.high_price = top_price;
    viewport.low_price = bottom_price;
    viewport.active_rows = 0;

    // Clear rows first? Or strict overwrite?
    // Array is POD, best to clear or strictly overwrite.
    // We will iterate and fill.

    // OPTIMIZATION: Instead of iterating the Infinite Canvas (Huge),
    // We iterate the VIEWPORT TARGET (Small) and probe the Canvas.
    // 256 Lookups * O(1) = Fast.

    size_t row_idx = 0;
    // Render from Top to Bottom (High Price to Low Price)
    for (int i = 0; i < VIEWPORT_ROWS; ++i) {
      double p = top_price - (i * tick_size);
      int64_t key = price_to_key(p); // Quantize to nearest tick key

      auto it = canvas_.find(key);
      if (it != canvas_.end()) {
        viewport.rows[i] = it->second; // Copy cached VolumeNode
        viewport.active_rows++;
      } else {
        viewport.rows[i] = {p, 0.0}; // Empty level
      }
    }
  }

  void reset_session() {
    canvas_.clear();
    current_volume_ = 0;
    current_delta_ = 0;
    session_high_ = -1.0; // flag
    session_low_ = 1e9;
  }

private:
  // "Infinite Canvas"
  // Key: Price * SCALER (int64)
  // Value: VolumeNode (Aggregated)
  std::unordered_map<int64_t, VolumeNode> canvas_;

  double current_volume_ = 0.0;
  double current_delta_ = 0.0;
  double session_high_ = -1.0;
  double session_low_ = 1e9;
};

} // namespace HotSpine::Analytics
