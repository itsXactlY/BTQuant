#include "MarketDataBridge.hpp"
#include <algorithm>
#include <cmath>

namespace BTQuant {

MarketDataBridge::MarketDataBridge(HotSpine::HotSpineReader *reader)
    : reader_(reader) {
  // Initialize with some reasonable defaults to avoid div by zero
  current_view_ = {0, 1.0, 0, 100000.0};

  // Pre-allocate some space
  raw_candles_.reserve(10000);
  candles_.reserve(10000);
}

MarketDataBridge::~MarketDataBridge() {}

void MarketDataBridge::Update() {
  if (!reader_)
    return;

  HotSpine::HotTrade trade;
  bool new_data = false;

  // Poll all available trades
  while (reader_->pollTrade(trade)) {
    // Simple candle aggregation logic
    if (!temp_candle_.active) {
      temp_candle_.active = true;
      temp_candle_.start_time = trade.ts_exchange;
      temp_candle_.open = trade.price;
      temp_candle_.high = trade.price;
      temp_candle_.low = trade.price;
      temp_candle_.close = trade.price;
      temp_candle_.volume = trade.size;
    } else {
      // Check if we entered a new interval (1 second)
      if (trade.ts_exchange >= temp_candle_.start_time + CANDLE_INTERVAL_US) {
        // Finalize old candle
        RawCandle rc;
        rc.start_time = temp_candle_.start_time;
        rc.open = temp_candle_.open;
        rc.high = temp_candle_.high;
        rc.low = temp_candle_.low;
        rc.close = temp_candle_.close;

        {
          std::lock_guard<std::mutex> lock(data_mutex_);
          raw_candles_.push_back(rc);
        }
        new_data = true;

        // Start new candle
        temp_candle_.start_time = trade.ts_exchange; // Simplified alignment
        temp_candle_.open = trade.price;
        temp_candle_.high = trade.price;
        temp_candle_.low = trade.price;
        temp_candle_.close = trade.price;
        temp_candle_.volume = trade.size;
      } else {
        temp_candle_.high = std::max(temp_candle_.high, trade.price);
        temp_candle_.low = std::min(temp_candle_.low, trade.price);
        temp_candle_.close = trade.price;
        temp_candle_.volume += trade.size;
      }
    }
  }

  // Poll Orderbook
  HotSpine::HotOrderbookSnapshot snap;
  while (reader_->pollOrderbook(snap)) {
    // Just keep the latest
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_snapshot_ = snap;
  }

  if (new_data) {
    RebuildNDC();
  }
}

const std::vector<CandleData> &MarketDataBridge::GetRenderData() const {
  return candles_;
}

void MarketDataBridge::UpdateViewRect(const ViewRect &rect) {
  current_view_ = rect;
  RebuildNDC();
}

void MarketDataBridge::RebuildNDC() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  candles_.clear();
  candles_.reserve(raw_candles_.size());

  // Filter visible candles first? Or just map all?
  // Optimization: Only map visible candles.
  // However, the pipeline might expect a contiguous buffer.
  // We'll map all for now, or just map visible ones if the GPU buffer is
  // rebuilt every frame. The prompt says "Upload this vector ...
  // UpdateGPUBuffer". If we only upload visible ones, we save bandwidth.

  for (const auto &raw : raw_candles_) {
    // Optional: Simple Frustum Culling based on time
    if (raw.start_time < current_view_.min_time ||
        raw.start_time > current_view_.max_time) {
      // continue; // Uncomment to optimize
    }

    CandleData c;
    // X is time in NDC
    c.x = MapToScreenX((double)raw.start_time);

    // Y values in NDC
    c.open = MapToScreenY(raw.open);
    c.high = MapToScreenY(raw.high);
    c.low = MapToScreenY(raw.low);
    c.close = MapToScreenY(raw.close);

    // Color: Green if close >= open, Red otherwise.
    // Packed color format AABBGGRR (little endian) or RRGGBBAA?
    // Vulkan usually wants whatever the shader expects.
    // Let's assume standard 0xAABBGGRR.
    // Green: 0xFF00FF00, Red: 0xFF0000FF.
    if (raw.close >= raw.open) {
      c.color = 0xFF00FF00; // Green (ARGB) - Opaque Green
    } else {
      c.color = 0xFFFF0000; // Red (ARGB) - Opaque Red
    }

    candles_.push_back(c);
  }
}

void MarketDataBridge::DrawDOM(ImDrawList *drawList, ImVec2 pos, ImVec2 size) {
  if (!drawList)
    return;

  std::lock_guard<std::mutex> lock(data_mutex_);

  const float mid_x = pos.x + size.x * 0.5f;
  const float row_height = 20.0f;
  const int max_rows = static_cast<int>(size.y / row_height);

  // Background
  drawList->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                          IM_COL32(20, 20, 20, 200));

  // Draw Asks (Top, Red) - descending from Best Ask up?
  // Usually DOM shows Best Ask adjacent to Best Bid.
  // Center of list is Spread.
  // Asks go UP from center. Bids go DOWN from center.

  float center_y = pos.y + size.y * 0.5f;

  // Find max size for simple bar scaling
  double max_vol = 1.0;
  for (int i = 0; i < current_snapshot_.asks_count; ++i)
    max_vol = std::max(max_vol, current_snapshot_.asks[i].size);
  for (int i = 0; i < current_snapshot_.bids_count; ++i)
    max_vol = std::max(max_vol, current_snapshot_.bids[i].size);

  // Draw Asks (going up from center)
  for (int i = 0; i < current_snapshot_.asks_count && i < max_rows / 2; ++i) {
    float y = center_y - (i + 1) * row_height;
    if (y < pos.y)
      break;

    const auto &level = current_snapshot_.asks[i];

    // Bar
    float bar_width = (float)(level.size / max_vol) * (size.x * 0.5f);
    drawList->AddRectFilled(ImVec2(mid_x, y),
                            ImVec2(mid_x + bar_width, y + row_height),
                            IM_COL32(100, 0, 0, 150));

    // Text
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.2f", level.price);
    drawList->AddText(ImVec2(mid_x + 5, y), IM_COL32(255, 100, 100, 255),
                      buffer);

    snprintf(buffer, sizeof(buffer), "%.4f", level.size);
    ImVec2 txt_sz = ImGui::CalcTextSize(buffer);
    drawList->AddText(ImVec2(mid_x + size.x * 0.5f - txt_sz.x - 5, y),
                      IM_COL32(200, 200, 200, 255), buffer);
  }

  // Draw Bids (going down from center)
  for (int i = 0; i < current_snapshot_.bids_count && i < max_rows / 2; ++i) {
    float y = center_y + i * row_height;
    if (y + row_height > pos.y + size.y)
      break;

    const auto &level = current_snapshot_.bids[i];

    // Bar (on left side?) Or same side? Usually DOM is split or stacked.
    // Let's mirror bars. Bids on left?
    // Logic: "Draws horizontal depth bars (rects) behind the text."
    // Task doesn't specify layout. Standard is Center Price, Bars outward.

    // Bar going left from center
    float bar_width = (float)(level.size / max_vol) * (size.x * 0.5f);
    drawList->AddRectFilled(ImVec2(mid_x - bar_width, y),
                            ImVec2(mid_x, y + row_height),
                            IM_COL32(0, 100, 0, 150));

    // Text
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.2f", level.price);
    ImVec2 txt_sz = ImGui::CalcTextSize(buffer);
    drawList->AddText(ImVec2(mid_x - txt_sz.x - 5, y),
                      IM_COL32(100, 255, 100, 255), buffer);

    snprintf(buffer, sizeof(buffer), "%.4f", level.size);
    drawList->AddText(ImVec2(pos.x + 5, y), IM_COL32(200, 200, 200, 255),
                      buffer);
  }

  // Spread line
  drawList->AddLine(ImVec2(pos.x, center_y), ImVec2(pos.x + size.x, center_y),
                    IM_COL32(150, 150, 150, 100));
}

float MarketDataBridge::MapToScreenX(double time) {
  if (std::abs(current_view_.max_time - current_view_.min_time) < 0.0001)
    return 0.0f;
  return -1.0f +
         2.0f * (float)((time - current_view_.min_time) /
                        (current_view_.max_time - current_view_.min_time));
}

float MarketDataBridge::MapToScreenY(double price) {
  if (std::abs(current_view_.max_price - current_view_.min_price) < 0.00001)
    return 0.0f;
  // Map max_price -> -1.0 (top)
  // Map min_price -> 1.0 (bottom)
  // Formula: -1.0 + 2.0 * (max - p) / (max - min) ?
  // Check: p=max -> -1 + 0 = -1. Correct.
  // Check: p=min -> -1 + 2 = 1. Correct.
  return -1.0f +
         2.0f * (float)((current_view_.max_price - price) /
                        (current_view_.max_price - current_view_.min_price));
}

} // namespace BTQuant
