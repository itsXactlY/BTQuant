#pragma once

#include "../../../tests/new/include/hotspine_reader.hpp"
#include "CandlePipeline.h" // For CandleData
#include "imgui.h"
#include <mutex>
#include <vector>

namespace BTQuant {

struct ViewRect {
  double min_time = 0.0;
  double max_time = 0.0;
  double min_price = 0.0;
  double max_price = 0.0;
};

class MarketDataBridge {
public:
  MarketDataBridge(HotSpine::HotSpineReader *reader);
  ~MarketDataBridge();

  // Polls reader and updates internal cache
  void Update();

  // Sets the current view for NDC calculation
  void UpdateViewRect(const ViewRect &rect);

  // Returns data ready for GPU upload
  const std::vector<CandleData> &GetRenderData() const;

  // Drawns the Orderbook Depth of Market (DOM)
  void DrawDOM(ImDrawList *drawList, ImVec2 pos, ImVec2 size);

private:
  float MapToScreenX(double time);
  float MapToScreenY(double price);
  void RebuildNDC();

  HotSpine::HotSpineReader *reader_;

  struct RawCandle {
    double open = 0;
    double high = 0;
    double low = 0;
    double close = 0;
    uint64_t start_time = 0; // Microseconds
  };

  // storage for raw high-precision data
  std::vector<RawCandle> raw_candles_;

  // Local caches
  std::vector<CandleData> candles_;
  HotSpine::HotOrderbookSnapshot current_snapshot_;

  std::mutex data_mutex_;
  ViewRect current_view_;

  // For candle generation from trades (simplistic)
  struct CurrentCandle {
    double open = 0;
    double high = 0;
    double low = 0;
    double close = 0;
    double volume = 0;
    uint64_t start_time = 0;
    bool active = false;
  } temp_candle_;

  // Accumulate trades to build candles
  static constexpr uint64_t CANDLE_INTERVAL_US =
      1000000; // 1 second candles for demo
};

} // namespace BTQuant
