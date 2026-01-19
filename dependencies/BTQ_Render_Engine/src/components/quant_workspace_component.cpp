#include "../../include/components/quant_workspace_component.hpp"
#include "../../include/components/theme_customization_component.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge)
    : UIComponent({0, 0}, {0, 0}), bridge_(bridge) {
  market_data_processor_ = std::make_unique<RenderEngine::MarketDataProcessor>();
  theme_customization_ = std::make_unique<ThemeCustomizationComponent>(config_);
}

void QuantWorkspaceComponent::update(float dt) {
  theme_customization_->update(dt);
}

void QuantWorkspaceComponent::render_gui() {
  // Render theme and layout management
  theme_customization_->render_gui();

  // Render workspace content
  auto instruments = bridge_->GetInstruments();

  ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetFrameHeight()), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y - ImGui::GetFrameHeight()), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Workspace", nullptr,
                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar)) {

    render_timeframe_selector();

    // Design: Use an ImGui Table (e.g., 3 columns for 3 exchanges) to organize
    // charts automatically.
    if (ImGui::BeginTable("WorkspaceGrid", 3,
                          ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
                              ImGuiTableFlags_SizingStretchSame)) {
      for (auto const &[id, instrument] : instruments) {
        ImGui::TableNextColumn();
        render_instrument_chart(id, *instrument);
      }
      ImGui::EndTable();
    }
    ImGui::End();
  }
}

void QuantWorkspaceComponent::render_timeframe_selector() {
  ImGui::Text("Timeframe: ");
  ImGui::SameLine();
  
  if (ImGui::RadioButton("1m", current_timeframe_ == RenderEngine::TimeFrame::TF_1MIN)) {
    current_timeframe_ = RenderEngine::TimeFrame::TF_1MIN;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("5m", current_timeframe_ == RenderEngine::TimeFrame::TF_5MIN)) {
    current_timeframe_ = RenderEngine::TimeFrame::TF_5MIN;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("15m", current_timeframe_ == RenderEngine::TimeFrame::TF_15MIN)) {
    current_timeframe_ = RenderEngine::TimeFrame::TF_15MIN;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("1h", current_timeframe_ == RenderEngine::TimeFrame::TF_1HOUR)) {
    current_timeframe_ = RenderEngine::TimeFrame::TF_1HOUR;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("4h", current_timeframe_ == RenderEngine::TimeFrame::TF_4HOUR)) {
    current_timeframe_ = RenderEngine::TimeFrame::TF_4HOUR;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("1d", current_timeframe_ == RenderEngine::TimeFrame::TF_1DAY)) {
    current_timeframe_ = RenderEngine::TimeFrame::TF_1DAY;
  }
}

void QuantWorkspaceComponent::render_instrument_chart(
    const std::string &id, const MarketInstrument &instrument) {
  std::lock_guard<std::mutex> lock(instrument.data_mutex);

  ImGui::PushID(id.c_str());
  ImGui::Text("%s - %s", instrument.symbol.c_str(),
              instrument.exchange.c_str());

  // Get OHLCV data from MarketDataProcessor
  auto candles = market_data_processor_->getCandles(instrument.symbol_id, current_timeframe_);
  
  if (ImPlot::BeginPlot("##Candles", ImVec2(-1, 300))) {
    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxis(ImAxis_Y1, nullptr,
                      ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_Opposite);

    if (!candles.empty()) {
      static std::vector<double> t, o, h, l, c;
      t.resize(candles.size());
      o.resize(candles.size());
      h.resize(candles.size());
      l.resize(candles.size());
      c.resize(candles.size());

      for (size_t i = 0; i < candles.size(); ++i) {
        const auto& candle = candles[i];
        t[i] = static_cast<double>(candle.timestamp) / 1000000.0; // Convert to seconds since epoch
        o[i] = candle.open;
        h[i] = candle.high;
        l[i] = candle.low;
        c[i] = candle.close;
      }

      // Plot Candles
      auto PlotCandlesticksManual = [&](int cnt) {
        ImDrawList *draw_list = ImPlot::GetPlotDrawList();
        for (int i = 0; i < cnt; ++i) {
          ImVec4 color = (c[i] >= o[i]) ? ImVec4(0, 0.8f, 0.2f, 1)
                                        : ImVec4(0.9f, 0.1f, 0.1f, 1);
          ImU32 col32 = ImGui::ColorConvertFloat4ToU32(color);

          ImVec2 uv_min = ImPlot::PlotToPixels(t[i], l[i]);
          ImVec2 uv_max = ImPlot::PlotToPixels(t[i], h[i]);
          draw_list->AddLine(uv_min, uv_max, col32);

          float width = 4.0f;
          ImVec2 p_open = ImPlot::PlotToPixels(t[i], o[i]);
          ImVec2 p_close = ImPlot::PlotToPixels(t[i], c[i]);
          
          float candle_width = 4.0f;
          if (c[i] >= o[i]) {
            // Bullish candle (green) - filled
            draw_list->AddRectFilled(
                ImVec2(p_open.x - candle_width / 2, p_open.y),
                ImVec2(p_close.x + candle_width / 2, p_close.y),
                col32);
          } else {
            // Bearish candle (red) - hollow
            draw_list->AddRect(
                ImVec2(p_open.x - candle_width / 2, p_open.y),
                ImVec2(p_close.x + candle_width / 2, p_close.y),
                col32);
          }
        }
      };

      PlotCandlesticksManual(static_cast<int>(candles.size()));

      // Calculate and plot moving averages
      if (candles.size() >= 10) {
        std::vector<double> sma_10(candles.size());
        for (size_t i = 9; i < candles.size(); ++i) {
          double sum = 0;
          for (size_t j = i - 9; j <= i; ++j) {
            sum += candles[j].close;
          }
          sma_10[i] = sum / 10;
        }
        ImPlot::PlotLine("SMA 10", t.data(), sma_10.data(), static_cast<int>(candles.size()));
      }

      if (candles.size() >= 20) {
        std::vector<double> sma_20(candles.size());
        for (size_t i = 19; i < candles.size(); ++i) {
          double sum = 0;
          for (size_t j = i - 19; j <= i; ++j) {
            sum += candles[j].close;
          }
          sma_20[i] = sum / 20;
        }
        ImPlot::PlotLine("SMA 20", t.data(), sma_20.data(), static_cast<int>(candles.size()));
      }

      if (candles.size() >= 50) {
        std::vector<double> sma_50(candles.size());
        for (size_t i = 49; i < candles.size(); ++i) {
          double sum = 0;
          for (size_t j = i - 49; j <= i; ++j) {
            sum += candles[j].close;
          }
          sma_50[i] = sum / 50;
        }
        ImPlot::PlotLine("SMA 50", t.data(), sma_50.data(), static_cast<int>(candles.size()));
      }
    }

    ImPlot::EndPlot();
  }

  // Add Statistical Insight and Order Book Mini-view
  ImGui::Columns(2, "Details", false);
  render_stats_panel(instrument);
  ImGui::NextColumn();
  render_order_book_mini(instrument);
  ImGui::Columns(1);

  ImGui::PopID();
}

void QuantWorkspaceComponent::render_order_book_mini(
    const MarketInstrument &instrument) {
  // Mini Order Book View
  ImGui::TextDisabled("LOB Snapshot");
  if (ImGui::BeginTable("MiniLOB", 2, ImGuiTableFlags_SizingFixedFit)) {
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 60.0f);

    const auto &snap = instrument.latest_snapshot;

    // Asks (Red) - Show best 3 (last 3 of array? No, usually 0 is best.
    // Assuming 0 is best bid/ask in snapshot arrays)
    for (int i = 2; i >= 0; --i) {
      if (i >= (int)snap.asks.size())
        continue;

      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%.2f", snap.asks[i].price);
      ImGui::TableNextColumn();
      ImGui::Text("%.3f", snap.asks[i].size);
    }

    // Mid
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    // Safe spread calc
    double best_ask = snap.asks[0].price;
    double best_bid = snap.bids[0].price;
    double mid = (best_ask + best_bid) * 0.5;

    if (mid == 0.0)
      mid = instrument.closes.empty()
                ? 0.0
                : instrument.closes[instrument.write_idx > 0 ? instrument.write_idx - 1
                                                              : 0]; // Fallback

    ImGui::TextColored(ImVec4(1, 1, 1, 0.5f), "--- %.2f ---", mid);
    ImGui::TableNextColumn();

    // Bids (Green)
    for (int i = 0; i < 3; ++i) {
      if (i >= (int)snap.bids.size())
        continue;

      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "%.2f", snap.bids[i].price);
      ImGui::TableNextColumn();
      ImGui::Text("%.3f", snap.bids[i].size);
    }
    ImGui::EndTable();
  }
}

void QuantWorkspaceComponent::render_stats_panel(
    const MarketInstrument &instrument) {
  if (instrument.size > 0) {
    size_t last_idx = (instrument.write_idx == 0)
                          ? (MarketInstrument::HISTORY_CAPACITY - 1)
                          : (instrument.write_idx - 1);
    size_t prev_idx = (last_idx == 0) ? (MarketInstrument::HISTORY_CAPACITY - 1)
                                      : (last_idx - 1);

    double last = instrument.closes[last_idx];
    double prev = (instrument.size > 1) ? instrument.closes[prev_idx] : last;

    double change = last - prev;
    double pct = (prev != 0) ? (change / prev) * 100.0 : 0.0;

    ImGui::Text("Last: %.2f", last);
    ImGui::TextColored(change >= 0 ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
                       "Chg: %+.2f (%+.2f%%)", change, pct);
    ImGui::Text("Vol: %.1f", instrument.volumes[last_idx]);
  }
}

} // namespace BTQuant
