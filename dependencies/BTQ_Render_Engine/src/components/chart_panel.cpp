#include "../../include/components/chart_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace BTQuant {

static std::string timeframe_to_string(RenderEngine::TimeFrame tf) {
  switch (tf) {
  case RenderEngine::TimeFrame::TF_1MS:
    return "1ms";
  case RenderEngine::TimeFrame::TF_10MS:
    return "10ms";
  case RenderEngine::TimeFrame::TF_100MS:
    return "100ms";
  case RenderEngine::TimeFrame::TF_500MS:
    return "500ms";
  case RenderEngine::TimeFrame::TF_1SEC:
    return "1s";
  case RenderEngine::TimeFrame::TF_3SEC:
    return "3s";
  case RenderEngine::TimeFrame::TF_5SEC:
    return "5s";
  case RenderEngine::TimeFrame::TF_15SEC:
    return "15s";
  default:
    return "Unknown";
  }
}

ChartPanel::ChartPanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    ChartManager *chart_manager)
    : PanelBase(config), bridge_(bridge), processor_(processor),
      chart_manager_(chart_manager) {
  indicator_renderer_ = new IndicatorRenderer(nullptr, processor_);
}

void ChartPanel::initialize() {
  // Create the chart
  auto id_opt = chart_manager_->getSymbolId(symbol_);
  uint32_t symbol_id = id_opt ? *id_opt : 10007; // Default BTC-USDT
  chart_id_ =
      chart_manager_->create_chart(symbol_, exchange_, symbol_id, timeframe_);
}

void ChartPanel::update(float dt) {
  // Chart manager handles updates
  (void)dt;
}

void ChartPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  // Get the chart instance
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it == charts.end()) {
    ImGui::Text("Chart not found");
    end_panel_window();
    return;
  }

  const ChartInstance &chart = it->second;

  // Render chart controls in a collapsible header
  if (ImGui::CollapsingHeader("Chart Controls",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    render_chart_controls();
  }

  // Render the chart (no indicators)
  render_instrument_chart(chart);

  end_panel_window();
}

void ChartPanel::set_symbol(const std::string &symbol,
                            const std::string &exchange) {
  symbol_ = symbol;
  exchange_ = exchange;
  config_.title = symbol_ + " Chart [" + timeframe_to_string(timeframe_) + "]";

  // Recreate chart with new symbol
  if (chart_id_ != 0) {
    chart_manager_->destroy_chart(chart_id_);
  }
  initialize();
}

void ChartPanel::set_timeframe(RenderEngine::TimeFrame timeframe) {
  timeframe_ = timeframe;
  config_.title = symbol_ + " Chart [" + timeframe_to_string(timeframe_) + "]";

  // Recreate chart with new timeframe
  if (chart_id_ != 0) {
    chart_manager_->destroy_chart(chart_id_);
  }
  initialize();
}

void ChartPanel::render_chart_controls() {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  // Symbol input
  static char symbol_input[32] = "BTC-USDT";
  std::strncpy(symbol_input, symbol_.c_str(), sizeof(symbol_input) - 1);
  if (ImGui::InputText("Symbol", symbol_input, sizeof(symbol_input))) {
    set_symbol(symbol_input, exchange_);
  }
  ImGui::SameLine();

  // Timeframe selector (1ms-15sec only)
  const char *timeframes[] = {"1ms", "10ms", "100ms", "500ms",
                              "1s",  "3s",   "5s",    "15s"};
  int selected = static_cast<int>(timeframe_);
  if (ImGui::Combo("Timeframe", &selected, timeframes,
                   IM_ARRAYSIZE(timeframes))) {
    set_timeframe(static_cast<RenderEngine::TimeFrame>(selected));
  }

  // Auto-follow Window Size
  ImGui::SameLine();
  int window_size_int = static_cast<int>(auto_follow_window_);
  if (ImGui::SliderInt("Window", &window_size_int, 100, 10000, "%d")) {
    auto_follow_window_ = static_cast<float>(window_size_int);
  }

  // Auto-follow checkbox
  ImGui::Checkbox("Auto-follow", &follow_latest_);

  ImGui::PopStyleVar();
}

void ChartPanel::render_indicator_selector() {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  ImGui::Checkbox("SMA 10", &indicator_config_.show_sma_10);
  ImGui::SameLine();
  ImGui::Checkbox("SMA 20", &indicator_config_.show_sma_20);
  ImGui::SameLine();
  ImGui::Checkbox("SMA 50", &indicator_config_.show_sma_50);

  ImGui::Checkbox("EMA 10", &indicator_config_.show_ema_10);
  ImGui::SameLine();
  ImGui::Checkbox("EMA 20", &indicator_config_.show_ema_20);
  ImGui::SameLine();
  ImGui::Checkbox("EMA 50", &indicator_config_.show_ema_50);

  ImGui::Checkbox("RSI", &indicator_config_.show_rsi);
  ImGui::SameLine();
  ImGui::Checkbox("MACD", &indicator_config_.show_macd);
  ImGui::SameLine();
  ImGui::Checkbox("Bollinger", &indicator_config_.show_bollinger);
  ImGui::SameLine();
  ImGui::Checkbox("Vol Profile", &indicator_config_.show_volume_profile);

  ImGui::PopStyleVar();
}

void ChartPanel::render_instrument_chart(const ChartInstance &chart) {
  if (chart.dates.empty()) {
    ImGui::Text("Loading chart data for %s...", symbol_.c_str());
    return;
  }

  // Diagnostic info & Controls
  ImGui::TextColored(
      ImVec4(0.0f, 1.0f, 0.8f, 1.0f),
      " | Candles: %zu | Last: %.2f | TF: %.4gs", chart.dates.size(),
      chart.closes.empty() ? 0.0f : chart.closes.back(),
      RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) /
          1000000.0);

  // Neon Chart Styling (ThemeManager)
  const auto &colors = ThemeManager::getInstance().getColors();
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, colors.background);
  ImPlot::PushStyleColor(ImPlotCol_PlotBg,
                         colors.panel_bg); // Use panel bg or specific dark
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, colors.border);
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(10, 10));

  if (ImPlot::BeginPlot("##Chart", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoTitle |
                            ImPlotFlags_Crosshairs)) {

    // Setup Axes
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_None);
    // SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);

    // Setup X2 for Volume Profile (Hidden, Inverted?)
    if (indicator_config_.show_volume_profile) {
      ImPlot::SetupAxis(ImAxis_X2, nullptr,
                        ImPlotAxisFlags_NoTickLabels |
                            ImPlotAxisFlags_NoGridLines);
      // We want bars on the right side.
      // If we set limits manually, e.g. 0 to MaxVolume.
      // And we probably want to invert it so 0 is on the right?
      // Or just map normal 0..Max.
      ImPlot::SetupAxisLimits(
          ImAxis_X2, 0, 1000,
          ImPlotCond_Always); // Placeholder, updated dynamically?
                              // Actually, let's auto fit X2.
    }

    // Auto-follow logic
    if (follow_latest_) {
      double time_max = chart.dates.back();
      // Use raw duration to support both Seconds and Microseconds timestamps
      double duration_raw =
          RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_);

      // Window = Configurable candles * duration per candle
      double window_size = duration_raw * auto_follow_window_;

      // Determine padding based on window size
      double padding = window_size * 0.05;

      // Update cached limits for consistent logic
      last_view_min_ = time_max - window_size;
      last_view_max_ = time_max + padding;

      ImPlot::SetupAxisLimits(ImAxis_X1, last_view_min_, last_view_max_,
                              ImPlotCond_Always);
    } else {
      ImPlot::SetupAxisLimits(ImAxis_X1, chart.dates.front(),
                              chart.dates.back(), ImPlotCond_Once);
    }

    // MANUAL Y-AXIS SCALING
    // We must invoke SetupAxisLimits BEFORE GetPlotLimits to satisfy ImGui
    // constraints. We use the limits from the *previous* frame (or
    // auto-calculated above).
    {
      double view_x_min = last_view_min_;
      double view_x_max = last_view_max_;

      if (view_x_min == 0 && view_x_max == 0 && !chart.dates.empty()) {
        view_x_min = chart.dates.front();
        view_x_max = chart.dates.back();
      }

      // Binary search for visible range based on our ESTIMATE/CACHE
      size_t start_idx = 0;
      size_t end_idx = chart.dates.size();

      auto lower =
          std::lower_bound(chart.dates.begin(), chart.dates.end(), view_x_min);
      if (lower != chart.dates.begin())
        --lower;
      start_idx = std::distance(chart.dates.begin(), lower);

      auto upper =
          std::upper_bound(chart.dates.begin(), chart.dates.end(), view_x_max);
      if (upper != chart.dates.end())
        ++upper;
      end_idx = std::distance(chart.dates.begin(), upper);

      end_idx = std::min(end_idx, chart.dates.size());

      if (start_idx < end_idx) {
        float y_min = std::numeric_limits<float>::max();
        float y_max = std::numeric_limits<float>::lowest();
        bool found_data = false;

        for (size_t i = start_idx; i < end_idx; ++i) {
          float low = chart.lows[i];
          float high = chart.highs[i];
          if (low > 0 && high > 0) { // Valid data
            if (low < y_min)
              y_min = low;
            if (high > y_max)
              y_max = high;
            found_data = true;
          }
        }

        if (found_data) {
          float range = y_max - y_min;
          if (range == 0)
            range = y_max * 0.01f;
          if (range == 0)
            range = 1.0f;

          y_min -= range * 0.1f;
          y_max += range * 0.1f;

          ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always);
        }
      }
    }

    // VOLUME PROFILE OVERLAY
    if (indicator_config_.show_volume_profile) {
      auto id_opt = chart_manager_->getSymbolId(symbol_);
      if (id_opt) {
        auto profile = processor_->getVolumeProfile(*id_opt, timeframe_);
        if (!profile.empty()) {
          std::vector<double> prices;
          std::vector<double> volumes;
          prices.reserve(profile.size());
          volumes.reserve(profile.size());

          double max_vol = 0;
          for (const auto &level : profile) {
            if (level.price >= last_view_min_ &&
                level.price <=
                    last_view_max_ // Check Y axis? No, Price is Y.
                                   // Wait, ImPlot axis logic: X is Time, Y is
                                   // Price. Volume Profile: Bars extend from
                                   // Right to Left or Left to Right? If
                                   // standard PlotBars Horizontal: xs = Volume
                                   // (Length), ys = Price (Position) But X axis
                                   // is TIME. We can't map Volume to Time
                                   // directly unless we use a secondary X axis.
            ) {
              // We need to map Volume to Time units or use a separate axis.
              // Easiest is to scale Volume to fit in the current Time window.
              // Or simply draw it on the right edge.
            }
            prices.push_back(level.price);
            volumes.push_back(level.total_volume);
            if (level.total_volume > max_vol)
              max_vol = level.total_volume;
          }

          if (max_vol > 0) {
            // Option 1: Scale volumes to time range (width of view)
            double time_width = last_view_max_ - last_view_min_;
            double scale =
                (time_width * 0.3) / max_vol; // Take up 30% of screen width

            std::vector<double> scaled_volumes;
            scaled_volumes.reserve(volumes.size());
            // Position bars at the right edge
            std::vector<double> bar_starts;
            bar_starts.reserve(volumes.size());

            for (double v : volumes) {
              // Bars growing from Right to Left?
              // ImPlotBars doesn't support "start from X".
              // We might need to use PlotRects or simply plot bars at (X_end -
              // vol) to X_end? Or just generic PlotBarsHorizontal at X_end -
              // vol/2 ?? ImPlot::PlotBars with Horizontal flag plots centered
              // at X? No, "positions" are Y coordinates. "values" are X
              // lengths. Bars start at 0? We need them to start at
              // `last_view_max_` and go left. Actually, maybe just plot them as
              // standard bars on X2 axis? Let's try simple PlotBars with
              // Horizontal flag. X = Volume, Y = Price. But X axis is Time. We
              // need to use ImPlot::SetAxes(ImAxis_X2, ImAxis_Y1); Setup X2 to
              // be Volume.
              scaled_volumes.push_back(v);
            }

            ImPlot::SetAxes(ImAxis_X2, ImAxis_Y1);
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
            ImPlot::PlotBars("VP", volumes.data(), prices.data(),
                             static_cast<int>(prices.size()), 0.0,
                             ImPlotBarsFlags_Horizontal);
            ImPlot::PopStyleVar();
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1); // Reset
          }
        }
      }
    }

    // Drag & Drop Target for Price Levels (Must be after ALL Setup calls)
    if (ImPlot::BeginDragDropTargetPlot()) {
      if (const ImGuiPayload *payload =
              ImGui::AcceptDragDropPayload("PRICE_LEVEL")) {
        double dropped_price = *(const double *)payload->Data;
        std::cout << "[ChartPanel] Dropped Price Level: " << dropped_price
                  << std::endl;
      }
      ImPlot::EndDragDropTarget();
    }

    // Now get the actual limits being used for THIS frame's rendering and NEXT
    // frame's scaling This locks setup, so it must happen AFTER SetupAxisLimits
    ImPlotRect limits = ImPlot::GetPlotLimits();

    // Store for next frame
    last_view_min_ = limits.X.Min;
    last_view_max_ = limits.X.Max;

    // Recalculate start/end for CULLING (Rendering optimization)
    // We can reuse the indices if the view hasn't drifted much, but better to
    // be precise for drawing
    size_t start_idx = 0;
    size_t end_idx = chart.dates.size();
    {
      auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(),
                                    limits.X.Min);
      if (lower != chart.dates.begin())
        --lower;
      start_idx = std::distance(chart.dates.begin(), lower);

      auto upper = std::upper_bound(chart.dates.begin(), chart.dates.end(),
                                    limits.X.Max);
      if (upper != chart.dates.end())
        ++upper;
      end_idx = std::min((size_t)std::distance(chart.dates.begin(), upper),
                         chart.dates.size());
    }

    // Calculate candle width based on timeframe
    double duration_sec =
        RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) /
        1000000.0;
    double candle_half_width = duration_sec * 0.4;

    // Safeguard for very small TFs
    if (candle_half_width < 0.000001)
      candle_half_width = 0.000001;

    ImDrawList *draw_list = ImPlot::GetPlotDrawList();

    // Calculate minimum pixel width for candles
    // We need at least 3 pixels for a visible candle body
    const float MIN_BODY_WIDTH_PX = 3.0f;
    const float MIN_BODY_HEIGHT_PX = 1.0f;

    // Draw ONLY visible candles
    for (size_t i = start_idx; i < end_idx; ++i) {
      double x = chart.dates[i];
      if (x == 0)
        continue;

      float open = chart.opens[i];
      float high = chart.highs[i];
      float low = chart.lows[i];
      float close = chart.closes[i];

      // Skip invalid candles
      if (high == 0 || low == 0 || open == 0 || close == 0)
        continue;

      bool bullish = close >= open;
      const auto &colors = ThemeManager::getInstance().getColors();
      ImU32 color = bullish
                        ? ImGui::ColorConvertFloat4ToU32(colors.candle_up)
                        : ImGui::ColorConvertFloat4ToU32(colors.candle_down);
      // Wicks same as body or distinct? Using same for neon look
      ImU32 wick_color = color;

      // Transform to screen coordinates
      ImVec2 wick_top = ImPlot::PlotToPixels(x, high);
      ImVec2 wick_bot = ImPlot::PlotToPixels(x, low);
      ImVec2 body_tl =
          ImPlot::PlotToPixels(x - candle_half_width, bullish ? close : open);
      ImVec2 body_br =
          ImPlot::PlotToPixels(x + candle_half_width, bullish ? open : close);

      // Ensure minimum body width in pixels
      float body_width = std::abs(body_br.x - body_tl.x);
      if (body_width < MIN_BODY_WIDTH_PX) {
        float extra = (MIN_BODY_WIDTH_PX - body_width) / 2.0f;
        body_tl.x -= extra;
        body_br.x += extra;
      }

      // Ensure minimum body height in pixels (for doji candles)
      float body_height = std::abs(body_br.y - body_tl.y);
      if (body_height < MIN_BODY_HEIGHT_PX) {
        float mid_y = (body_tl.y + body_br.y) / 2.0f;
        body_tl.y = mid_y - MIN_BODY_HEIGHT_PX / 2.0f;
        body_br.y = mid_y + MIN_BODY_HEIGHT_PX / 2.0f;
      }

      // Draw wick (vertical line)
      draw_list->AddLine(wick_top, wick_bot, wick_color, 1.0f);

      // Draw body (filled rectangle)
      draw_list->AddRectFilled(body_tl, body_br, color);
    }

    ImPlot::EndPlot();
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor(3);
}

} // namespace BTQuant