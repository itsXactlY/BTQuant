#include "../../include/components/tape_panel.hpp"

#include <algorithm>
#include <ctime>
#include <sstream>

#include "imgui.h"
#include "implot.h"
#include "../../include/components/theme_manager.hpp"

namespace BTQuant {

TapePanel::TapePanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
  cached_trades_.reserve(MAX_VISIBLE_TRADES);

  // Initialize trade pace history vectors
  trade_pace_1min_history_.reserve(100);  // Keep last 100 measurements
  trade_pace_5min_history_.reserve(100);
  trade_pace_15min_history_.reserve(100);

  // C++26: Subscribe to push notifications instead of polling
  subscribe_to_updates();
}

TapePanel::~TapePanel() {
  // C++26: Clean unsubscription on destruction
  if (processor_ && subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }
}

void TapePanel::subscribe_to_updates() {
  if (!processor_ || symbol_id_ == 0) return;

  // Unsubscribe from previous symbol if any
  if (subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }

  // Subscribe to TRADE notifications for this symbol
  subscription_id_ = processor_->subscribe(
      symbol_id_, RenderEngine::NotificationType::TRADE,
      [this](uint32_t /*symbol_id*/, RenderEngine::NotificationType /*type*/) {
        // Thread-safe: atomic flag set from worker thread
        this->markDirty();
      });
}

void TapePanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();
  render_controls();

  // Update and render trade pace chart in the header area
  updateTradePaceHistory();
  renderTradePaceChart();

  ImGui::Separator();

  // C++26 Reactive: Refresh data when new trades arrive or first load
  if (processor_ && symbol_id_ != 0) {
    if (consumeDirty() || cached_trades_.empty()) {
      auto analytics = processor_->getSymbolAnalytics(symbol_id_);
      cached_trades_ = analytics.recent_trades;

      // Keep only most recent trades for display
      if (cached_trades_.size() > MAX_VISIBLE_TRADES) {
        cached_trades_.erase(cached_trades_.begin(),
                             cached_trades_.begin() + (cached_trades_.size() - MAX_VISIBLE_TRADES));
      }
    }
  }

  render_trade_table();

  end_panel_window();
}

void TapePanel::set_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  cached_trades_.clear();

  // Re-subscribe to new symbol
  subscribe_to_updates();
  markDirty();  // Force immediate refresh
}

void TapePanel::render_filter_controls() {
  ImGui::Separator();
  ImGui::Text("Filter Options:");

  // Minimum Size Filter
  ImGui::AlignTextToFramePadding();
  ImGui::Text("Min Size:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100);
  ImGui::InputText("##min_size", min_size_input_, sizeof(min_size_input_));

  // Exchange Filter
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::Text("Exchange:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(120);
  ImGui::InputText("##exchange", exchange_input_, sizeof(exchange_input_));

  // Start Time Filter
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::Text("Start Time:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100);
  ImGui::InputText("##start_time", start_time_input_, sizeof(start_time_input_));

  // End Time Filter
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::Text("End Time:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100);
  ImGui::InputText("##end_time", end_time_input_, sizeof(end_time_input_));

  // Apply button
  ImGui::SameLine();
  if (ImGui::Button("Apply")) {
    // Parse minimum size
    try {
      min_size_filter_ = std::stod(std::string(min_size_input_));
    } catch (...) {
      min_size_filter_ = 0.0;
    }

    // Set exchange filter
    exchange_filter_ = std::string(exchange_input_);

    // Parse start time - supports both raw timestamp and HH:MM:SS format
    if (strlen(start_time_input_) > 0) {
      std::string time_str = std::string(start_time_input_);
      if (time_str.find(':') != std::string::npos) {
        // Parse HH:MM:SS format
        start_time_filter_ = parseTimeString(time_str);
      } else {
        // Parse as raw timestamp
        try {
          start_time_filter_ = std::stoull(time_str);
        } catch (...) {
          start_time_filter_ = 0;
        }
      }
    } else {
      start_time_filter_ = 0;
    }

    // Parse end time - supports both raw timestamp and HH:MM:SS format
    if (strlen(end_time_input_) > 0) {
      std::string time_str = std::string(end_time_input_);
      if (time_str.find(':') != std::string::npos) {
        // Parse HH:MM:SS format
        end_time_filter_ = parseTimeString(time_str);
      } else {
        // Parse as raw timestamp
        try {
          end_time_filter_ = std::stoull(time_str);
        } catch (...) {
          end_time_filter_ = 0;
        }
      }
    } else {
      end_time_filter_ = 0;
    }

    markDirty(); // Refresh the display with new filters
  }

  // Reset button
  ImGui::SameLine();
  if (ImGui::Button("Reset")) {
    min_size_filter_ = 0.0;
    exchange_filter_ = "";
    start_time_filter_ = 0;
    end_time_filter_ = 0;
    strcpy(min_size_input_, "0.0");
    strcpy(exchange_input_, "");
    strcpy(start_time_input_, "");
    strcpy(end_time_input_, "");
    markDirty(); // Refresh the display
  }
}

// Helper function to parse time string in HH:MM:SS format to microseconds since epoch
uint64_t TapePanel::parseTimeString(const std::string& time_str) {
  // This is a simplified parser - in a real implementation, you'd want to handle
  // date components and timezone properly
  // For now, we'll assume the time is today and convert to microseconds since epoch

  // Parse HH:MM:SS[.mmm] format
  int hours = 0, minutes = 0, seconds = 0;
  double fraction = 0.0;

  size_t pos = 0;
  try {
    // Parse hours
    size_t colon_pos = time_str.find(':');
    if (colon_pos != std::string::npos) {
      hours = std::stoi(time_str.substr(pos, colon_pos));
      pos = colon_pos + 1;

      // Parse minutes
      colon_pos = time_str.find(':', pos);
      if (colon_pos != std::string::npos) {
        minutes = std::stoi(time_str.substr(pos, colon_pos - pos));
        pos = colon_pos + 1;

        // Parse seconds and optional milliseconds
        std::string sec_part = time_str.substr(pos);
        size_t dot_pos = sec_part.find('.');
        if (dot_pos != std::string::npos) {
          seconds = std::stoi(sec_part.substr(0, dot_pos));
          fraction = std::stod(sec_part);
          fraction -= seconds; // Remove integer part
        } else {
          seconds = std::stoi(sec_part);
        }
      }
    }
  } catch (...) {
    return 0; // Return 0 if parsing fails
  }

  // Convert to seconds since midnight
  uint64_t total_seconds = hours * 3600 + minutes * 60 + seconds;

  // Get today's date and combine with time
  time_t now = time(nullptr);
  tm local_tm = *localtime(&now);
  local_tm.tm_hour = hours;
  local_tm.tm_min = minutes;
  local_tm.tm_sec = seconds;

  uint64_t timestamp = mktime(&local_tm) * 1000000; // Convert to microseconds
  timestamp += static_cast<uint64_t>(fraction * 1000000); // Add fractional microseconds

  return timestamp;
}

void TapePanel::render_controls() {
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();
  ImGui::Checkbox("Auto-scroll", &auto_scroll_);
  ImGui::SameLine();
  ImGui::Text("| Trades: %zu", cached_trades_.size());
  ImGui::SameLine();

  // Display trade pace information in the header
  if (!cached_trades_.empty()) {
    double tpm_1min = calculateTradesPerMinute(cached_trades_, 60 * 1000000);      // 1 minute in microseconds
    double tpm_5min = calculateTradesPerMinute(cached_trades_, 5 * 60 * 1000000);  // 5 minutes in microseconds
    double tpm_15min = calculateTradesPerMinute(cached_trades_, 15 * 60 * 1000000); // 15 minutes in microseconds

    ImGui::Text("| TPM: 1m:%.1f 5m:%.1f 15m:%.1f", tpm_1min, tpm_5min, tpm_15min);
  }

  ImGui::SameLine();
  ImGui::Checkbox("Filters", &show_filters_);
  ImGui::SameLine();
  ImGui::Checkbox("Histogram", &show_histogram_);

  if (show_filters_) {
    render_filter_controls();
    ImGui::Separator();
  }

  if (show_histogram_) {
    render_trade_size_histogram();
    ImGui::Separator();
  }
}

void TapePanel::render_trade_size_histogram() {
  ImGui::Separator();
  ImGui::Text("Trade Size Distribution (Logarithmic Bins)");

  // Compute the histogram
  auto histogram = computeLogarithmicTradeSizeHistogram(cached_trades_);

  if (histogram.empty()) {
    ImGui::Text("No data available for histogram");
    return;
  }

  // Create a small plot for the histogram
  char plot_title[64];
  snprintf(plot_title, sizeof(plot_title), "##TradeSizeHist_%s", config_.title.c_str());

  // Prepare data for plotting
  std::vector<double> hist_counts(histogram.size());
  std::vector<const char*> labels;  // Changed to const char*
  std::vector<std::string> label_strings; // Store the actual strings

  for (size_t i = 0; i < histogram.size(); ++i) {
    hist_counts[i] = static_cast<double>(histogram[i].count);

    // Format the label for the bucket
    char label[64];
    if (i == histogram.size() - 1) {
      // Last bucket is ">= upper bound"
      snprintf(label, sizeof(label), ">=%.3f", histogram[i].lower_bound);
    } else {
      snprintf(label, sizeof(label), "%.3f-%.3f", histogram[i].lower_bound, histogram[i].upper_bound);
    }
    label_strings.push_back(std::string(label));  // Store the string
    labels.push_back(label_strings.back().c_str());  // Add c_str pointer to labels vector
  }

  // Create a small plot area
  if (ImGui::BeginChild("TradeSizeHistogramArea", ImVec2(0, 150), true)) {
    if (ImPlot::BeginPlot(plot_title, nullptr, nullptr, ImVec2(-1, -1),
                          ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {

      ImPlot::SetupAxes(nullptr, "Count", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);

      // Set up custom x-axis labels
      std::vector<double> x_positions(histogram.size());
      for (size_t i = 0; i < histogram.size(); ++i) {
        x_positions[i] = static_cast<double>(i);
      }

      // Plot the histogram bars
      ImPlot::PlotBars("Counts", x_positions.data(), hist_counts.data(), static_cast<int>(histogram.size()), 0.8);

      // Set custom x-axis labels - need to convert vector<const char*> to const char* const*
      // This is tricky, so let's use a different approach
      // Just remove the custom axis scale setup since we don't need it for a simple histogram

      ImPlot::EndPlot();
    }
    ImGui::EndChild();
  }

  // Display detailed statistics in a table
  if (ImGui::CollapsingHeader("Detailed Statistics")) {
    if (ImGui::BeginTable("TradeSizeStats", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY)) {
      ImGui::TableSetupColumn("Range", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 80.0f);
      ImGui::TableSetupColumn("Total Size", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Avg Size", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Percentage", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableHeadersRow();

      int total_trades = 0;
      for (const auto& bucket : histogram) {
        total_trades += bucket.count;
      }

      for (const auto& bucket : histogram) {
        ImGui::TableNextRow();

        // Range column
        ImGui::TableSetColumnIndex(0);
        if (&bucket == &histogram.back()) {
          ImGui::Text(">=%.3f", bucket.lower_bound);
        } else {
          ImGui::Text("%.3f-%.3f", bucket.lower_bound, bucket.upper_bound);
        }

        // Count column
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%d", bucket.count);

        // Total Size column
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.2f", bucket.total_size);

        // Average Size column
        ImGui::TableSetColumnIndex(3);
        if (bucket.count > 0) {
          ImGui::Text("%.3f", bucket.total_size / bucket.count);
        } else {
          ImGui::Text("0.000");
        }

        // Percentage column
        ImGui::TableSetColumnIndex(4);
        if (total_trades > 0) {
          double percentage = (static_cast<double>(bucket.count) / total_trades) * 100.0;
          ImGui::Text("%.2f%%", percentage);
        } else {
          ImGui::Text("0.00%%");
        }
      }

      ImGui::EndTable();
    }
  }
}

void TapePanel::render_trade_table() {
  // Unique table ID per panel instance to avoid ID conflicts
  char table_id[64];
  snprintf(table_id, sizeof(table_id), "TapeTable##%s", config_.title.c_str());

  // Calculate average trade size for thresholds
  double total_size = 0.0;
  int valid_trade_count = 0;
  for (const auto& trade : cached_trades_) {
      // Apply filters to calculate average only on filtered trades
      if (trade.size < min_size_filter_) continue;
      if (!exchange_filter_.empty()) {
          std::string exchange = bridge_ ? bridge_->getExchangeName(trade.symbol_id) : "";
          if (exchange != exchange_filter_) continue;
      }
      if (start_time_filter_ > 0 && trade.timestamp < start_time_filter_) continue;
      if (end_time_filter_ > 0 && trade.timestamp > end_time_filter_) continue;

      total_size += trade.size;
      valid_trade_count++;
  }

  double avg_trade_size = (valid_trade_count > 0) ? total_size / valid_trade_count : 0.0;
  double large_trade_threshold = avg_trade_size * 5.0;
  double block_trade_threshold = avg_trade_size * 10.0;

  if (ImGui::BeginTable(table_id, 4,
                        ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable)) {
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Side", ImGuiTableColumnFlags_WidthFixed, 40.0f);
    ImGui::TableHeadersRow();

    // Count filtered trades to determine the total for the clipper
    int filtered_trade_count = 0;
    for (const auto& trade : cached_trades_) {
        // Apply filters
        if (trade.size < min_size_filter_) continue;
        if (!exchange_filter_.empty()) {
            std::string exchange = bridge_ ? bridge_->getExchangeName(trade.symbol_id) : "";
            if (exchange != exchange_filter_) continue;
        }
        if (start_time_filter_ > 0 && trade.timestamp < start_time_filter_) continue;
        if (end_time_filter_ > 0 && trade.timestamp > end_time_filter_) continue;

        filtered_trade_count++;
    }

    // Use ImGuiListClipper for virtualized scrolling to handle 100,000+ trades efficiently
    ImGuiListClipper clipper;
    clipper.Begin(filtered_trade_count);

    int filtered_index = 0; // Index in the filtered list
    int original_index = static_cast<int>(cached_trades_.size()) - 1; // Start from newest trade

    while (original_index >= 0 && clipper.Step()) {
        // Process all rows in the current clipper step
        while (clipper.DisplayStart < clipper.DisplayEnd && original_index >= 0) {
            const auto& trade = cached_trades_[original_index];

            // Apply filters
            bool skip_trade = false;
            if (trade.size < min_size_filter_) skip_trade = true;
            if (!skip_trade && !exchange_filter_.empty()) {
                std::string exchange = bridge_ ? bridge_->getExchangeName(trade.symbol_id) : "";
                if (exchange != exchange_filter_) skip_trade = true;
            }
            if (!skip_trade && start_time_filter_ > 0 && trade.timestamp < start_time_filter_) skip_trade = true;
            if (!skip_trade && end_time_filter_ > 0 && trade.timestamp > end_time_filter_) skip_trade = true;

            if (!skip_trade) {
                // This trade passes all filters
                if (filtered_index >= clipper.DisplayStart) {
                    // This trade should be displayed in the current view
                    ImGui::PushID(filtered_index);
                    ImGui::TableNextRow();

                    // Check if this trade is part of a cluster
                    bool is_clustered = isTradeClustered(original_index, cached_trades_);

                    // Set background color for clustered trades
                    if (is_clustered) {
                        const auto& colors = ThemeManager::getInstance().getColors();
                        ImU32 cluster_bg_color = ImGui::GetColorU32(ImVec4(0.8f, 0.6f, 0.2f, 0.3f)); // Light amber with transparency
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, cluster_bg_color);
                    }

                    // Time column (HH:MM:SS.mmm)
                    ImGui::TableSetColumnIndex(0);
                    if (trade.timestamp > 0) {
                        time_t time_sec = trade.timestamp / 1000000;  // micros to seconds
                        uint64_t millis = (trade.timestamp / 1000) % 1000;
                        char time_str[16];
                        strftime(time_str, sizeof(time_str), "%H:%M:%S", localtime(&time_sec));
                        ImGui::Text("%s.%03lu", time_str, static_cast<unsigned long>(millis));
                    } else {
                        ImGui::Text("-");
                    }

                    // Price column - Enhanced coloring based on trade size
                    ImGui::TableSetColumnIndex(1);
                    const auto& colors = ThemeManager::getInstance().getColors();

                    // Determine color based on trade size and direction
                    ImVec4 price_color = colors.text; // Default color
                    if (trade.size >= block_trade_threshold && block_trade_threshold > 0) {
                        // Block trades (>avg*10) in orange
                        price_color = ImVec4(1.0f, 0.5f, 0.0f, 1.0f); // Orange
                    } else if (trade.size >= large_trade_threshold && large_trade_threshold > 0) {
                        // Large trades (>avg*5) in yellow
                        price_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow
                    } else {
                        // Regular trades: buy in green, sell in red
                        price_color = trade.is_buy ? colors.accent_green : colors.accent_red;
                    }

                    // For block trades, we'll use a slightly different approach since we can't guarantee bold font availability
                    // Instead, we'll just use the distinctive orange color and potentially add other visual indicators later
                    ImGui::TextColored(price_color, "%.4f", trade.price);

                    // Size column
                    ImGui::TableSetColumnIndex(2);
                    // Color size based on trade size thresholds
                    ImVec4 size_color = colors.text; // Default color
                    if (trade.size >= block_trade_threshold && block_trade_threshold > 0) {
                        // Block trades in orange
                        size_color = ImVec4(1.0f, 0.5f, 0.0f, 1.0f); // Orange
                    } else if (trade.size >= large_trade_threshold && large_trade_threshold > 0) {
                        // Large trades in yellow
                        size_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow
                    }

                    ImGui::TextColored(size_color, "%.4f", trade.size);

                    // Side column
                    ImGui::TableSetColumnIndex(3);
                    ImVec4 side_color = colors.text; // Default color
                    if (trade.size >= block_trade_threshold && block_trade_threshold > 0) {
                        // Block trades in orange
                        side_color = ImVec4(1.0f, 0.5f, 0.0f, 1.0f); // Orange
                    } else if (trade.size >= large_trade_threshold && large_trade_threshold > 0) {
                        // Large trades in yellow
                        side_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow
                    } else {
                        // Regular trades: buy in green, sell in red
                        side_color = trade.is_buy ? colors.accent_green : colors.accent_red;
                    }

                    if (trade.is_buy) {
                        ImGui::TextColored(side_color, "BUY");
                    } else {
                        ImGui::TextColored(side_color, "SELL");
                    }

                    ImGui::PopID();

                    if (filtered_index >= clipper.DisplayEnd - 1) break; // Move to next clipper step
                }
                filtered_index++;
            }
            original_index--;
        }
    }

    // Auto-scroll to bottom (newest trades)
    if (auto_scroll_ && filtered_trade_count > 0) {
        ImGui::SetScrollHereY(0.0f);
    }

    ImGui::EndTable();
}

}  // namespace BTQuant

// Compute logarithmic trade size histogram
std::vector<BTQuant::TapePanel::LogBucket> BTQuant::TapePanel::computeLogarithmicTradeSizeHistogram(const std::vector<RenderEngine::TradeData>& trades) const {
  // Define logarithmic bucket boundaries (base 10)
  // Starting from 0.001 up to 10000 with logarithmic spacing
  std::vector<double> bucket_bounds = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0};

  // Initialize buckets
  std::vector<LogBucket> buckets;
  for (size_t i = 0; i < bucket_bounds.size() - 1; ++i) {
    LogBucket bucket;
    bucket.lower_bound = bucket_bounds[i];
    bucket.upper_bound = bucket_bounds[i + 1];
    bucket.count = 0;
    bucket.total_size = 0.0;
    buckets.push_back(bucket);
  }

  // Process each trade
  for (const auto& trade : trades) {
    // Skip trades with size <= 0
    if (trade.size <= 0.0) continue;

    // Find the appropriate bucket for this trade size
    bool placed_in_bucket = false;
    for (auto& bucket : buckets) {
      if (trade.size >= bucket.lower_bound && trade.size < bucket.upper_bound) {
        bucket.count++;
        bucket.total_size += trade.size;
        placed_in_bucket = true;
        break;
      }
    }

    // If trade size is larger than the largest bucket, put it in the last bucket
    if (!placed_in_bucket && trade.size >= buckets.back().lower_bound) {
      buckets.back().count++;
      buckets.back().total_size += trade.size;
    }
  }

  return buckets;
}

// Check if a trade at the given index is part of a cluster of rapid trades at the same price
bool BTQuant::TapePanel::isTradeClustered(int index, const std::vector<RenderEngine::TradeData>& trades) const {
    if (index < 0 || index >= static_cast<int>(trades.size())) {
        return false;
    }

    const auto& current_trade = trades[index];

    // Look backward and forward to find trades at the same price within the time window
    int cluster_count = 1; // Start with 1 for the current trade

    // Look backward (older trades)
    for (int i = index + 1; i < static_cast<int>(trades.size()); ++i) {
        const auto& trade = trades[i];

        // Check if price matches (within tolerance)
        if (std::abs(trade.price - current_trade.price) <= PRICE_MATCH_TOLERANCE) {
            // Check if timestamp is within the clustering window
            if (current_trade.timestamp >= trade.timestamp &&
                current_trade.timestamp - trade.timestamp <= CLUSTER_TIME_WINDOW_US) {
                cluster_count++;
            } else {
                // Stop looking if we're outside the time window
                break;
            }
        }
    }

    // Look forward (newer trades)
    for (int i = index - 1; i >= 0; --i) {
        const auto& trade = trades[i];

        // Check if price matches (within tolerance)
        if (std::abs(trade.price - current_trade.price) <= PRICE_MATCH_TOLERANCE) {
            // Check if timestamp is within the clustering window
            if (trade.timestamp >= current_trade.timestamp &&
                trade.timestamp - current_trade.timestamp <= CLUSTER_TIME_WINDOW_US) {
                cluster_count++;
            } else {
                // Stop looking if we're outside the time window
                break;
            }
        }
    }

    // Return true if we found enough trades in the cluster
    return cluster_count >= MIN_CLUSTER_SIZE;
}

// Calculate trades per minute for a given time window
double BTQuant::TapePanel::calculateTradesPerMinute(const std::vector<RenderEngine::TradeData>& trades, uint64_t window_microseconds) const {
    if (trades.empty()) {
        return 0.0;
    }

    // Get the current time (latest trade timestamp)
    uint64_t current_time = trades.back().timestamp;
    uint64_t start_time = current_time - window_microseconds;

    // Count trades within the time window
    int trade_count = 0;
    for (const auto& trade : trades) {
        if (trade.timestamp >= start_time && trade.timestamp <= current_time) {
            trade_count++;
        }
    }

    // Convert microseconds to minutes for the rate calculation
    double window_minutes = static_cast<double>(window_microseconds) / (1000000.0 * 60.0);

    // Calculate trades per minute
    if (window_minutes > 0.0) {
        return static_cast<double>(trade_count) / window_minutes;
    } else {
        return 0.0;
    }
}

// Update trade pace history for all time windows
void BTQuant::TapePanel::updateTradePaceHistory() {
    if (cached_trades_.empty()) {
        return;
    }

    // Calculate current trades per minute for each time window
    double tpm_1min = calculateTradesPerMinute(cached_trades_, 60 * 1000000);      // 1 minute in microseconds
    double tpm_5min = calculateTradesPerMinute(cached_trades_, 5 * 60 * 1000000);  // 5 minutes in microseconds
    double tpm_15min = calculateTradesPerMinute(cached_trades_, 15 * 60 * 1000000); // 15 minutes in microseconds

    // Add current measurements to history
    uint64_t current_time = cached_trades_.back().timestamp;

    trade_pace_1min_history_.push_back({current_time, tpm_1min});
    trade_pace_5min_history_.push_back({current_time, tpm_5min});
    trade_pace_15min_history_.push_back({current_time, tpm_15min});

    // Limit history size to prevent memory bloat
    const size_t MAX_HISTORY_SIZE = 1000;
    if (trade_pace_1min_history_.size() > MAX_HISTORY_SIZE) {
        trade_pace_1min_history_.erase(trade_pace_1min_history_.begin(),
                                      trade_pace_1min_history_.begin() + (trade_pace_1min_history_.size() - MAX_HISTORY_SIZE));
    }
    if (trade_pace_5min_history_.size() > MAX_HISTORY_SIZE) {
        trade_pace_5min_history_.erase(trade_pace_5min_history_.begin(),
                                      trade_pace_5min_history_.begin() + (trade_pace_5min_history_.size() - MAX_HISTORY_SIZE));
    }
    if (trade_pace_15min_history_.size() > MAX_HISTORY_SIZE) {
        trade_pace_15min_history_.erase(trade_pace_15min_history_.begin(),
                                       trade_pace_15min_history_.begin() + (trade_pace_15min_history_.size() - MAX_HISTORY_SIZE));
    }
}

// Render the trade pace chart in the header
void BTQuant::TapePanel::renderTradePaceChart() {
    if (trade_pace_1min_history_.empty()) {
        return;
    }

    // Create a small plot for the trade pace chart
    char plot_title[64];
    snprintf(plot_title, sizeof(plot_title), "##TradePace_%s", config_.title.c_str());

    // Prepare data for plotting - only show the last N points to keep the chart readable
    const size_t DISPLAY_POINTS = 50;  // Show last 50 points
    size_t start_idx = trade_pace_1min_history_.size() > DISPLAY_POINTS ?
                      trade_pace_1min_history_.size() - DISPLAY_POINTS : 0;

    std::vector<double> timestamps;
    std::vector<double> tpm_1min_values;
    std::vector<double> tpm_5min_values;
    std::vector<double> tpm_15min_values;

    for (size_t i = start_idx; i < trade_pace_1min_history_.size(); ++i) {
        // Normalize timestamps for the plot (relative to the first shown point)
        double normalized_time = static_cast<double>(trade_pace_1min_history_[i].timestamp - trade_pace_1min_history_[start_idx].timestamp) / 1000000.0; // Convert to seconds
        timestamps.push_back(normalized_time);

        tpm_1min_values.push_back(trade_pace_1min_history_[i].trades_per_minute);
        tpm_5min_values.push_back(trade_pace_5min_history_[i].trades_per_minute);
        tpm_15min_values.push_back(trade_pace_15min_history_[i].trades_per_minute);
    }

    // Create a small plot area in the header
    if (ImGui::BeginChild("TradePaceChartArea", ImVec2(0, 80), true)) {
        if (ImPlot::BeginPlot(plot_title, "Time (s)", "Trades/Min", ImVec2(-1, -1),
                              ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {

            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);

            // Plot the three different time windows
            if (!tpm_1min_values.empty()) {
                ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f); // Red for 1min
                ImPlot::PlotLine("1 Min", timestamps.data(), tpm_1min_values.data(), static_cast<int>(timestamps.size()));
            }

            if (!tpm_5min_values.empty()) {
                ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 1.0f); // Green for 5min
                ImPlot::PlotLine("5 Min", timestamps.data(), tpm_5min_values.data(), static_cast<int>(timestamps.size()));
            }

            if (!tpm_15min_values.empty()) {
                ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.0f, 1.0f, 1.0f), 1.0f); // Blue for 15min
                ImPlot::PlotLine("15 Min", timestamps.data(), tpm_15min_values.data(), static_cast<int>(timestamps.size()));
            }

            ImPlot::EndPlot();
        }
        ImGui::EndChild();
    }
}
}  // namespace BTQuant
