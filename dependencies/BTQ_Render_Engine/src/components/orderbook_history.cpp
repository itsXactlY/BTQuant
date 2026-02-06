#include "../../include/components/orderbook_history.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "imgui.h"
#include "implot.h"
#include "components/theme_manager.hpp"

namespace BTQuant {

OrderbookHistoryPanel::OrderbookHistoryPanel(const PanelConfig& config,
                                           std::shared_ptr<HotSpineDataBridge> bridge,
                                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
    // Initialize with current time for last capture to prevent immediate capture
    last_capture_time_ = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void OrderbookHistoryPanel::update(float /*dt*/) {
    // Request data update from data bridge
    bridge_->sync();

    // Check if it's time to capture a new snapshot
    uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
    if (current_time - last_capture_time_ >= static_cast<uint64_t>(capture_interval_ms_ * 1000)) {
        // Get active symbols to determine which one to capture
        auto active_symbols = processor_->getActiveSymbols();
        if (!active_symbols.empty()) {
            uint32_t sym_id = active_symbols[0]; // Use first active symbol
            std::string sym_name = bridge_->getSymbolName(sym_id);
            
            // Capture snapshot if we have valid orderbook data
            auto orderbook_opt = processor_->getOrderbookData(sym_id);
            if (orderbook_opt.has_value()) {
                captureSnapshot(sym_id, sym_name);
                last_capture_time_ = current_time;
            }
        }
    }

    // Handle playback if enabled
    if (is_playing_ && !snapshots_.empty()) {
        uint64_t now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
        if (last_playback_time_ == 0) {
            last_playback_time_ = now;
        }
        
        // Calculate time elapsed since last playback update
        uint64_t elapsed_us = now - last_playback_time_;
        double elapsed_ms = elapsed_us / 1000.0;
        
        // Adjust for playback speed
        double adjusted_elapsed_ms = elapsed_ms * playback_speed_;
        
        // Move to next snapshot based on adjusted elapsed time
        if (adjusted_elapsed_ms >= capture_interval_ms_) {
            goToNextSnapshot();
            last_playback_time_ = now;
        }
    }
}

void OrderbookHistoryPanel::render() {
    begin_panel_window();

    // If panel is hidden via X button, we still need to call end
    if (!is_visible()) {
        end_panel_window();
        return;
    }

    // Control panel at the top
    ImGui::Text("Orderbook History Controls");
    ImGui::Separator();

    // Capture settings
    ImGui::Text("Capture Settings:");
    ImGui::SameLine();
    ImGui::PushItemWidth(100);
    ImGui::InputInt("Interval (ms)", &capture_interval_ms_);
    if (capture_interval_ms_ < 100) capture_interval_ms_ = 100; // Minimum 100ms
    ImGui::PopItemWidth();
    
    ImGui::SameLine();
    ImGui::PushItemWidth(100);
    size_t max_snapshots_ui = static_cast<size_t>(max_snapshots_);
    if (ImGui::InputScalar("Max Snapshots", ImGuiDataType_U64, &max_snapshots_ui)) {
        max_snapshots_ = static_cast<size_t>(max_snapshots_ui);
    }
    ImGui::PopItemWidth();

    // Playback controls
    ImGui::Text("Playback Controls:");
    
    if (ImGui::Button(is_playing_ ? "Pause" : "Play")) {
        if (is_playing_) {
            pausePlayback();
        } else {
            startPlayback();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        stopPlayback();
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(100);
    ImGui::SliderFloat("Speed", &playback_speed_, 0.1f, 5.0f, "%.1fx");
    ImGui::PopItemWidth();

    // Navigation controls
    ImGui::Text("Navigation:");
    if (ImGui::Button("<< Prev") || ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        goToPreviousSnapshot();
    }
    ImGui::SameLine();
    if (ImGui::Button("Next >>") || ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
        goToNextSnapshot();
    }
    ImGui::SameLine();
    
    // Snapshot index selector
    if (!snapshots_.empty()) {
        int current_idx = current_snapshot_index_ >= 0 ? current_snapshot_index_ : 0;
        if (ImGui::SliderInt("Snapshot", &current_idx, 0, static_cast<int>(snapshots_.size()) - 1)) {
            goToSnapshot(current_idx);
        }
        ImGui::SameLine();
        ImGui::Text("(%zu total)", snapshots_.size());
    } else {
        ImGui::Text("(No snapshots captured yet)");
    }

    // Comparison controls
    ImGui::Checkbox("Enable Comparison", &enable_comparison_);
    if (enable_comparison_ && !snapshots_.empty()) {
        ImGui::SameLine();
        int comp_idx = comparison_snapshot_index_ >= 0 ? comparison_snapshot_index_ : 0;
        if (ImGui::SliderInt("Compare With", &comp_idx, 0, static_cast<int>(snapshots_.size()) - 1)) {
            setComparisonSnapshot(comp_idx);
        }
    }

    // Level count selector
    ImGui::Text("Display Settings:");
    ImGui::SameLine();
    ImGui::Text("Levels:");
    ImGui::SameLine();
    ImGui::PushItemWidth(100);
    if (ImGui::BeginCombo("##LevelCount", LEVEL_OPTION_NAMES[get_level_option_index()])) {
        for (int i = 0; i < 6; ++i) {
            bool is_selected = (LEVEL_OPTIONS[i] == selected_levels_count_);
            if (ImGui::Selectable(LEVEL_OPTION_NAMES[i], is_selected)) {
                selected_levels_count_ = LEVEL_OPTIONS[i];
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::Separator();

    // Display snapshots
    if (enable_comparison_ && comparison_snapshot_index_ >= 0 && 
        current_snapshot_index_ >= 0 && snapshots_.size() > static_cast<size_t>(comparison_snapshot_index_) && 
        snapshots_.size() > static_cast<size_t>(current_snapshot_index_)) {
        // Render comparison view
        render_comparison_view();
    } else if (current_snapshot_index_ >= 0 && snapshots_.size() > static_cast<size_t>(current_snapshot_index_)) {
        // Render current snapshot
        const auto& snapshot = snapshots_[current_snapshot_index_];
        std::string title = "Current State (" + std::to_string(current_snapshot_index_) + ")";
        render_orderbook_ladder(snapshot.bids, snapshot.asks, title);
    } else if (!snapshots_.empty()) {
        // Render latest snapshot if current index is invalid
        const auto& snapshot = snapshots_.back();
        std::string title = "Latest State (" + std::to_string(snapshots_.size()-1) + ")";
        render_orderbook_ladder(snapshot.bids, snapshot.asks, title);
    } else {
        ImGui::Text("No snapshots available. Waiting for data...");
    }

    end_panel_window();
}

void OrderbookHistoryPanel::captureSnapshot(uint32_t symbol_id, const std::string& symbol_name) {
    auto orderbook_opt = processor_->getOrderbookData(symbol_id);
    if (orderbook_opt.has_value()) {
        const auto& orderbook = orderbook_opt.value();
        
        // Create new snapshot
        OrderbookSnapshot snapshot(orderbook, symbol_id);
        snapshot.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        // Add to snapshots
        snapshots_.push_back(snapshot);
        
        // Update symbol info
        symbol_id_ = symbol_id;
        symbol_name_ = symbol_name;
        config_.title = symbol_name + " Orderbook History";
        
        // Clean up old snapshots if we exceed the limit
        cleanupOldSnapshots();
        
        // Update current index to the new snapshot
        current_snapshot_index_ = static_cast<int>(snapshots_.size()) - 1;
    }
}

void OrderbookHistoryPanel::startPlayback() {
    if (!snapshots_.empty()) {
        is_playing_ = true;
        last_playback_time_ = 0; // Reset to trigger immediate update
    }
}

void OrderbookHistoryPanel::pausePlayback() {
    is_playing_ = false;
}

void OrderbookHistoryPanel::stopPlayback() {
    is_playing_ = false;
    if (!snapshots_.empty()) {
        current_snapshot_index_ = static_cast<int>(snapshots_.size()) - 1;
    }
}

void OrderbookHistoryPanel::setPlaybackSpeed(float speed) {
    playback_speed_ = speed;
}

void OrderbookHistoryPanel::goToPreviousSnapshot() {
    if (!snapshots_.empty() && current_snapshot_index_ > 0) {
        current_snapshot_index_--;
    }
}

void OrderbookHistoryPanel::goToNextSnapshot() {
    if (!snapshots_.empty() && current_snapshot_index_ < static_cast<int>(snapshots_.size()) - 1) {
        current_snapshot_index_++;
    }
}

void OrderbookHistoryPanel::goToSnapshot(int index) {
    if (index >= 0 && static_cast<size_t>(index) < snapshots_.size()) {
        current_snapshot_index_ = index;
    }
}

void OrderbookHistoryPanel::setCaptureIntervalMs(int interval_ms) {
    if (interval_ms >= 100) {  // Minimum 100ms
        capture_interval_ms_ = interval_ms;
    }
}

void OrderbookHistoryPanel::setMaxSnapshots(size_t max_snapshots) {
    max_snapshots_ = max_snapshots;
    cleanupOldSnapshots();
}

void OrderbookHistoryPanel::enableComparison(bool enable) {
    enable_comparison_ = enable;
}

void OrderbookHistoryPanel::setComparisonSnapshot(int index) {
    if (index >= 0 && static_cast<size_t>(index) < snapshots_.size()) {
        comparison_snapshot_index_ = index;
    }
}

int OrderbookHistoryPanel::get_level_option_index() {
    for (int i = 0; i < 6; ++i) {
        if (LEVEL_OPTIONS[i] == selected_levels_count_) {
            return i;
        }
    }
    return 1; // Default to 20 if not found
}

void OrderbookHistoryPanel::render_orderbook_ladder(const std::vector<RenderEngine::PriceLevel>& bids,
                                                   const std::vector<RenderEngine::PriceLevel>& asks,
                                                   const std::string& title) {
    if (!title.empty()) {
        ImGui::Text("%s", title.c_str());
    }

    // Calculate statistics
    double avg_size = calculateAverageOrderSize(bids, asks);
    double max_vol = getMaxVolume(bids, asks);
    if (max_vol < 1.0) max_vol = 1.0;

    // Calculate cumulative volumes for liquidity bars
    std::vector<double> cumulative_bids(bids.size());
    std::vector<double> cumulative_asks(asks.size());

    // Calculate cumulative bid volumes (from best bid outward)
    double bid_sum = 0.0;
    for (size_t i = 0; i < bids.size(); ++i) {
        bid_sum += bids[i].size;
        cumulative_bids[i] = bid_sum;
    }

    // Calculate cumulative ask volumes (from best ask outward)
    double ask_sum = 0.0;
    for (size_t i = 0; i < asks.size(); ++i) {
        ask_sum += asks[i].size;
        cumulative_asks[i] = ask_sum;
    }

    // Find max cumulative volume for scaling
    double max_cumulative_vol = max_vol; // fallback to individual max if no cumulative data
    if (!cumulative_bids.empty()) max_cumulative_vol = std::max(max_cumulative_vol, cumulative_bids.back());
    if (!cumulative_asks.empty()) max_cumulative_vol = std::max(max_cumulative_vol, cumulative_asks.back());

    // Use Table instead of Columns for modern layout
    if (ImGui::BeginTable("OrderbookHistoryTable", 6,
                          ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame)) {
        // Setup Columns
        ImGui::TableSetupColumn("Bid", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Ask", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Bid Cumulative", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Ask Cumulative", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Spread", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableHeadersRow();

        const auto& colors = ThemeManager::getInstance().getColors();

        // Determine how many levels to show based on selected_levels_count_
        int max_levels_to_show = selected_levels_count_ == -1 ?
                                 std::max(static_cast<int>(asks.size()), static_cast<int>(bids.size())) :
                                 selected_levels_count_;

        // Use channel splitting to draw backgrounds before text content
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->ChannelsSplit(2); // Split into 2 channels: 0 for backgrounds, 1 for text (default)

        // Switch to background channel (0) to draw heatmap backgrounds first
        draw_list->ChannelsSetCurrent(0);

        // Render Asks (Sell) - Top down, but only to calculate positions
        int ask_count = std::min(static_cast<int>(asks.size()), max_levels_to_show);
        for (int i = ask_count - 1; i >= 0; --i) {
            const auto& level = asks[i];
            ImGui::TableNextRow();

            // Calculate heatmap intensity for this level
            float intensity = std::clamp((float)(level.size / max_vol), 0.0f, 1.0f);
            if (intensity > 0.05f) {
                // Calculate position for the entire row background
                ImVec2 row_pos = ImGui::GetCursorScreenPos();
                float row_height = ImGui::GetTextLineHeightWithSpacing();

                // Get the width of the table row
                float table_width = ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX();

                // Calculate the background rectangle for the entire row
                ImVec2 pos_min = row_pos;
                ImVec2 pos_max = ImVec2(row_pos.x + table_width, row_pos.y + row_height);

                // Red heatmap for asks
                ImU32 bg_color = ImGui::GetColorU32(ImVec4(1.0f, 0.5f, 0.0f, intensity * 0.3f));

                // Draw the rectangle in the background channel
                draw_list->AddRectFilled(pos_min, pos_max, bg_color);
            }
        }

        // Spread Row - also need to account for this in positioning
        ImGui::TableNextRow();

        // Render Bids (Buy) - but only to calculate positions
        int bid_count = std::min(static_cast<int>(bids.size()), max_levels_to_show);
        for (int i = 0; i < bid_count; ++i) {
            const auto& level = bids[i];
            ImGui::TableNextRow();

            // Calculate heatmap intensity for this level
            float intensity = std::clamp((float)(level.size / max_vol), 0.0f, 1.0f);
            if (intensity > 0.05f) {
                // Calculate position for the entire row background
                ImVec2 row_pos = ImGui::GetCursorScreenPos();
                float row_height = ImGui::GetTextLineHeightWithSpacing();

                // Get the width of the table row
                float table_width = ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX();

                // Calculate the background rectangle for the entire row
                ImVec2 pos_min = row_pos;
                ImVec2 pos_max = ImVec2(row_pos.x + table_width, row_pos.y + row_height);

                // Blue heatmap for bids
                ImU32 bg_color = ImGui::GetColorU32(ImVec4(0.0f, 0.6f, 1.0f, intensity * 0.3f));

                // Draw the rectangle in the background channel
                draw_list->AddRectFilled(pos_min, pos_max, bg_color);
            }
        }

        // Switch back to the default channel (1) for text content
        draw_list->ChannelsSetCurrent(1);

        // Now render the actual content in the default channel
        // Render Asks (Sell) - Top down
        for (int i = ask_count - 1; i >= 0; --i) {
            const auto& level = asks[i];
            ImGui::TableNextRow();
            ImGui::PushID(i);  // Unique ID for this row/side

            // Check if this is a large order
            bool is_large_order = avg_size > 0 &&
                                 (level.size / avg_size) * 100.0 >= 200.0; // Using 200% as threshold

            // 1. Bid (Empty)
            ImGui::TableSetColumnIndex(0);

            // 2. Price
            ImGui::TableSetColumnIndex(1);
            // Center Price text
            float cursor_check =
                ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x -
                                          ImGui::CalcTextSize(std::to_string(level.price).c_str()).x) *
                                             0.5f;
            ImGui::SetCursorPosX(cursor_check);

            if (is_large_order) {
                // Draw yellow background for the entire price cell
                ImVec2 cell_pos = ImGui::GetCursorScreenPos();
                ImVec2 cell_size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeightWithSpacing());
                ImGui::GetWindowDrawList()->AddRectFilled(
                    cell_pos,
                    ImVec2(cell_pos.x + cell_size.x, cell_pos.y + cell_size.y),
                    ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.3f))); // Semi-transparent yellow background
            }

            ImGui::Text("%.2f", level.price);
            ImGui::SameLine();
            if (is_large_order) {
                // Draw yellow background for large orders
                ImVec2 text_pos = ImGui::GetCursorScreenPos();
                ImVec2 text_size = ImGui::CalcTextSize(std::format("%.2f", level.price).c_str());
                ImGui::GetWindowDrawList()->AddRectFilled(
                    text_pos,
                    ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                    ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

                // Draw text with increased weight effect
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
                ImGui::TextColored(colors.accent_red, "%.2f", level.price);
                ImGui::PopStyleColor();
            } else {
                ImGui::TextColored(colors.accent_red, "%.2f", level.price);
            }

            // 3. Ask Size (with Bar)
            ImGui::TableSetColumnIndex(2);
            {
                float width = ImGui::GetContentRegionAvail().x;
                float bar_width = width * (float)(level.size / max_vol);
                ImVec2 pos = ImGui::GetCursorScreenPos();

                ImGui::GetWindowDrawList()->AddRectFilled(
                    pos, ImVec2(pos.x + bar_width, pos.y + ImGui::GetTextLineHeightWithSpacing()),
                    ImGui::GetColorU32(
                        ImVec4(colors.accent_red.x, colors.accent_red.y, colors.accent_red.z, 0.2f)));

                if (is_large_order) {
                    // Draw yellow background for large orders
                    ImVec2 text_pos = ImGui::GetCursorScreenPos();
                    ImVec2 text_size = ImGui::CalcTextSize(std::format("%.4f", level.size).c_str());
                    ImGui::GetWindowDrawList()->AddRectFilled(
                        text_pos,
                        ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                        ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

                    // Draw text with increased weight effect
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
                    ImGui::Text("%.4f", level.size);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::Text("%.4f", level.size);
                }
            }

            // 4. Bid Cumulative (Empty for asks)
            ImGui::TableSetColumnIndex(3);

            // 5. Ask Cumulative
            ImGui::TableSetColumnIndex(4);
            if (i < static_cast<int>(cumulative_asks.size())) {
                ImGui::Text("%.2f", cumulative_asks[i]);
            }

            // 6. Spread (Empty for regular rows)
            ImGui::TableSetColumnIndex(5);

            ImGui::PopID();
        }

        // Spread Row
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("--- SPREAD ---");

        // Render Bids (Buy)
        for (int i = 0; i < bid_count; ++i) {
            const auto& level = bids[i];
            ImGui::TableNextRow();
            ImGui::PushID(i + 1000);  // Offset to ensure uniqueness from Asks

            // Check if this is a large order
            bool is_large_order = avg_size > 0 &&
                                 (level.size / avg_size) * 100.0 >= 200.0; // Using 200% as threshold

            // 1. Bid Size (with Bar)
            ImGui::TableSetColumnIndex(0);
            {
                // Draw bar from right to left for Bid to "point" to Price
                float width = ImGui::GetContentRegionAvail().x;
                float bar_width = width * (float)(level.size / max_vol);
                ImVec2 pos = ImGui::GetCursorScreenPos();

                ImGui::GetWindowDrawList()->AddRectFilled(
                    ImVec2(pos.x + width - bar_width, pos.y),
                    ImVec2(pos.x + width, pos.y + ImGui::GetTextLineHeightWithSpacing()),
                    ImGui::GetColorU32(
                        ImVec4(colors.accent_green.x, colors.accent_green.y, colors.accent_green.z, 0.2f)));

                // Text Right Aligned
                auto text = std::format("{:.4f}", level.size);
                float text_width = ImGui::CalcTextSize(text.c_str()).x;
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + width - text_width);

                if (is_large_order) {
                    // Draw yellow background for large orders
                    ImVec2 text_pos = ImGui::GetCursorScreenPos();
                    ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
                    ImGui::GetWindowDrawList()->AddRectFilled(
                        text_pos,
                        ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                        ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

                    // Draw text with increased weight effect
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
                    ImGui::TextUnformatted(text.c_str());
                    ImGui::PopStyleColor();
                } else {
                    ImGui::TextUnformatted(text.c_str());
                }
            }

            // 2. Price
            ImGui::TableSetColumnIndex(1);
            float cursor_check =
                ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x -
                                          ImGui::CalcTextSize(std::to_string(level.price).c_str()).x) *
                                             0.5f;
            ImGui::SetCursorPosX(cursor_check);

            if (is_large_order) {
                // Draw yellow background for the entire price cell
                ImVec2 cell_pos = ImGui::GetCursorScreenPos();
                ImVec2 cell_size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeightWithSpacing());
                ImGui::GetWindowDrawList()->AddRectFilled(
                    cell_pos,
                    ImVec2(cell_pos.x + cell_size.x, cell_pos.y + cell_size.y),
                    ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.3f))); // Semi-transparent yellow background
            }

            ImGui::Text("%.2f", level.price);
            ImGui::SameLine();
            if (is_large_order) {
                // Draw yellow background for large orders
                ImVec2 text_pos = ImGui::GetCursorScreenPos();
                ImVec2 text_size = ImGui::CalcTextSize(std::format("%.2f", level.price).c_str());
                ImGui::GetWindowDrawList()->AddRectFilled(
                    text_pos,
                    ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                    ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

                // Draw text with increased weight effect
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
                ImGui::TextColored(colors.accent_green, "%.2f",
                                   level.price);  // Green for Bid Price
                ImGui::PopStyleColor();
            } else {
                ImGui::TextColored(colors.accent_green, "%.2f",
                                   level.price);  // Green for Bid Price
            }

            // 3. Ask (Empty)
            ImGui::TableSetColumnIndex(2);

            // 4. Bid Cumulative
            ImGui::TableSetColumnIndex(3);
            if (i < static_cast<int>(cumulative_bids.size())) {
                ImGui::Text("%.2f", cumulative_bids[i]);
            }

            // 5. Ask Cumulative (Empty for bids)
            ImGui::TableSetColumnIndex(4);

            // 6. Spread (Empty for regular rows)
            ImGui::TableSetColumnIndex(5);

            ImGui::PopID();
        }

        // Merge the channels back together
        draw_list->ChannelsMerge();

        ImGui::EndTable();
    }
}

void OrderbookHistoryPanel::render_comparison_view() {
    if (current_snapshot_index_ < 0 || comparison_snapshot_index_ < 0 ||
        snapshots_.size() <= static_cast<size_t>(current_snapshot_index_) ||
        snapshots_.size() <= static_cast<size_t>(comparison_snapshot_index_)) {
        ImGui::Text("Invalid snapshot indices for comparison");
        return;
    }

    const auto& current_snapshot = snapshots_[current_snapshot_index_];
    const auto& comparison_snapshot = snapshots_[comparison_snapshot_index_];

    ImGui::Text("Comparing: Current (%d) vs Historical (%d)", 
                current_snapshot_index_, comparison_snapshot_index_);
    
    // Create side-by-side comparison
    if (ImGui::BeginTable("ComparisonTable", 2, ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Current State", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Historical State", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        ImGui::TableNextColumn();
        std::string current_title = "Current (" + std::to_string(current_snapshot_index_) + ")";
        render_orderbook_ladder(current_snapshot.bids, current_snapshot.asks, current_title);

        ImGui::TableNextColumn();
        std::string hist_title = "Historical (" + std::to_string(comparison_snapshot_index_) + ")";
        render_orderbook_ladder(comparison_snapshot.bids, comparison_snapshot.asks, hist_title);

        ImGui::EndTable();
    }
}

void OrderbookHistoryPanel::cleanupOldSnapshots() {
    while (snapshots_.size() > max_snapshots_) {
        snapshots_.pop_front();
        // Adjust indices if they become invalid
        if (current_snapshot_index_ >= static_cast<int>(snapshots_.size())) {
            current_snapshot_index_ = static_cast<int>(snapshots_.size()) - 1;
        }
        if (comparison_snapshot_index_ >= static_cast<int>(snapshots_.size())) {
            comparison_snapshot_index_ = static_cast<int>(snapshots_.size()) - 1;
        }
    }
}

double OrderbookHistoryPanel::calculateAverageOrderSize(const std::vector<RenderEngine::PriceLevel>& bids,
                                                      const std::vector<RenderEngine::PriceLevel>& asks) const {
    size_t total_levels = bids.size() + asks.size();
    if (total_levels == 0) return 0.0;

    double total_size = 0.0;
    for (const auto& level : bids) total_size += level.size;
    for (const auto& level : asks) total_size += level.size;

    return total_size / total_levels;
}

double OrderbookHistoryPanel::getMaxVolume(const std::vector<RenderEngine::PriceLevel>& bids,
                                          const std::vector<RenderEngine::PriceLevel>& asks) const {
    double max_vol = 1.0;
    for (const auto& level : bids) max_vol = std::max(max_vol, level.size);
    for (const auto& level : asks) max_vol = std::max(max_vol, level.size);
    return max_vol;
}

}  // namespace BTQuant