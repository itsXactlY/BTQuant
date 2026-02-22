/// @file footprint_panel.cpp
/// @brief Implements the footprint panel showing volume clusters and imbalances.

#include "components/footprint_panel.hpp"

#include <imgui.h>
#include <algorithm>
#include <cmath>

#include "analytics/cluster_engine.hpp"
#include "ChartMath.hpp"

namespace BTQuant {

// MMT Color constants
static constexpr ImU32 COLOR_NEON_MINT = IM_COL32(0x00, 0xE5, 0x66, 0xFF);  // #00E566 - Neon Mint
static constexpr ImU32 COLOR_ORANGE = IM_COL32(0xFF, 0x80, 0x00, 0xFF);     // #FF8000 - Orange
static constexpr ImU32 COLOR_VOID = IM_COL32(0x0B, 0x0E, 0x11, 0xFF);       // #0B0E11 - Deep Void
static constexpr ImU32 COLOR_YELLOW = IM_COL32(0xFF, 0xC8, 0x00, 0xFF);     // Highlight color

FootprintPanel::FootprintPanel(const PanelConfig& config,
                             std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                             std::shared_ptr<ClusterEngine> cluster_engine)
    : PanelBase(config)
    , processor_(std::move(processor))
    , cluster_engine_(std::move(cluster_engine)) {}

void FootprintPanel::render_content() {
    if (!cluster_engine_) {
        ImGui::Text("No cluster engine");
        return;
    }

    // Get cluster data
    const auto* bins = cluster_engine_->get_bins();
    size_t bin_count = cluster_engine_->get_bin_count();
    double day_low = cluster_engine_->get_day_low();
    double tick_size = cluster_engine_->get_tick_size();

    if (!bins || bin_count == 0) {
        ImGui::Text("No cluster data");
        return;
    }

    // Display CVD
    int64_t cvd = cluster_engine_->get_cvd();
    ImGui::Text("CVD: %lld", cvd);
    
    // Show POC
    size_t poc_bin = cluster_engine_->get_poc_bin();
    double poc_price = day_low + poc_bin * tick_size;
    ImGui::Text("POC: %.2f", poc_price);

    ImGui::Separator();

    // Render footprint heatmap
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    if (canvas_size.x <= 0 || canvas_size.y <= 0) return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

    // Calculate cell dimensions
    float cell_width = canvas_size.x / static_cast<float>(bin_count);
    if (cell_width < 1.0f) cell_width = 1.0f; // Minimum 1px width

    // Find max volume for normalization
    float max_volume = 1.0f;
    for (size_t i = 0; i < bin_count; ++i) {
        float total_vol = bins[i].buy_vol + bins[i].sell_vol;
        if (total_vol > max_volume) max_volume = total_vol;
    }

    // Render each bin as a horizontal bar
    for (size_t i = 0; i < bin_count; ++i) {
        float x = canvas_pos.x + static_cast<float>(i) * cell_width;
        float height = canvas_size.y;

        // Calculate intensities
        float buy_intensity = bins[i].buy_vol / max_volume;
        float sell_intensity = bins[i].sell_vol / max_volume;

        // Draw buy volume (green)
        if (buy_intensity > 0.0f) {
            ImVec4 buy_color = ImVec4(0.0f, 0.9f, 0.4f, buy_intensity * 0.8f); // Neon mint with alpha
            ImU32 buy_col32 = ImGui::GetColorU32(buy_color);
            
            // Draw left half for buy volume
            float half_height = height * 0.5f;
            ImVec2 p1(x, canvas_pos.y);
            ImVec2 p2(x + cell_width, canvas_pos.y + half_height * buy_intensity);
            draw_list->AddRectFilled(p1, p2, buy_col32);
        }

        // Draw sell volume (orange)
        if (sell_intensity > 0.0f) {
            ImVec4 sell_color = ImVec4(1.0f, 0.5f, 0.0f, sell_intensity * 0.8f); // Orange with alpha
            ImU32 sell_col32 = ImGui::GetColorU32(sell_color);
            
            // Draw right half for sell volume
            float half_height = height * 0.5f;
            ImVec2 p1(x, canvas_pos.y + height - half_height * sell_intensity);
            ImVec2 p2(x + cell_width, canvas_pos.y + height);
            draw_list->AddRectFilled(p1, p2, sell_col32);
        }

        // Diagonal imbalance detection
        if (i + 1 < bin_count) {
            float bid_vol = bins[i].buy_vol;
            float ask_vol = bins[i + 1].sell_vol;
            if (ask_vol > 0 && bid_vol / ask_vol > 3.0f) {
                // Draw highlight box around this level
                ImVec2 rect_min(x, canvas_pos.y);
                ImVec2 rect_max(x + cell_width, canvas_pos.y + height);
                draw_list->AddRect(rect_min, rect_max, COLOR_YELLOW, 0.0f, 0, 2.0f);
            }
        }
    }

    // Add price labels on the right side
    ImGui::Dummy(canvas_size);
}

void FootprintPanel::set_cluster_engine(std::shared_ptr<ClusterEngine> engine) {
    cluster_engine_ = std::move(engine);
}

}  // namespace BTQuant