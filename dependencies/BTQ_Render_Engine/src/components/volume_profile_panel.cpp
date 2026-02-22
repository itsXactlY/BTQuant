/// @file volume_profile_panel.cpp
/// @brief Implements the volume profile panel showing CVD and volume distribution.

#include "components/volume_profile_panel.hpp"

#include <imgui.h>
#include <algorithm>
#include <cmath>

#include "analytics/cluster_engine.hpp"
#include "ChartMath.hpp"
#include "cache_manager.hpp"

namespace BTQuant {

// MMT Color constants
static constexpr ImU32 COLOR_NEON_MINT = IM_COL32(0x00, 0xE5, 0x66, 0xFF);  // #00E566 - Neon Mint
static constexpr ImU32 COLOR_ORANGE = IM_COL32(0xFF, 0x80, 0x00, 0xFF);     // #FF8000 - Orange
static constexpr ImU32 COLOR_VOID = IM_COL32(0x0B, 0x0E, 0x11, 0xFF);       // #0B0E11 - Deep Void
static constexpr ImU32 COLOR_CVD_LINE = IM_COL32(0x4A, 0x90, 0xFF, 0xFF);    // #4A90FF - CVD Line

VolumeProfilePanel::VolumeProfilePanel(const PanelConfig& config,
                                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                                     std::shared_ptr<ClusterEngine> cluster_engine,
                                     std::shared_ptr<CacheManager> cache_manager)
    : PanelBase(config)
    , processor_(std::move(processor))
    , cluster_engine_(std::move(cluster_engine))
    , cache_manager_(std::move(cache_manager)) {}

void VolumeProfilePanel::render_content() {
    if (!cluster_engine_) {
        ImGui::Text("No cluster engine");
        return;
    }

    // Display CVD
    int64_t cvd = cluster_engine_->get_cvd();
    ImGui::Text("Cumulative Volume Delta (CVD): %lld", cvd);

    // Show POC
    size_t poc_bin = cluster_engine_->get_poc_bin();
    double day_low = cluster_engine_->get_day_low();
    double tick_size = cluster_engine_->get_tick_size();
    double poc_price = day_low + poc_bin * tick_size;
    ImGui::Text("Point of Control (POC): %.2f", poc_price);

    ImGui::Separator();

    // Get cluster data
    const auto* bins = cluster_engine_->get_bins();
    size_t bin_count = cluster_engine_->get_bin_count();

    if (!bins || bin_count == 0) {
        ImGui::Text("No cluster data");
        return;
    }

    // Render volume profile
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    if (canvas_size.x <= 0 || canvas_size.y <= 0) return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

    // Find max volume for normalization
    float max_volume = 1.0f;
    for (size_t i = 0; i < bin_count; ++i) {
        float total_vol = bins[i].buy_vol + bins[i].sell_vol;
        if (total_vol > max_volume) max_volume = total_vol;
    }

    // Render volume profile bars
    float bar_width = canvas_size.x / static_cast<float>(bin_count);
    if (bar_width < 1.0f) bar_width = 1.0f; // Minimum 1px width

    for (size_t i = 0; i < bin_count; ++i) {
        float x = canvas_pos.x + static_cast<float>(i) * bar_width;
        
        // Calculate volumes
        float buy_vol = bins[i].buy_vol;
        float sell_vol = bins[i].sell_vol;
        float total_vol = buy_vol + sell_vol;

        if (total_vol > 0.0f) {
            // Calculate heights for buy/sell volumes
            float buy_height = (buy_vol / max_volume) * canvas_size.y * 0.5f;
            float sell_height = (sell_vol / max_volume) * canvas_size.y * 0.5f;

            // Draw buy volume (top half, green)
            if (buy_vol > 0.0f) {
                ImVec2 p1(x, canvas_pos.y + canvas_size.y * 0.5f - buy_height);
                ImVec2 p2(x + bar_width, canvas_pos.y + canvas_size.y * 0.5f);
                draw_list->AddRectFilled(p1, p2, COLOR_NEON_MINT);
            }

            // Draw sell volume (bottom half, orange)
            if (sell_vol > 0.0f) {
                ImVec2 p1(x, canvas_pos.y + canvas_size.y * 0.5f);
                ImVec2 p2(x + bar_width, canvas_pos.y + canvas_size.y * 0.5f + sell_height);
                draw_list->AddRectFilled(p1, p2, COLOR_ORANGE);
            }
        }
    }

    // Add dummy to consume space
    ImGui::Dummy(canvas_size);
}

void VolumeProfilePanel::set_cluster_engine(std::shared_ptr<ClusterEngine> engine) {
    cluster_engine_ = std::move(engine);
}

void VolumeProfilePanel::set_cache_manager(std::shared_ptr<CacheManager> cache) {
    cache_manager_ = std::move(cache);
}

}  // namespace BTQuant