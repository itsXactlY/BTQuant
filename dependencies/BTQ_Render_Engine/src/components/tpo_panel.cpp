/// @file tpo_panel.cpp
/// @brief Implements the TPO panel for market profile visualization.

#include "components/tpo_panel.hpp"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "analytics/tpoengine.hpp"
#include "ChartMath.hpp"

namespace BTQuant {

// MMT Color constants
static constexpr ImU32 COLOR_NEON_MINT = IM_COL32(0x00, 0xE5, 0x66, 0xFF);  // #00E566 - Neon Mint
static constexpr ImU32 COLOR_ORANGE = IM_COL32(0xFF, 0x80, 0x00, 0xFF);     // #FF8000 - Orange
static constexpr ImU32 COLOR_VOID = IM_COL32(0x0B, 0x0E, 0x11, 0xFF);       // #0B0E11 - Deep Void
static constexpr ImU32 COLOR_HIGHLIGHT = IM_COL32(0x4A, 0x90, 0xFF, 0xFF);   // #4A90FF - Blue highlight
static constexpr ImU32 COLOR_SINGLE_PRINT_BG = IM_COL32(0x4A, 0x90, 0xFF, 0x3D); // Semi-transparent blue

TPOPanel::TPOPanel(const PanelConfig& config,
                   std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                   std::shared_ptr<TPOEngine> tpo_engine)
    : PanelBase(config)
    , processor_(std::move(processor))
    , tpo_engine_(std::move(tpo_engine)) {}

void TPOPanel::render_content() {
    if (!tpo_engine_) {
        ImGui::Text("No TPO engine");
        return;
    }

    // Get TPO profile
    auto profile = tpo_engine_->get_tpo_profile();
    if (profile.empty()) {
        ImGui::Text("No TPO data");
        return;
    }

    // Display value area
    size_t va_low, va_high;
    tpo_engine_->calculate_value_area(va_low, va_high);
    
    ImGui::Text("Value Area: %zu - %zu", va_low, va_high);

    ImGui::Separator();

    // Render TPO profile
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    if (canvas_size.x <= 0 || canvas_size.y <= 0) return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

    // Calculate cell dimensions
    float cell_height = canvas_size.y / static_cast<float>(profile.size());
    if (cell_height < 4.0f) {
        // Adaptive LOD: if cells are too small, only show colored rectangles
        render_condensed_view(draw_list, canvas_pos, canvas_size, profile, va_low, va_high);
    } else {
        // Detailed view with TPO characters
        render_detailed_view(draw_list, canvas_pos, canvas_size, profile, va_low, va_high);
    }

    // Add dummy to consume space
    ImGui::Dummy(canvas_size);
}

void TPOPanel::render_detailed_view(ImDrawList* draw_list, ImVec2 canvas_pos, ImVec2 canvas_size, 
                                   const std::vector<TPOBar>& profile, size_t va_low, size_t va_high) {
    float cell_height = canvas_size.y / static_cast<float>(profile.size());

    for (size_t i = 0; i < profile.size(); ++i) {
        const auto& bar = profile[i];
        
        float y = canvas_pos.y + static_cast<float>(i) * cell_height;
        
        // Draw background for value area
        if (i >= va_low && i <= va_high) {
            ImVec2 va_min(canvas_pos.x, y);
            ImVec2 va_max(canvas_pos.x + canvas_size.x, y + cell_height);
            draw_list->AddRectFilled(va_min, va_max, IM_COL32(0x15, 0x19, 0x1E, 0x33)); // Semi-transparent dark
        }
        
        // Draw single print background if applicable
        if (bar.is_single_print) {
            ImVec2 sp_min(canvas_pos.x, y);
            ImVec2 sp_max(canvas_pos.x + canvas_size.x, y + cell_height);
            draw_list->AddRectFilled(sp_min, sp_max, COLOR_SINGLE_PRINT_BG);
        }

        // Draw TPO characters
        float char_width = canvas_size.x / 16.0f; // 16 possible brackets (A-P)
        
        for (int bit = 0; bit < 16; ++bit) {
            if (bar.tpo_bits & (1 << bit)) {
                char c[2] = { bracket_to_char(static_cast<uint8_t>(bit)), '\0' };
                
                float x = canvas_pos.x + static_cast<float>(bit) * char_width;
                ImVec2 text_pos(x, y);
                
                // Use a brighter color for visibility
                ImU32 color = IM_COL32(0xB4, 0xC8, 0xFF, 0xFF); // Light blue-white
                draw_list->AddText(text_pos, color, c);
            }
        }
    }
}

void TPOPanel::render_condensed_view(ImDrawList* draw_list, ImVec2 canvas_pos, ImVec2 canvas_size, 
                                    const std::vector<TPOBar>& profile, size_t va_low, size_t va_high) {
    // In condensed view, just show colored bars representing TPO density
    float cell_height = 1.0f; // Minimum height
    
    for (size_t i = 0; i < profile.size(); ++i) {
        const auto& bar = profile[i];
        
        float y = canvas_pos.y + static_cast<float>(i) * cell_height;
        float intensity = static_cast<float>(bar.char_count) / 16.0f; // Normalize by max possible (16)
        
        // Draw background for value area
        if (i >= va_low && i <= va_high) {
            ImVec2 va_min(canvas_pos.x, y);
            ImVec2 va_max(canvas_pos.x + canvas_size.x, y + cell_height);
            draw_list->AddRectFilled(va_min, va_max, IM_COL32(0x15, 0x19, 0x1E, 0x66));
        }
        
        // Draw single print background if applicable
        if (bar.is_single_print) {
            ImVec2 sp_min(canvas_pos.x, y);
            ImVec2 sp_max(canvas_pos.x + canvas_size.x, y + cell_height);
            draw_list->AddRectFilled(sp_min, sp_max, COLOR_SINGLE_PRINT_BG);
        }
        
        // Color based on TPO activity
        if (intensity > 0.0f) {
            ImVec4 color;
            if (intensity > 0.7f) {
                color = ImVec4(1.0f, 0.5f, 0.0f, 0.8f); // Orange for high activity
            } else if (intensity > 0.3f) {
                color = ImVec4(0.0f, 0.9f, 0.4f, 0.6f); // Green for medium activity
            } else {
                color = ImVec4(0.5f, 0.5f, 0.5f, 0.4f); // Gray for low activity
            }
            
            ImVec2 rect_min(canvas_pos.x, y);
            ImVec2 rect_max(canvas_pos.x + canvas_size.x * intensity, y + cell_height);
            ImU32 col32 = ImGui::GetColorU32(color);
            draw_list->AddRectFilled(rect_min, rect_max, col32);
        }
    }
}

char TPOPanel::bracket_to_char(uint8_t bracket) {
    return bracket < 26 ? ('A' + bracket) : ('a' + bracket - 26);
}

void TPOPanel::set_tpo_engine(std::shared_ptr<TPOEngine> engine) {
    tpo_engine_ = std::move(engine);
}

}  // namespace BTQuant