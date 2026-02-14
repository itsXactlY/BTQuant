#include "../../include/components/multi_vwap_panel.hpp"
#include "imgui.h"

namespace BTQuant {

MultiVWAPPanel::MultiVWAPPanel(const PanelConfig& config)
    : PanelBase(config) {
    // Initialize with default VWAPs
    vwaps_ = {
        {"Session", ImVec4(0.0f, 0.8f, 0.8f, 1.0f), true, 3},
        {"Weekly", ImVec4(1.0f, 0.8f, 0.0f, 1.0f), true, 2},
        {"Monthly", ImVec4(0.8f, 0.0f, 0.8f, 1.0f), false, 2}
    };
}

void MultiVWAPPanel::initialize() {
    // Stub implementation
}

void MultiVWAPPanel::render() {
    begin_panel_window();
    
    if (!is_visible()) {
        end_panel_window();
        return;
    }
    
    render_vwap_list();
    ImGui::Separator();
    render_vwap_settings();
    ImGui::Separator();
    render_chart_overlay();
    
    end_panel_window();
}

void MultiVWAPPanel::render_vwap_list() {
    ImGui::Text("Active VWAPs (%zu)", vwaps_.size());
    
    if (ImGui::BeginChild("VWAPList", ImVec2(0, 150), true)) {
        for (size_t i = 0; i < vwaps_.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            
            // Visibility checkbox
            bool visible = vwaps_[i].show_sd_bands;
            ImGui::Checkbox("##visible", &vwaps_[i].show_sd_bands);
            ImGui::SameLine();
            
            // Color indicator
            ImGui::ColorButton("##color", vwaps_[i].color, ImGuiColorEditFlags_NoInputs, ImVec2(20, 20));
            ImGui::SameLine();
            
            // Period name
            ImGui::Text("%s", vwaps_[i].period.c_str());
            
            // SD bands toggle
            ImGui::SameLine(ImGui::GetWindowWidth() - 100);
            ImGui::Text("SD: %d", vwaps_[i].sd_levels);
            
            // Remove button
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) {
                remove_vwap(static_cast<int>(i));
            }
            
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
    
    // Add VWAP button
    if (ImGui::Button("Add VWAP")) {
        ImGui::OpenPopup("AddVWAPPopup");
    }
    
    if (ImGui::BeginPopup("AddVWAPPopup")) {
        const char* periods[] = {"Session", "Weekly", "Monthly", "Custom"};
        static int selected = 0;
        ImGui::Combo("Period", &selected, periods, IM_ARRAYSIZE(periods));
        
        if (ImGui::Button("Add")) {
            add_vwap(periods[selected], ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void MultiVWAPPanel::render_vwap_settings() {
    ImGui::Text("VWAP Settings");
    
    // Period selector
    const char* periods[] = {"Session", "Weekly", "Monthly", "Custom"};
    ImGui::Combo("Default Period", &selected_period_, periods, IM_ARRAYSIZE(periods));
    
    // SD levels
    static int sd_levels = 3;
    ImGui::SliderInt("SD Levels", &sd_levels, 1, 5);
    
    // Band opacity
    static float band_opacity = 0.3f;
    ImGui::SliderFloat("Band Opacity", &band_opacity, 0.1f, 0.5f);
}

void MultiVWAPPanel::render_chart_overlay() {
    ImGui::Text("Chart Overlay Preview");
    
    // Stub: Draw a placeholder chart area
    ImVec2 canvas_size = ImVec2(ImGui::GetContentRegionAvail().x, 150);
    if (ImGui::BeginChild("ChartPreview", canvas_size, true)) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_end = ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y);
        
        // Draw placeholder VWAP lines
        float mid_y = (canvas_pos.y + canvas_end.y) / 2;
        for (const auto& vwap : vwaps_) {
            if (vwap.show_sd_bands) {
                ImU32 color = ImGui::ColorConvertFloat4ToU32(vwap.color);
                draw_list->AddLine(ImVec2(canvas_pos.x, mid_y), 
                                   ImVec2(canvas_end.x, mid_y), 
                                   color, 2.0f);
                
                // Draw SD bands (stub)
                for (int sd = 1; sd <= vwap.sd_levels; ++sd) {
                    float offset = sd * 10.0f;
                    ImVec4 band_color = vwap.color;
                    band_color.w = 0.2f;
                    draw_list->AddLine(ImVec2(canvas_pos.x, mid_y - offset), 
                                       ImVec2(canvas_end.x, mid_y - offset), 
                                       ImGui::ColorConvertFloat4ToU32(band_color), 1.0f);
                    draw_list->AddLine(ImVec2(canvas_pos.x, mid_y + offset), 
                                       ImVec2(canvas_end.x, mid_y + offset), 
                                       ImGui::ColorConvertFloat4ToU32(band_color), 1.0f);
                }
            }
        }
    }
    ImGui::EndChild();
    
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "[Stub] Real VWAP data will be calculated from market data");
}

void MultiVWAPPanel::add_vwap(const std::string& period, const ImVec4& color) {
    vwaps_.push_back({period, color, true, 3});
}

void MultiVWAPPanel::remove_vwap(int index) {
    if (index >= 0 && index < static_cast<int>(vwaps_.size())) {
        vwaps_.erase(vwaps_.begin() + index);
    }
}

void MultiVWAPPanel::clear_vwaps() {
    vwaps_.clear();
}

}  // namespace BTQuant
