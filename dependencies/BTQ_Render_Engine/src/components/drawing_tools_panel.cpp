#include "../../include/components/drawing_tools_panel.hpp"
#include "imgui.h"
#include <algorithm>

namespace BTQuant {

DrawingToolsPanel::DrawingToolsPanel(const PanelConfig& config)
    : PanelBase(config) {
    // Initialize with default drawing tools
    tools_ = {
        {"Horizontal Line", ToolType::HORIZONTAL_LINE, true, ImVec4(1, 1, 1, 1), 1.0f, false},
        {"Vertical Line", ToolType::VERTICAL_LINE, true, ImVec4(1, 1, 1, 1), 1.0f, false},
        {"Trend Line", ToolType::TREND_LINE, true, ImVec4(0, 0.8f, 1, 1), 2.0f, false},
        {"Ray", ToolType::RAY, true, ImVec4(1, 0.8f, 0, 1), 1.0f, false},
        {"Rectangle", ToolType::RECTANGLE, true, ImVec4(0.5f, 0.5f, 0.5f, 1), 1.0f, false},
        {"Fibonacci Retracement", ToolType::FIBONACCI_RETRACEMENT, true, ImVec4(0.8f, 0.2f, 0.8f, 1), 1.0f, true},
        {"Fibonacci Extension", ToolType::FIBONACCI_EXTENSION, false, ImVec4(0.2f, 0.8f, 0.2f, 1), 1.0f, true},
        {"Text Label", ToolType::TEXT, true, ImVec4(1, 1, 1, 1), 1.0f, false},
        {"Arrow", ToolType::ARROW, true, ImVec4(1, 0.5f, 0, 1), 2.0f, false},
        {"Pitchfork", ToolType::PITCHFORK, false, ImVec4(0.8f, 0.8f, 0.2f, 1), 1.0f, false},
        {"Channel", ToolType::CHANNEL, false, ImVec4(0.2f, 0.6f, 0.8f, 1), 1.0f, false},
        {"Regression", ToolType::REGRESSION, false, ImVec4(0.6f, 0.2f, 0.2f, 1), 1.0f, false}
    };
}

void DrawingToolsPanel::initialize() {
    // Stub implementation
}

void DrawingToolsPanel::render() {
    begin_panel_window();
    
    if (!is_visible()) {
        end_panel_window();
        return;
    }
    
    render_tool_palette();
    ImGui::Separator();
    render_tool_settings();
    ImGui::Separator();
    render_favorites();
    ImGui::Separator();
    render_drawings_list();
    
    end_panel_window();
}

void DrawingToolsPanel::render_tool_palette() {
    ImGui::Text("Drawing Tools");
    
    // Tool categories
    if (ImGui::BeginTabBar("ToolCategories")) {
        if (ImGui::BeginTabItem("Lines")) {
            render_tool_buttons({ToolType::HORIZONTAL_LINE, ToolType::VERTICAL_LINE, 
                                ToolType::TREND_LINE, ToolType::RAY});
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("Shapes")) {
            render_tool_buttons({ToolType::RECTANGLE, ToolType::CHANNEL, ToolType::ARROW});
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("Fibonacci")) {
            render_tool_buttons({ToolType::FIBONACCI_RETRACEMENT, ToolType::FIBONACCI_EXTENSION});
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("Advanced")) {
            render_tool_buttons({ToolType::PITCHFORK, ToolType::REGRESSION, ToolType::TEXT});
            ImGui::EndTabItem();
        }
        
        ImGui::EndTabBar();
    }
}

void DrawingToolsPanel::render_tool_buttons(const std::vector<ToolType>& types) {
    int columns = 4;
    ImGui::Columns(columns, nullptr, false);
    
    for (const auto& type : types) {
        auto it = std::find_if(tools_.begin(), tools_.end(), 
                               [type](const DrawingTool& t) { return t.type == type; });
        
        if (it != tools_.end()) {
            ImGui::PushID(static_cast<int>(type));
            
            // Tool button with selection highlight
            bool is_selected = (selected_tool_ == type);
            if (is_selected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.4f, 0.6f, 0.8f, 1.0f));
            }
            
            // Button with tool name
            if (ImGui::Button(it->name.c_str(), ImVec2(80, 30))) {
                selected_tool_ = type;
            }
            
            if (is_selected) {
                ImGui::PopStyleColor();
            }
            
            // Favorite toggle on right-click
            if (ImGui::BeginPopupContextItem()) {
                if (it->is_favorite) {
                    if (ImGui::Selectable("Remove from Favorites")) {
                        it->is_favorite = false;
                    }
                } else {
                    if (ImGui::Selectable("Add to Favorites")) {
                        it->is_favorite = true;
                    }
                }
                ImGui::EndPopup();
            }
            
            ImGui::PopID();
            ImGui::NextColumn();
        }
    }
    
    ImGui::Columns(1);
}

void DrawingToolsPanel::render_tool_settings() {
    ImGui::Text("Tool Settings");
    
    auto it = std::find_if(tools_.begin(), tools_.end(), 
                           [this](const DrawingTool& t) { return t.type == selected_tool_; });
    
    if (it == tools_.end()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1), "Select a tool to edit settings");
        return;
    }
    
    // Color picker
    ImGui::ColorEdit4("Color", (float*)&it->color, 
                      ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaPreview);
    
    // Line width
    ImGui::SliderFloat("Line Width", &it->line_width, 0.5f, 5.0f);
    
    // Visibility
    ImGui::Checkbox("Visible", &it->visible);
    
    // Tool-specific settings
    if (it->type == ToolType::FIBONACCI_RETRACEMENT || 
        it->type == ToolType::FIBONACCI_EXTENSION) {
        ImGui::Separator();
        ImGui::Text("Fibonacci Levels:");
        
        static bool show_236 = true;
        static bool show_382 = true;
        static bool show_500 = true;
        static bool show_618 = true;
        static bool show_786 = true;
        
        ImGui::Checkbox("0.236", &show_236); ImGui::SameLine();
        ImGui::Checkbox("0.382", &show_382); ImGui::SameLine();
        ImGui::Checkbox("0.500", &show_500);
        ImGui::Checkbox("0.618", &show_618); ImGui::SameLine();
        ImGui::Checkbox("0.786", &show_786);
    }
    
    if (it->type == ToolType::TEXT) {
        static char text_content[256] = "";
        ImGui::InputText("Text", text_content, sizeof(text_content));
        static int font_size = 12;
        ImGui::SliderInt("Font Size", &font_size, 8, 24);
    }
}

void DrawingToolsPanel::render_favorites() {
    ImGui::Text("Favorites");
    
    // Count favorites
    int fav_count = 0;
    for (const auto& tool : tools_) {
        if (tool.is_favorite) fav_count++;
    }
    
    if (fav_count == 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1), 
                          "Right-click tools to add favorites");
        return;
    }
    
    // Display favorites as small buttons
    for (auto& tool : tools_) {
        if (tool.is_favorite) {
            ImGui::PushID(static_cast<int>(tool.type));
            
            // Color indicator
            ImGui::ColorButton("##color", tool.color, 
                              ImGuiColorEditFlags_NoInputs, ImVec2(20, 20));
            ImGui::SameLine();
            
            // Favorite button
            if (ImGui::SmallButton(tool.name.c_str())) {
                selected_tool_ = tool.type;
            }
            
            ImGui::SameLine();
            ImGui::PopID();
        }
    }
}

void DrawingToolsPanel::render_drawings_list() {
    ImGui::Text("Active Drawings (%zu)", drawings_.size());
    
    if (ImGui::BeginChild("DrawingsList", ImVec2(0, 100), true)) {
        for (size_t i = 0; i < drawings_.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            
            // Visibility toggle
            ImGui::Checkbox("##visible", &drawings_[i].visible);
            ImGui::SameLine();
            
            // Drawing info
            ImGui::Text("%s", drawings_[i].name.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1), 
                              "(%s)", drawings_[i].symbol.c_str());
            
            // Delete button
            ImGui::SameLine(ImGui::GetWindowWidth() - 50);
            if (ImGui::SmallButton("X")) {
                drawings_.erase(drawings_.begin() + static_cast<int>(i));
            }
            
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
    
    // Clear all button
    if (ImGui::Button("Clear All Drawings")) {
        ImGui::OpenPopup("ConfirmClear");
    }
    
    if (ImGui::BeginPopup("ConfirmClear")) {
        ImGui::Text("Clear all drawings?");
        if (ImGui::Button("Yes")) {
            drawings_.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("No")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    ImGui::SameLine();
    
    // Lock/Unlock drawings
    static bool drawings_locked = false;
    if (ImGui::Button(drawings_locked ? "Unlock Drawings" : "Lock Drawings")) {
        drawings_locked = !drawings_locked;
    }
    
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), 
                       "[Stub] Drawings will be rendered on chart panels");
}

void DrawingToolsPanel::add_drawing(const std::string& name, const std::string& symbol) {
    drawings_.push_back({name, symbol, true});
}

void DrawingToolsPanel::remove_drawing(int index) {
    if (index >= 0 && index < static_cast<int>(drawings_.size())) {
        drawings_.erase(drawings_.begin() + index);
    }
}

void DrawingToolsPanel::clear_drawings() {
    drawings_.clear();
}

}  // namespace BTQuant
