#pragma once

#include <vector>
#include <string>
#include <memory>
#include <variant>
#include <glm/glm.hpp>

#include "imgui.h"
#include "implot.h"
#include "implot_internal.h"  // For ImPlotLimits

namespace BTQuant {

// Base structure for all drawing tools
struct DrawingTool {
    std::string id;
    ImVec4 color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    float thickness = 1.0f;
    bool visible = true;
    bool locked = false;
    
    DrawingTool(const std::string& tool_id) : id(tool_id) {}
    virtual ~DrawingTool() = default;
    virtual void render() = 0;
};

// Trend line tool - connects two points with a line
struct TrendLine : public DrawingTool {
    ImVec2 point1;  // Start point in plot coordinates
    ImVec2 point2;  // End point in plot coordinates
    
    TrendLine(const std::string& tool_id, ImVec2 p1, ImVec2 p2) 
        : DrawingTool(tool_id), point1(p1), point2(p2) {}
    
    void render() override {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        ImVec2 screen_p1 = ImPlot::PlotToPixels(point1.x, point1.y);
        ImVec2 screen_p2 = ImPlot::PlotToPixels(point2.x, point2.y);
        
        draw_list->AddLine(screen_p1, screen_p2, 
                          IM_COL32((int)(color.x * 255), (int)(color.y * 255), (int)(color.z * 255), (int)(color.w * 255)), 
                          thickness);
    }
};

// Horizontal line tool - extends infinitely horizontally at a specific price level
struct HorizontalLine : public DrawingTool {
    double y_value;  // Y coordinate (price level) in plot coordinates
    
    HorizontalLine(const std::string& tool_id, double y) 
        : DrawingTool(tool_id), y_value(y) {}
    
    void render() override {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        
        // Get plot bounds to draw line across the entire visible area
        ImPlotRect limits = ImPlot::GetPlotLimits();

        ImVec2 screen_start = ImPlot::PlotToPixels(limits.X.Min, y_value);
        ImVec2 screen_end = ImPlot::PlotToPixels(limits.X.Max, y_value);
        
        draw_list->AddLine(screen_start, screen_end, 
                          IM_COL32((int)(color.x * 255), (int)(color.y * 255), (int)(color.z * 255), (int)(color.w * 255)), 
                          thickness);
    }
};

// Fibonacci retracement tool - calculates and draws fibonacci levels between two points
struct FibonacciRetracement : public DrawingTool {
    ImVec2 point1;  // Start point in plot coordinates
    ImVec2 point2;  // End point in plot coordinates
    std::vector<double> ratios = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0};  // Standard fibonacci ratios
    
    FibonacciRetracement(const std::string& tool_id, ImVec2 p1, ImVec2 p2) 
        : DrawingTool(tool_id), point1(p1), point2(p2) {}
    
    void render() override {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        
        // Calculate the price range between the two points
        double price_range = point2.y - point1.y;
        
        // Draw each fibonacci level
        for (double ratio : ratios) {
            double level_price = point1.y + (price_range * ratio);
            
            // Get plot bounds to draw line across the entire visible area
            ImPlotRect limits = ImPlot::GetPlotLimits();

            ImVec2 screen_start = ImPlot::PlotToPixels(limits.X.Min, level_price);
            ImVec2 screen_end = ImPlot::PlotToPixels(limits.X.Max, level_price);
            
            // Calculate color with transparency based on ratio
            ImVec4 level_color = color;
            level_color.w *= 0.7f;  // Reduce alpha for better visibility
            
            draw_list->AddLine(screen_start, screen_end, 
                              IM_COL32((int)(level_color.x * 255), (int)(level_color.y * 255), (int)(level_color.z * 255), (int)(level_color.w * 255)), 
                              thickness);
            
            // Add label for the ratio
            ImVec2 label_pos = ImVec2(screen_start.x + 5, screen_start.y - 10);
            char label[16];
            snprintf(label, sizeof(label), "%.3f (%.1f%%)", ratio, ratio * 100);
            draw_list->AddText(label_pos, 
                              IM_COL32((int)(color.x * 255), (int)(color.y * 255), (int)(color.z * 255), (int)(color.w * 255)), 
                              label);
        }
    }
};

// Rectangle tool - draws a rectangle between two diagonal points
struct Rectangle : public DrawingTool {
    ImVec2 point1;  // First corner in plot coordinates
    ImVec2 point2;  // Opposite corner in plot coordinates
    
    Rectangle(const std::string& tool_id, ImVec2 p1, ImVec2 p2) 
        : DrawingTool(tool_id), point1(p1), point2(p2) {}
    
    void render() override {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        ImVec2 screen_p1 = ImPlot::PlotToPixels(point1.x, point1.y);
        ImVec2 screen_p2 = ImPlot::PlotToPixels(point2.x, point2.y);
        
        // Draw rectangle outline
        draw_list->AddRect(screen_p1, screen_p2, 
                          IM_COL32((int)(color.x * 255), (int)(color.y * 255), (int)(color.z * 255), (int)(color.w * 255)), 
                          0.0f, ImDrawFlags_None, thickness);
    }
};

// Text annotation tool - places text at a specific location
struct TextAnnotation : public DrawingTool {
    ImVec2 position;  // Position in plot coordinates
    std::string text;
    float font_size = 14.0f;
    
    TextAnnotation(const std::string& tool_id, ImVec2 pos, const std::string& txt) 
        : DrawingTool(tool_id), position(pos), text(txt) {}
    
    void render() override {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        ImVec2 screen_pos = ImPlot::PlotToPixels(position.x, position.y);
        
        // Draw text with background rectangle for better visibility
        ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
        ImVec2 bg_min = ImVec2(screen_pos.x, screen_pos.y - text_size.y);
        ImVec2 bg_max = ImVec2(screen_pos.x + text_size.x, screen_pos.y);
        
        // Background rectangle
        draw_list->AddRectFilled(bg_min, bg_max, IM_COL32(0, 0, 0, 200));
        
        // Text
        draw_list->AddText(screen_pos, 
                          IM_COL32((int)(color.x * 255), (int)(color.y * 255), (int)(color.z * 255), (int)(color.w * 255)), 
                          text.c_str());
    }
};

// Main drawing tools manager
class DrawingToolsManager {
public:
    DrawingToolsManager();
    ~DrawingToolsManager();
    
    // Add a new drawing tool
    void add_tool(std::unique_ptr<DrawingTool> tool);
    
    // Remove a drawing tool by ID
    void remove_tool(const std::string& id);
    
    // Clear all drawing tools
    void clear_all_tools();
    
    // Render all visible drawing tools
    void render_all();
    
    // Toggle visibility of a tool
    void toggle_visibility(const std::string& id);
    
    // Update tool properties
    void update_tool_color(const std::string& id, ImVec4 new_color);
    void update_tool_thickness(const std::string& id, float new_thickness);
    
    // UI controls for managing drawing tools
    void render_ui_controls();
    
    // Save/load drawing tools to/from persistent storage
    void save_to_file(const std::string& filename);
    void load_from_file(const std::string& filename);
    
private:
    std::vector<std::unique_ptr<DrawingTool>> tools_;
    std::string selected_tool_id_;
    bool show_ui_controls_ = true;
};

} // namespace BTQuant