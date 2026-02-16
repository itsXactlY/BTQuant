#pragma once

#include "panel_base.hpp"
#include <string>
#include <vector>

namespace BTQuant {

/**
 * DrawingToolsPanel - Chart drawing tools management
 * 
 * Features (Stub):
 * - Tool selection (Line, Trend Line, Horizontal Line, Fibonacci, Rectangle, etc.)
 * - Tool properties (color, line width, style)
 * - Drawing history/management
 * - Snap to price/time options
 * - Favorite tools
 */
class DrawingToolsPanel : public PanelBase {
public:
    explicit DrawingToolsPanel(const PanelConfig& config);
    ~DrawingToolsPanel() override = default;

    void initialize() override;
    void render_content() override;

    // Tool management
    void select_tool(const std::string& tool_name);
    void clear_all_drawings();
    void undo_last();
    void redo_last();

private:
    enum class ToolType {
        NONE,
        HORIZONTAL_LINE,
        VERTICAL_LINE,
        TREND_LINE,
        RAY,
        FIBONACCI_RETRACEMENT,
        FIBONACCI_EXTENSION,
        RECTANGLE,
        CHANNEL,
        ARROW,
        PITCHFORK,
        REGRESSION,
        TEXT
    };
    
    struct DrawingTool {
        std::string name;
        ToolType type;
        bool is_favorite;
        ImVec4 color;
        float line_width;
        bool extended_line;  // For rays/trend lines
        bool visible = true;
    };
    
    struct ActiveDrawing {
        std::string name;
        std::string symbol;
        bool visible;
    };
    
    std::vector<DrawingTool> tools_;
    std::vector<ActiveDrawing> drawings_;
    ToolType selected_tool_ = ToolType::NONE;
    
    void render_tool_palette();
    void render_tool_buttons(const std::vector<ToolType>& types);
    void render_tool_settings();
    void render_favorites();
    void render_drawings_list();
    
    // Helper methods
    void add_drawing(const std::string& name, const std::string& symbol);
    void remove_drawing(int index);
    void clear_drawings();
};

}  // namespace BTQuant
