/**
 * BTQuant Advanced Dashboard Layout Manager Implementation
 * 
 * Professional layout management system for the six-panel financial dashboard
 * with responsive design, dynamic resizing, and professional spacing.
 * 
 * Layout Structure:
 * ┌─────────────────┬─────────────────┬─────────────────┐
 * │   Symbol Grid   │ Momentum Heatmap│  Connection     │
 * │                 │                 │  Status         │
 * ├─────────────────┼─────────────────┼─────────────────┤
 * │   Order Book    │   Price Chart   │   System Log    │
 * │                 │                 │                 │
 * └─────────────────┴─────────────────┴─────────────────┘
 * 
 * Features:
 * - Responsive six-panel layout system
 * - Professional spacing and alignment
 * - Dynamic resizing with constraints
 * - Dark theme color management
 * - High-DPI display support
 * - Animation and transition support
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <cmath>

namespace BTQuant {

// Layout panel identifiers
enum class PanelType {
    SymbolGrid = 0,
    MomentumHeatmap = 1,
    ConnectionStatus = 2,
    OrderBook = 3,
    PriceChart = 4,
    SystemLog = 5,
    Count = 6
};

// Panel configuration structure
struct PanelConfig {
    PanelType type;
    glm::vec2 position;
    glm::vec2 size;
    glm::vec2 min_size;
    glm::vec2 max_size;
    float weight_x;
    float weight_y;
    bool resizable;
    bool visible;
    uint32_t z_order;
    std::string title;
    glm::vec4 background_color;
    glm::vec4 border_color;
    float border_width;
};

class DashboardLayoutManager {
public:
    DashboardLayoutManager(uint32_t viewport_width, uint32_t viewport_height, const DashboardTheme& theme);
    ~DashboardLayoutManager();
    
    // Layout management
    void update_viewport_size(uint32_t width, uint32_t height);
    void calculate_layout();
    void animate_layout_changes(float delta_time);
    
    // Panel management
    PanelConfig& get_panel_config(PanelType type);
    const PanelConfig& get_panel_config(PanelType type) const;
    void set_panel_visibility(PanelType type, bool visible);
    void set_panel_size(PanelType type, const glm::vec2& size);
    void set_panel_position(PanelType type, const glm::vec2& position);
    
    // Layout queries
    glm::vec2 get_panel_position(PanelType type) const;
    glm::vec2 get_panel_size(PanelType type) const;
    bool is_panel_visible(PanelType type) const;
    
    // Responsive design
    void apply_responsive_layout();
    void set_layout_mode(LayoutMode mode);
    
    // Professional styling
    void apply_professional_spacing();
    void update_panel_colors(const DashboardTheme& theme);
    
    // Animation and transitions
    void enable_smooth_transitions(bool enable) { smooth_transitions_ = enable; }
    void set_transition_duration(float duration) { transition_duration_ = duration; }
    
    // Utility functions
    PanelType get_panel_at_position(const glm::vec2& position) const;
    bool is_position_in_panel(const glm::vec2& position, PanelType type) const;
    
private:
    // Layout configuration
    uint32_t viewport_width_;
    uint32_t viewport_height_;
    DashboardTheme theme_;
    
    // Panel configurations
    std::array<PanelConfig, static_cast<size_t>(PanelType::Count)> panels_;
    std::array<PanelConfig, static_cast<size_t>(PanelType::Count)> target_panels_; // For animations
    
    // Layout parameters
    float panel_spacing_;
    float border_width_;
    float title_bar_height_;
    glm::vec2 content_padding_;
    
    // Animation state
    bool smooth_transitions_;
    float transition_duration_;
    float animation_time_;
    
    // Layout modes
    enum class LayoutMode {
        Standard,
        Compact,
        Widescreen,
        Mobile
    } current_layout_mode_;
    
    // Private methods
    void initialize_default_layout();
    void calculate_standard_layout();
    void calculate_compact_layout();
    void calculate_widescreen_layout();
    void calculate_mobile_layout();
    
    void apply_layout_constraints();
    void update_panel_positions();
    void interpolate_panel_configs(float t);
    
    glm::vec2 calculate_available_space() const;
    void distribute_space_proportionally();
    void apply_minimum_size_constraints();
    void apply_maximum_size_constraints();
};

DashboardLayoutManager::DashboardLayoutManager(uint32_t viewport_width, uint32_t viewport_height, 
                                             const DashboardTheme& theme)
    : viewport_width_(viewport_width), viewport_height_(viewport_height), theme_(theme) {
    
    // Initialize layout parameters
    panel_spacing_ = 8.0f;
    border_width_ = 1.0f;
    title_bar_height_ = 24.0f;
    content_padding_ = glm::vec2(8.0f, 8.0f);
    
    // Animation settings
    smooth_transitions_ = true;
    transition_duration_ = 0.3f;
    animation_time_ = 0.0f;
    
    current_layout_mode_ = LayoutMode::Standard;
    
    initialize_default_layout();
    calculate_layout();
}

DashboardLayoutManager::~DashboardLayoutManager() {
    // Cleanup if needed
}

void DashboardLayoutManager::initialize_default_layout() {
    // Symbol Grid (Top-Left)
    panels_[static_cast<size_t>(PanelType::SymbolGrid)] = {
        .type = PanelType::SymbolGrid,
        .position = {0.0f, 0.0f},
        .size = {400.0f, 300.0f},
        .min_size = {300.0f, 200.0f},
        .max_size = {600.0f, 500.0f},
        .weight_x = 0.33f,
        .weight_y = 0.5f,
        .resizable = true,
        .visible = true,
        .z_order = 1,
        .title = "Symbol Grid",
        .background_color = theme_.background_panel,
        .border_color = theme_.border_color,
        .border_width = border_width_
    };
    
    // Momentum Heatmap (Top-Center)
    panels_[static_cast<size_t>(PanelType::MomentumHeatmap)] = {
        .type = PanelType::MomentumHeatmap,
        .position = {400.0f, 0.0f},
        .size = {400.0f, 300.0f},
        .min_size = {300.0f, 200.0f},
        .max_size = {800.0f, 500.0f},
        .weight_x = 0.34f,
        .weight_y = 0.5f,
        .resizable = true,
        .visible = true,
        .z_order = 1,
        .title = "Momentum Heatmap",
        .background_color = theme_.background_panel,
        .border_color = theme_.border_color,
        .border_width = border_width_
    };
    
    // Connection Status (Top-Right)
    panels_[static_cast<size_t>(PanelType::ConnectionStatus)] = {
        .type = PanelType::ConnectionStatus,
        .position = {800.0f, 0.0f},
        .size = {320.0f, 300.0f},
        .min_size = {250.0f, 150.0f},
        .max_size = {400.0f, 400.0f},
        .weight_x = 0.33f,
        .weight_y = 0.5f,
        .resizable = true,
        .visible = true,
        .z_order = 1,
        .title = "Connection Status",
        .background_color = theme_.background_panel,
        .border_color = theme_.border_color,
        .border_width = border_width_
    };
    
    // Order Book (Bottom-Left)
    panels_[static_cast<size_t>(PanelType::OrderBook)] = {
        .type = PanelType::OrderBook,
        .position = {0.0f, 300.0f},
        .size = {400.0f, 380.0f},
        .min_size = {300.0f, 250.0f},
        .max_size = {600.0f, 600.0f},
        .weight_x = 0.33f,
        .weight_y = 0.5f,
        .resizable = true,
        .visible = true,
        .z_order = 1,
        .title = "Order Book",
        .background_color = theme_.background_panel,
        .border_color = theme_.border_color,
        .border_width = border_width_
    };
    
    // Price Chart (Bottom-Center)
    panels_[static_cast<size_t>(PanelType::PriceChart)] = {
        .type = PanelType::PriceChart,
        .position = {400.0f, 300.0f},
        .size = {400.0f, 380.0f},
        .min_size = {350.0f, 250.0f},
        .max_size = {800.0f, 600.0f},
        .weight_x = 0.34f,
        .weight_y = 0.5f,
        .resizable = true,
        .visible = true,
        .z_order = 1,
        .title = "Price Chart",
        .background_color = theme_.background_panel,
        .border_color = theme_.border_color,
        .border_width = border_width_
    };
    
    // System Log (Bottom-Right)
    panels_[static_cast<size_t>(PanelType::SystemLog)] = {
        .type = PanelType::SystemLog,
        .position = {800.0f, 300.0f},
        .size = {320.0f, 380.0f},
        .min_size = {250.0f, 200.0f},
        .max_size = {500.0f, 600.0f},
        .weight_x = 0.33f,
        .weight_y = 0.5f,
        .resizable = true,
        .visible = true,
        .z_order = 1,
        .title = "System Log",
        .background_color = theme_.background_panel,
        .border_color = theme_.border_color,
        .border_width = border_width_
    };
    
    // Copy to target panels for animation
    target_panels_ = panels_;
}

void DashboardLayoutManager::update_viewport_size(uint32_t width, uint32_t height) {
    viewport_width_ = width;
    viewport_height_ = height;
    
    // Trigger layout recalculation
    calculate_layout();
}

void DashboardLayoutManager::calculate_layout() {
    switch (current_layout_mode_) {
        case LayoutMode::Standard:
            calculate_standard_layout();
            break;
        case LayoutMode::Compact:
            calculate_compact_layout();
            break;
        case LayoutMode::Widescreen:
            calculate_widescreen_layout();
            break;
        case LayoutMode::Mobile:
            calculate_mobile_layout();
            break;
    }
    
    apply_layout_constraints();
    apply_professional_spacing();
}

void DashboardLayoutManager::calculate_standard_layout() {
    glm::vec2 available_space = calculate_available_space();
    
    // Calculate grid dimensions (3x2 layout)
    float col_width = (available_space.x - 2 * panel_spacing_) / 3.0f;
    float row_height = (available_space.y - panel_spacing_) / 2.0f;
    
    // Top row
    target_panels_[static_cast<size_t>(PanelType::SymbolGrid)].position = {panel_spacing_, panel_spacing_};
    target_panels_[static_cast<size_t>(PanelType::SymbolGrid)].size = {col_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::MomentumHeatmap)].position = {
        panel_spacing_ + col_width + panel_spacing_, panel_spacing_
    };
    target_panels_[static_cast<size_t>(PanelType::MomentumHeatmap)].size = {col_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::ConnectionStatus)].position = {
        panel_spacing_ + 2 * (col_width + panel_spacing_), panel_spacing_
    };
    target_panels_[static_cast<size_t>(PanelType::ConnectionStatus)].size = {col_width, row_height};
    
    // Bottom row
    float bottom_y = panel_spacing_ + row_height + panel_spacing_;
    
    target_panels_[static_cast<size_t>(PanelType::OrderBook)].position = {panel_spacing_, bottom_y};
    target_panels_[static_cast<size_t>(PanelType::OrderBook)].size = {col_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::PriceChart)].position = {
        panel_spacing_ + col_width + panel_spacing_, bottom_y
    };
    target_panels_[static_cast<size_t>(PanelType::PriceChart)].size = {col_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::SystemLog)].position = {
        panel_spacing_ + 2 * (col_width + panel_spacing_), bottom_y
    };
    target_panels_[static_cast<size_t>(PanelType::SystemLog)].size = {col_width, row_height};
}

void DashboardLayoutManager::calculate_compact_layout() {
    // Compact layout for smaller screens
    glm::vec2 available_space = calculate_available_space();
    
    // 2x3 layout for compact mode
    float col_width = (available_space.x - panel_spacing_) / 2.0f;
    float row_height = (available_space.y - 2 * panel_spacing_) / 3.0f;
    
    // Left column
    target_panels_[static_cast<size_t>(PanelType::SymbolGrid)].position = {panel_spacing_, panel_spacing_};
    target_panels_[static_cast<size_t>(PanelType::SymbolGrid)].size = {col_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::OrderBook)].position = {
        panel_spacing_, panel_spacing_ + row_height + panel_spacing_
    };
    target_panels_[static_cast<size_t>(PanelType::OrderBook)].size = {col_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::SystemLog)].position = {
        panel_spacing_, panel_spacing_ + 2 * (row_height + panel_spacing_)
    };
    target_panels_[static_cast<size_t>(PanelType::SystemLog)].size = {col_width, row_height};
    
    // Right column
    target_panels_[static_cast<size_t>(PanelType::MomentumHeatmap)].position = {
        panel_spacing_ + col_width + panel_spacing_, panel_spacing_
    };
    target_panels_[static_cast<size_t>(PanelType::MomentumHeatmap)].size = {col_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::PriceChart)].position = {
        panel_spacing_ + col_width + panel_spacing_, panel_spacing_ + row_height + panel_spacing_
    };
    target_panels_[static_cast<size_t>(PanelType::PriceChart)].size = {col_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::ConnectionStatus)].position = {
        panel_spacing_ + col_width + panel_spacing_, panel_spacing_ + 2 * (row_height + panel_spacing_)
    };
    target_panels_[static_cast<size_t>(PanelType::ConnectionStatus)].size = {col_width, row_height};
}

void DashboardLayoutManager::calculate_widescreen_layout() {
    // Optimized for widescreen displays
    glm::vec2 available_space = calculate_available_space();
    
    // 4x2 layout with emphasis on charts
    float chart_width = available_space.x * 0.5f;
    float side_width = (available_space.x - chart_width - 3 * panel_spacing_) / 2.0f;
    float row_height = (available_space.y - panel_spacing_) / 2.0f;
    
    // Left column
    target_panels_[static_cast<size_t>(PanelType::SymbolGrid)].position = {panel_spacing_, panel_spacing_};
    target_panels_[static_cast<size_t>(PanelType::SymbolGrid)].size = {side_width, row_height};
    
    target_panels_[static_cast<size_t>(PanelType::OrderBook)].position = {
        panel_spacing_, panel_spacing_ + row_height + panel_spacing_
    };
    target_panels_[static_cast<size_t>(PanelType::OrderBook)].size = {side_width, row_height};
    
    // Center (charts)
    float chart_x = panel_spacing_ + side_width + panel_spacing_;
    target_panels_[static_cast<size_t>(PanelType::PriceChart)].position = {chart_x, panel_spacing_};
    target_panels_[static_cast<size_t>(PanelType::PriceChart)].size = {chart_width, available_space.y};
    
    // Right column
    float right_x = chart_x + chart_width + panel_spacing_;
    target_panels_[static_cast<size_t>(PanelType::MomentumHeatmap)].position = {right_x, panel_spacing_};
    target_panels_[static_cast<size_t>(PanelType::MomentumHeatmap)].size = {side_width, row_height * 0.6f};
    
    target_panels_[static_cast<size_t>(PanelType::ConnectionStatus)].position = {
        right_x, panel_spacing_ + row_height * 0.6f + panel_spacing_
    };
    target_panels_[static_cast<size_t>(PanelType::ConnectionStatus)].size = {side_width, row_height * 0.4f - panel_spacing_};
    
    target_panels_[static_cast<size_t>(PanelType::SystemLog)].position = {
        right_x, panel_spacing_ + row_height + panel_spacing_
    };
    target_panels_[static_cast<size_t>(PanelType::SystemLog)].size = {side_width, row_height};
}

void DashboardLayoutManager::calculate_mobile_layout() {
    // Single column layout for mobile
    glm::vec2 available_space = calculate_available_space();
    
    float panel_width = available_space.x;
    float panel_height = (available_space.y - 5 * panel_spacing_) / 6.0f;
    
    for (size_t i = 0; i < static_cast<size_t>(PanelType::Count); ++i) {
        target_panels_[i].position = {panel_spacing_, panel_spacing_ + i * (panel_height + panel_spacing_)};
        target_panels_[i].size = {panel_width, panel_height};
    }
}

void DashboardLayoutManager::apply_layout_constraints() {
    for (auto& panel : target_panels_) {
        // Apply minimum size constraints
        panel.size.x = std::max(panel.size.x, panel.min_size.x);
        panel.size.y = std::max(panel.size.y, panel.min_size.y);
        
        // Apply maximum size constraints
        panel.size.x = std::min(panel.size.x, panel.max_size.x);
        panel.size.y = std::min(panel.size.y, panel.max_size.y);
        
        // Ensure panels stay within viewport
        panel.position.x = std::max(0.0f, panel.position.x);
        panel.position.y = std::max(0.0f, panel.position.y);
        
        if (panel.position.x + panel.size.x > viewport_width_) {
            panel.position.x = viewport_width_ - panel.size.x;
        }
        if (panel.position.y + panel.size.y > viewport_height_) {
            panel.position.y = viewport_height_ - panel.size.y;
        }
    }
}

void DashboardLayoutManager::apply_professional_spacing() {
    // Ensure consistent spacing between panels
    for (auto& panel : target_panels_) {
        // Add title bar space
        panel.position.y += title_bar_height_;
        panel.size.y -= title_bar_height_;
        
        // Add content padding
        panel.size.x -= 2 * content_padding_.x;
        panel.size.y -= 2 * content_padding_.y;
    }
}

void DashboardLayoutManager::animate_layout_changes(float delta_time) {
    if (!smooth_transitions_) {
        panels_ = target_panels_;
        return;
    }
    
    animation_time_ += delta_time;
    float t = std::min(animation_time_ / transition_duration_, 1.0f);
    
    // Smooth easing function
    t = t * t * (3.0f - 2.0f * t);
    
    interpolate_panel_configs(t);
    
    if (t >= 1.0f) {
        animation_time_ = 0.0f;
    }
}

void DashboardLayoutManager::interpolate_panel_configs(float t) {
    for (size_t i = 0; i < static_cast<size_t>(PanelType::Count); ++i) {
        panels_[i].position = glm::mix(panels_[i].position, target_panels_[i].position, t);
        panels_[i].size = glm::mix(panels_[i].size, target_panels_[i].size, t);
    }
}

glm::vec2 DashboardLayoutManager::calculate_available_space() const {
    return glm::vec2(
        static_cast<float>(viewport_width_) - 2 * panel_spacing_,
        static_cast<float>(viewport_height_) - 2 * panel_spacing_
    );
}

PanelConfig& DashboardLayoutManager::get_panel_config(PanelType type) {
    return panels_[static_cast<size_t>(type)];
}

const PanelConfig& DashboardLayoutManager::get_panel_config(PanelType type) const {
    return panels_[static_cast<size_t>(type)];
}

glm::vec2 DashboardLayoutManager::get_panel_position(PanelType type) const {
    return panels_[static_cast<size_t>(type)].position;
}

glm::vec2 DashboardLayoutManager::get_panel_size(PanelType type) const {
    return panels_[static_cast<size_t>(type)].size;
}

bool DashboardLayoutManager::is_panel_visible(PanelType type) const {
    return panels_[static_cast<size_t>(type)].visible;
}

void DashboardLayoutManager::apply_responsive_layout() {
    float aspect_ratio = static_cast<float>(viewport_width_) / static_cast<float>(viewport_height_);
    
    if (viewport_width_ < 800) {
        set_layout_mode(LayoutMode::Mobile);
    } else if (viewport_width_ < 1200) {
        set_layout_mode(LayoutMode::Compact);
    } else if (aspect_ratio > 2.0f) {
        set_layout_mode(LayoutMode::Widescreen);
    } else {
        set_layout_mode(LayoutMode::Standard);
    }
}

void DashboardLayoutManager::set_layout_mode(LayoutMode mode) {
    if (current_layout_mode_ != mode) {
        current_layout_mode_ = mode;
        calculate_layout();
    }
}

PanelType DashboardLayoutManager::get_panel_at_position(const glm::vec2& position) const {
    for (size_t i = 0; i < static_cast<size_t>(PanelType::Count); ++i) {
        if (is_position_in_panel(position, static_cast<PanelType>(i))) {
            return static_cast<PanelType>(i);
        }
    }
    return PanelType::Count; // Invalid panel
}

bool DashboardLayoutManager::is_position_in_panel(const glm::vec2& position, PanelType type) const {
    const auto& panel = panels_[static_cast<size_t>(type)];
    
    return position.x >= panel.position.x &&
           position.x <= panel.position.x + panel.size.x &&
           position.y >= panel.position.y &&
           position.y <= panel.position.y + panel.size.y;
}

} // namespace BTQuant