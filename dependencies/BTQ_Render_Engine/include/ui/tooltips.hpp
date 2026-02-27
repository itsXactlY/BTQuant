#pragma once

#include <string>
#include <unordered_map>

namespace BTQuant {
namespace UI {

/**
 * @brief Manages tooltips for UI controls throughout the application
 * 
 * The TooltipManager provides a centralized system for registering, storing,
 * and displaying tooltips for various UI controls. It allows for consistent
 * tooltip behavior across the entire application.
 */
class TooltipManager {
public:
    /**
     * @brief Constructor that initializes default tooltips
     */
    TooltipManager();

    /**
     * @brief Register a tooltip for a specific control
     * @param control_id Unique identifier for the control
     * @param tooltip_text The tooltip text to display
     */
    void register_tooltip(const std::string& control_id, const std::string& tooltip_text);

    /**
     * @brief Get the tooltip text for a specific control
     * @param control_id Unique identifier for the control
     * @return The tooltip text, or empty string if not found
     */
    std::string get_tooltip(const std::string& control_id) const;

    /**
     * @brief Show tooltip for the last ImGui item if it's hovered
     * @param control_id Unique identifier for the control
     */
    void show_tooltip(const std::string& control_id) const;

    /**
     * @brief Show tooltip for the last ImGui item if it's hovered (convenience method)
     * @param control_id Unique identifier for the control
     */
    void show_tooltip_for_last_item(const std::string& control_id) const;

    /**
     * @brief Show a simple tooltip for the last ImGui item
     * @param tooltip_text The tooltip text to display
     */
    void show_simple_tooltip(const std::string& tooltip_text) const;

private:
    /**
     * @brief Initialize default tooltips for common controls
     */
    void initialize_default_tooltips();

    std::unordered_map<std::string, std::string> tooltips_; ///< Map of control IDs to tooltip texts
};

/**
 * @brief Get the global tooltip manager instance
 * @return Reference to the global TooltipManager
 */
TooltipManager& get_global_tooltip_manager();

/**
 * @brief Show tooltip for a control using the global manager
 * @param control_id Unique identifier for the control
 */
void show_control_tooltip(const std::string& control_id);

/**
 * @brief Show a simple tooltip using the global manager
 * @param tooltip_text The tooltip text to display
 */
void show_simple_tooltip(const std::string& tooltip_text);

} // namespace UI
} // namespace BTQuant