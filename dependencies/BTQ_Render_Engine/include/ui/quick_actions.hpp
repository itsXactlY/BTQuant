#pragma once

#include "../components/panel_base.hpp"
#include <vector>
#include <functional>
#include <string>
#include <unordered_map>

namespace BTQuant {

struct QuickAction {
    std::string name;
    std::string icon;  // Icon character or identifier
    std::function<void()> callback;
    std::string tooltip;
    bool enabled = true;
    
    QuickAction(const std::string& n, const std::string& i, std::function<void()> cb, const std::string& tip = "")
        : name(n), icon(i), callback(cb), tooltip(tip) {}
};

/**
 * @brief QuickActionsToolbar - Floating toolbar with commonly used actions
 *
 * Implements a floating toolbar that provides quick access to commonly used
 * actions across the trading terminal. The toolbar can be positioned anywhere
 * on screen and contains customizable quick action buttons.
 */
class QuickActionsToolbar {
public:
    QuickActionsToolbar();
    
    /**
     * @brief Renders the quick actions toolbar
     */
    void render();
    
    /**
     * @brief Adds a quick action to the toolbar
     * @param action The action to add
     */
    void add_action(const QuickAction& action);
    
    /**
     * @brief Removes an action by name
     * @param name Name of the action to remove
     */
    void remove_action(const std::string& name);
    
    /**
     * @brief Sets the position of the toolbar
     * @param pos Position in screen coordinates
     */
    void set_position(const ImVec2& pos);
    
    /**
     * @brief Gets the current position of the toolbar
     */
    ImVec2 get_position() const { return position_; }
    
    /**
     * @brief Sets the visibility of the toolbar
     * @param visible Whether the toolbar should be visible
     */
    void set_visible(bool visible) { visible_ = visible; }
    
    /**
     * @brief Checks if the toolbar is visible
     */
    bool is_visible() const { return visible_; }
    
    /**
     * @brief Toggles the visibility of the toolbar
     */
    void toggle_visibility() { visible_ = !visible_; }
    
    /**
     * @brief Sets whether the toolbar is dockable
     * @param dockable Whether the toolbar can be docked
     */
    void set_dockable(bool dockable) { dockable_ = dockable; }
    
    /**
     * @brief Sets the size of the toolbar buttons
     */
    void set_button_size(const ImVec2& size) { button_size_ = size; }
    
    /**
     * @brief Gets the default set of quick actions for the trading terminal
     */
    static std::vector<QuickAction> get_default_actions();

private:
    std::vector<QuickAction> actions_;
    ImVec2 position_{10.0f, 100.0f};  // Default position
    ImVec2 button_size_{30.0f, 30.0f};
    bool visible_ = true;
    bool dockable_ = true;
    bool draggable_ = true;
    bool initialized_ = false;
    
    /**
     * @brief Initializes the default set of quick actions
     */
    void initialize_default_actions();
    
    /**
     * @brief Renders the toolbar window with proper styling
     */
    void render_toolbar_window();
    
    /**
     * @brief Handles dragging of the toolbar if it's draggable
     */
    void handle_dragging();
};

} // namespace BTQuant