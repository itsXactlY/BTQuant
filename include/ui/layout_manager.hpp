#ifndef BTQUANT_UI_LAYOUT_MANAGER_HPP
#define BTQUANT_UI_LAYOUT_MANAGER_HPP

/**
 * @brief UI namespace wrapper for layout management functionality
 * This header provides a unified interface to the UI layout management system
 */

// Include the main layout manager from the render engine
#include "ui/layout_manager.hpp"

namespace BTQuant {
/**
 * @brief UI namespace - Contains all user interface related functionality
 * This namespace wraps the UI components from the BTQ_Render_Engine
 */
namespace UI {

    // Expose the singleton instance getter
    inline BTQuant::UI::LayoutManager& getLayoutManager() {
        return BTQuant::UI::LayoutManager::getInstance();
    }

} // namespace UI
} // namespace BTQuant

#endif // BTQUANT_UI_LAYOUT_MANAGER_HPP