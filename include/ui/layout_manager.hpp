#ifndef BTQ_UI_LAYOUT_MANAGER_HPP
#define BTQ_UI_LAYOUT_MANAGER_HPP

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @namespace   BTQ::UI
/// @brief       Namespace wrapper for UI layout management components
/// @details     Contains classes and functions for managing UI layouts in the PubBTQuant application
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <vector>
#include <string>

namespace BTQ {
namespace UI {

/**
 * @brief Interface for layout management in the UI system
 */
class LayoutManager {
public:
    virtual ~LayoutManager() = default;

    /**
     * @brief Initialize the layout manager
     */
    virtual void initialize() = 0;

    /**
     * @brief Update the layout based on current window dimensions
     * @param width The width of the display area
     * @param height The height of the display area
     */
    virtual void updateLayout(int width, int height) = 0;

    /**
     * @brief Render the current layout
     */
    virtual void render() = 0;
};

/**
 * @brief Factory function to create a default layout manager
 * @return A shared pointer to a layout manager instance
 */
std::shared_ptr<LayoutManager> createLayoutManager();

} // namespace UI
} // namespace BTQ

#endif // BTQ_UI_LAYOUT_MANAGER_HPP