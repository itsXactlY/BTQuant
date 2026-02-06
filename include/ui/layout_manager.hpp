#ifndef BTQ_UI_LAYOUT_MANAGER_HPP
#define BTQ_UI_LAYOUT_MANAGER_HPP

#include <memory>

namespace btq {
namespace ui {

/**
 * @brief Abstract base class for UI layout management
 * Provides an interface for arranging and managing UI elements
 */
class LayoutManager {
public:
    virtual ~LayoutManager() = default;
    
    /**
     * @brief Arranges child elements according to the layout strategy
     */
    virtual void arrangeElements() = 0;
    
    /**
     * @brief Updates the layout when elements change
     */
    virtual void updateLayout() = 0;
    
    /**
     * @brief Gets the current layout width
     */
    virtual int getWidth() const = 0;
    
    /**
     * @brief Gets the current layout height
     */
    virtual int getHeight() const = 0;
    
    /**
     * @brief Sets the dimensions for the layout
     */
    virtual void setDimensions(int width, int height) = 0;
};

} // namespace ui
} // namespace btq

#endif // BTQ_UI_LAYOUT_MANAGER_HPP