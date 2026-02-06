#ifndef UI_LAYOUT_MANAGER_HPP
#define UI_LAYOUT_MANAGER_HPP

#include <memory>
#include <vector>
#include <functional>
#include <map>
#include <string>

namespace ui {

// Forward declarations
class Widget;
class Container;
class Layout;

// Type aliases for convenience
using WidgetPtr = std::shared_ptr<Widget>;
using ContainerPtr = std::shared_ptr<Container>;
using LayoutPtr = std::shared_ptr<Layout>;

/**
 * @brief LayoutManager handles the arrangement and positioning of UI elements
 */
class LayoutManager {
public:
    // Virtual destructor for proper inheritance
    virtual ~LayoutManager() = default;

    // Pure virtual methods that derived classes must implement
    virtual void arrangeElements() = 0;
    virtual void updateLayout() = 0;
    
    // Common interface methods
    virtual void addElement(WidgetPtr element) = 0;
    virtual void removeElement(WidgetPtr element) = 0;
    virtual void clearElements() = 0;
    
    // Layout management
    virtual void setLayout(LayoutPtr layout) = 0;
    virtual LayoutPtr getLayout() const = 0;
    
protected:
    // Protected constructor to prevent direct instantiation
    LayoutManager() = default;
};

// More specific layout manager implementations could go here
// For example: HorizontalLayoutManager, VerticalLayoutManager, GridLayoutmanager, etc.

/**
 * @brief Base class for UI widgets
 */
class Widget {
public:
    virtual ~Widget() = default;
    
    virtual int getX() const = 0;
    virtual int getY() const = 0;
    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    
    virtual void setPosition(int x, int y) = 0;
    virtual void setSize(int width, int height) = 0;
    
protected:
    Widget() = default;
};

/**
 * @brief Base class for UI containers that hold other widgets
 */
class Container : public Widget {
public:
    virtual void addChild(WidgetPtr child) = 0;
    virtual void removeChild(WidgetPtr child) = 0;
    virtual std::vector<WidgetPtr> getChildren() const = 0;
    
protected:
    Container() = default;
};

/**
 * @brief Base class for layout algorithms
 */
class Layout {
public:
    virtual ~Layout() = default;
    
    virtual void applyLayout(ContainerPtr container) = 0;
    
protected:
    Layout() = default;
};

// Type alias for convenience
using LayoutManagerPtr = std::shared_ptr<LayoutManager>;

/**
 * @brief Factory function to create a default layout manager
 * @return Shared pointer to a layout manager instance
 */
inline LayoutManagerPtr createLayoutManager() {
    // This would typically return a concrete implementation
    // For now, returning nullptr as a placeholder until concrete implementations are added
    return nullptr;
}

} // namespace ui

#endif // UI_LAYOUT_MANAGER_HPP