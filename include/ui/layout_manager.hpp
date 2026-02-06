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
    // Singleton access
    static LayoutManager& getInstance() {
        static LayoutManager instance;
        return instance;
    }

    // Delete copy and move constructors and assignment operators
    LayoutManager(const LayoutManager&) = delete;
    LayoutManager& operator=(const LayoutManager&) = delete;
    LayoutManager(LayoutManager&&) = delete;
    LayoutManager& operator=(LayoutManager&&) = delete;

    // Virtual destructor for proper inheritance
    virtual ~LayoutManager() = default;

    // Methods that can be overridden by derived classes
    virtual void arrangeElements();
    virtual void updateLayout();

    // Common interface methods
    virtual void addElement(WidgetPtr element);
    virtual void removeElement(WidgetPtr element);
    virtual void clearElements();

    // Layout management
    virtual void setLayout(LayoutPtr layout);
    virtual LayoutPtr getLayout() const;

protected:
    // Protected constructor to prevent direct instantiation
    LayoutManager() = default;

private:
    LayoutPtr currentLayout_;  // Store the current layout
};

// More specific layout manager implementations could go here
// For example: HorizontalLayoutManager, VerticalLayoutManager, GridLayoutManager, etc.

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