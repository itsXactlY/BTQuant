#include "../../include/ui/layout_manager.hpp"

namespace ui {

// Default implementations for virtual methods
void LayoutManager::arrangeElements() {
    // Default implementation - can be overridden by derived classes
}

void LayoutManager::updateLayout() {
    // Default implementation - can be overridden by derived classes
}

void LayoutManager::addElement(WidgetPtr element) {
    // Default implementation - can be overridden by derived classes
}

void LayoutManager::removeElement(WidgetPtr element) {
    // Default implementation - can be overridden by derived classes
}

void LayoutManager::clearElements() {
    // Default implementation - can be overridden by derived classes
}

void LayoutManager::setLayout(LayoutPtr layout) {
    currentLayout_ = layout;
}

LayoutPtr LayoutManager::getLayout() const {
    return currentLayout_;
}

} // namespace ui