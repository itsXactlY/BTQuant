#include "components/interaction_manager.hpp"

namespace BTQuant {

void InteractionManager::update() {}

void InteractionManager::registerHotKey(ImGuiKey key, const std::function<void()>&,
                                        const std::string&, bool, bool, bool) {}

void InteractionManager::setDragData(const std::string&, void*, size_t) {}
void* InteractionManager::getDragData(const std::string&) { return nullptr; }

void InteractionManager::startMouseDragInteraction(const std::string&, const ImVec2&) {}
bool InteractionManager::isMouseDragActive() const { return false; }
bool InteractionManager::isMouseDragOfType(const std::string&) const { return false; }
MouseDragData InteractionManager::getMouseDragData() const { return MouseDragData{}; }
void InteractionManager::updateMouseDragPosition(const ImVec2&) {}
void InteractionManager::endMouseDragInteraction() {}

void InteractionManager::startTimeRangeSelection(const ImVec2&) {}
bool InteractionManager::isTimeRangeSelectionActive() const { return false; }
std::pair<double, double> InteractionManager::getTimeRangeSelection() const { return {0.0, 0.0}; }
void InteractionManager::updateTimeRangeSelection(const ImVec2&) {}
void InteractionManager::endTimeRangeSelection() {}

}  // namespace BTQuant
