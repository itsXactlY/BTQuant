#include "../../include/components/interaction_manager.hpp"

#include <iostream>

namespace BTQuant {

void InteractionManager::update() {
  for (const auto& binding : hotkeys_) {
    // Check modifiers
    bool ctrl_pressed = ImGui::GetIO().KeyCtrl;
    bool shift_pressed = ImGui::GetIO().KeyShift;
    bool alt_pressed = ImGui::GetIO().KeyAlt;

    if (binding.ctrl != ctrl_pressed) continue;
    if (binding.shift != shift_pressed) continue;
    if (binding.alt != alt_pressed) continue;

    // Check key (IsKeyPressed handles repeat rate if needed, here we use
    // default false for no repeat)
    if (ImGui::IsKeyPressed(binding.key, false)) {
      if (binding.callback) {
        // std::cout << "[Interaction] Hotkey triggered: " <<
        // binding.description << std::endl;
        binding.callback();
      }
    }
  }

  // Update mouse drag state based on ImGui input
  if (mouse_drag_data_.has_started && !ImGui::IsMouseDown(0)) {
    // Mouse button released, end the drag interaction
    endMouseDragInteraction();
  }
}

void InteractionManager::registerHotKey(ImGuiKey key, const std::function<void()>& callback,
                                        const std::string& desc, bool ctrl, bool alt, bool shift) {
  KeyBinding binding;
  binding.key = key;
  binding.callback = callback;
  binding.description = desc;
  binding.ctrl = ctrl;
  binding.alt = alt;
  binding.shift = shift;
  hotkeys_.push_back(binding);
}

void InteractionManager::setDragData(const std::string& type, void* data, size_t size) {
  current_drag_.type = type;
  current_drag_.data = data;
  current_drag_.size = size;
}

void* InteractionManager::getDragData(const std::string& type) {
  if (current_drag_.type == type) {
    return current_drag_.data;
  }
  return nullptr;
}

// Mouse drag interaction methods
void InteractionManager::startMouseDragInteraction(const std::string& type, const ImVec2& start_pos) {
  mouse_drag_data_.interaction_type = type;
  mouse_drag_data_.start_pos = start_pos;
  mouse_drag_data_.current_pos = start_pos;
  mouse_drag_data_.is_active = true;
  mouse_drag_data_.has_started = true;
}

bool InteractionManager::isMouseDragActive() const {
  return mouse_drag_data_.is_active;
}

bool InteractionManager::isMouseDragOfType(const std::string& type) const {
  return mouse_drag_data_.is_active && mouse_drag_data_.interaction_type == type;
}

MouseDragData InteractionManager::getMouseDragData() const {
  return mouse_drag_data_;
}

void InteractionManager::updateMouseDragPosition(const ImVec2& current_pos) {
  if (mouse_drag_data_.is_active) {
    mouse_drag_data_.current_pos = current_pos;
  }
}

void InteractionManager::endMouseDragInteraction() {
  mouse_drag_data_.is_active = false;
  mouse_drag_data_.has_started = false;
  mouse_drag_data_.interaction_type.clear();
}

// Time range selection methods
void InteractionManager::startTimeRangeSelection(const ImVec2& start_pos) {
  startMouseDragInteraction("time_range_selection", start_pos);
  // Initialize time range with the same start and end time initially
  time_range_selection_.first = start_pos.x;  // Assuming x represents time
  time_range_selection_.second = start_pos.x;
}

bool InteractionManager::isTimeRangeSelectionActive() const {
  return isMouseDragOfType("time_range_selection");
}

std::pair<double, double> InteractionManager::getTimeRangeSelection() const {
  return time_range_selection_;
}

void InteractionManager::updateTimeRangeSelection(const ImVec2& current_pos) {
  if (isTimeRangeSelectionActive()) {
    // Update the time range based on drag direction
    double start_time = mouse_drag_data_.start_pos.x;
    double end_time = current_pos.x;

    // Ensure start_time is always less than or equal to end_time
    if (start_time <= end_time) {
      time_range_selection_.first = start_time;
      time_range_selection_.second = end_time;
    } else {
      time_range_selection_.first = end_time;
      time_range_selection_.second = start_time;
    }

    // Update the mouse drag position
    updateMouseDragPosition(current_pos);
  }
}

void InteractionManager::endTimeRangeSelection() {
  endMouseDragInteraction();
}

}  // namespace BTQuant
