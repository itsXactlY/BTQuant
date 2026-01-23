#include "../../include/components/interaction_manager.hpp"
#include <iostream>

namespace BTQuant {

void InteractionManager::update() {
  for (const auto &binding : hotkeys_) {
    // Check modifiers
    bool ctrl_pressed = ImGui::GetIO().KeyCtrl;
    bool shift_pressed = ImGui::GetIO().KeyShift;
    bool alt_pressed = ImGui::GetIO().KeyAlt;

    if (binding.ctrl != ctrl_pressed)
      continue;
    if (binding.shift != shift_pressed)
      continue;
    if (binding.alt != alt_pressed)
      continue;

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
}

void InteractionManager::registerHotKey(ImGuiKey key,
                                        const std::function<void()> &callback,
                                        const std::string &desc, bool ctrl,
                                        bool alt, bool shift) {
  KeyBinding binding;
  binding.key = key;
  binding.callback = callback;
  binding.description = desc;
  binding.ctrl = ctrl;
  binding.alt = alt;
  binding.shift = shift;
  hotkeys_.push_back(binding);
}

void InteractionManager::setDragData(const std::string &type, void *data,
                                     size_t size) {
  current_drag_.type = type;
  current_drag_.data = data;
  current_drag_.size = size;
}

void *InteractionManager::getDragData(const std::string &type) {
  if (current_drag_.type == type) {
    return current_drag_.data;
  }
  return nullptr;
}

} // namespace BTQuant
