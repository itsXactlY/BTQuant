#pragma once

#include "imgui.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace BTQuant {

struct KeyBinding {
  ImGuiKey key;
  bool ctrl = false;
  bool shift = false;
  bool alt = false;
  std::function<void()> callback;
  std::string description;
};

class InteractionManager {
public:
  static InteractionManager &getInstance() {
    static InteractionManager instance;
    return instance;
  }

  void update();

  void registerHotKey(ImGuiKey key, const std::function<void()> &callback,
                      const std::string &desc, bool ctrl = false,
                      bool alt = false, bool shift = false);

  // Drag data handling (simplified)
  void setDragData(const std::string &type, void *data, size_t size);
  void *getDragData(const std::string &type);

private:
  InteractionManager() = default;
  std::vector<KeyBinding> hotkeys_;

  struct DragPayload {
    std::string type;
    void *data = nullptr;
    size_t size = 0;
  } current_drag_;
};

} // namespace BTQuant
