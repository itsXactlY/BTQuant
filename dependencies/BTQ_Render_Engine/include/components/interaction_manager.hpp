#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "imgui.h"

namespace BTQuant {

struct KeyBinding {
  ImGuiKey key;
  bool ctrl = false;
  bool shift = false;
  bool alt = false;
  std::function<void()> callback;
  std::string description;
};

// Structure to hold mouse drag interaction data
struct MouseDragData {
  ImVec2 start_pos;
  ImVec2 current_pos;
  bool is_active = false;
  bool has_started = false;
  std::string interaction_type;  // e.g., "time_range_selection", "profile_creation"
};

class InteractionManager {
 public:
  static InteractionManager& getInstance() {
    static InteractionManager instance;
    return instance;
  }

  void update();

  void registerHotKey(ImGuiKey key, const std::function<void()>& callback, const std::string& desc,
                      bool ctrl = false, bool alt = false, bool shift = false);

  // Drag data handling (simplified)
  void setDragData(const std::string& type, void* data, size_t size);
  void* getDragData(const std::string& type);

  // Mouse drag interaction methods
  void startMouseDragInteraction(const std::string& type, const ImVec2& start_pos);
  bool isMouseDragActive() const;
  bool isMouseDragOfType(const std::string& type) const;
  MouseDragData getMouseDragData() const;
  void updateMouseDragPosition(const ImVec2& current_pos);
  void endMouseDragInteraction();

  // Time range selection methods
  void startTimeRangeSelection(const ImVec2& start_pos);
  bool isTimeRangeSelectionActive() const;
  std::pair<double, double> getTimeRangeSelection() const;  // Returns (start_time, end_time)
  void updateTimeRangeSelection(const ImVec2& current_pos);
  void endTimeRangeSelection();

 private:
  InteractionManager() = default;
  std::vector<KeyBinding> hotkeys_;

  struct DragPayload {
    std::string type;
    void* data = nullptr;
    size_t size = 0;
  } current_drag_;

  MouseDragData mouse_drag_data_;
  std::pair<double, double> time_range_selection_;  // (start_time, end_time)
};

}  // namespace BTQuant
