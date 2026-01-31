#pragma once

#include "vulkan_base_types.hpp"
#include <chrono>
#include <glm/glm.hpp>
#include <imgui.h>
#include <string>

namespace BTQuant {

// Forward declarations
class VulkanCore;
namespace RenderEngine {
struct TradeData;
struct OrderbookData;
} // namespace RenderEngine

// ============================================================================
// Input \u0026 Event Types
// ============================================================================

enum class InputEventType {
  KeyDown,
  KeyUp,
  MouseMove,
  MouseButton,
  Scroll,
  Resize,
  Focus,
  Blur
};

enum class MouseButton {
  Left = 1,
  Middle = 2,
  Right = 3,
  ScrollUp = 4,
  ScrollDown = 5
};

struct InputEvent {
  InputEventType type;
  int keycode;
  int key;
  glm::vec2 mouse_pos;
  glm::vec2 position;
  glm::vec2 delta;
  bool pressed;
  float scroll_delta;
  MouseButton mouse_button;
  std::chrono::high_resolution_clock::time_point timestamp;
  uint32_t modifiers;
};

// ============================================================================
// UI Theme
// ============================================================================

struct DashboardTheme {
  ImVec4 accent_primary = {0.0f, 0.95f, 1.0f, 1.0f};
  ImVec4 accent_secondary = {1.0f, 0.0f, 0.3f, 1.0f};
  ImVec4 price_up = {0.0f, 0.95f, 1.0f, 1.0f};
  ImVec4 price_down = {1.0f, 0.0f, 0.3f, 1.0f};
  ImVec4 background = {0.04f, 0.04f, 0.04f, 1.0f};
  ImVec4 background_secondary = {0.1f, 0.1f, 0.1f, 1.0f};
  ImVec4 background_panel = {0.08f, 0.08f, 0.08f, 1.0f};
  ImVec4 background_primary = {0.04f, 0.04f, 0.04f, 1.0f};
  ImVec4 border_color = {0.2f, 0.2f, 0.2f, 1.0f};
  ImVec4 text_primary = {0.9f, 0.9f, 0.9f, 1.0f};
  ImVec4 text_muted = {0.5f, 0.5f, 0.5f, 1.0f};
  void *monospace_font = nullptr;
};

// ============================================================================
// UI Component Base
// ============================================================================

struct UIComponent {
  UIComponent(const glm::vec2 &p, const glm::vec2 &s)
      : position_(p), size_(s), visible_(true), dirty_frames_(3) {}
  virtual ~UIComponent() = default;
  virtual void initialize_vulkan_resources(VulkanCore *) {}
  virtual void update(float dt) = 0;
  virtual void render_gui() = 0;
  virtual void clear_data() {}
  void mark_dirty() { dirty_frames_ = 3; }
  bool is_dirty() const { return dirty_frames_ > 0; }
  virtual void handle_trade(const RenderEngine::TradeData &) {}
  virtual void handle_orderbook(const RenderEngine::OrderbookData &) {}
  virtual void handle_input(const InputEvent &) {}
  bool is_visible() const { return visible_; }
  glm::vec2 get_position() const { return position_; }
  glm::vec2 get_size() const { return size_; }
  glm::vec2 position_, size_;
  bool visible_;
  int dirty_frames_;
};

} // namespace BTQuant
