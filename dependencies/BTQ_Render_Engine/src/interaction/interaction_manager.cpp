/**
 * BTQuant Advanced Interaction Manager Implementation
 *
 * Comprehensive input handling system for professional trading dashboard
 * with multi-touch, gesture recognition, and context-aware interactions.
 */

#include "../include/interaction_manager.hpp"
#include "../../include/vulkan_dashboard_advanced.hpp"
#include <X11/keysym.h>
#include <algorithm>
#include <cmath>

namespace BTQuant {

// ============================================================================
// Context Menu Implementation
// ============================================================================

ContextMenu::ContextMenu(const glm::vec2 &position) : position_(position) {}

ContextMenu::~ContextMenu() {
  // Cleanup Vulkan resources
}

void ContextMenu::add_item(const ContextMenuItem &item) {
  items_.push_back(item);
  rebuild_geometry();
}

void ContextMenu::add_separator() {
  ContextMenuItem separator;
  separator.separator = true;
  items_.push_back(separator);
  rebuild_geometry();
}

void ContextMenu::show() {
  visible_ = true;
  selected_item_ = -1;
}

void ContextMenu::hide() {
  visible_ = false;
  selected_item_ = -1;
}

void ContextMenu::render(VkCommandBuffer /*cmd*/) {
  if (!visible_)
    return;

  // Render context menu background and items
  // Implementation would use Vulkan rendering pipeline
}

bool ContextMenu::handle_input(const InputEvent &event) {
  if (!visible_)
    return false;

  if (event.type == InputEventType::MouseMove) {
    // Update selected item based on mouse position
    float item_height = 25.0f;
    int item_index =
        static_cast<int>((event.position.y - position_.y) / item_height);

    if (item_index >= 0 && item_index < static_cast<int>(items_.size())) {
      if (!items_[item_index].separator) {
        selected_item_ = item_index;
      }
    } else {
      selected_item_ = -1;
    }
    return true;
  }

  if (event.type == InputEventType::MouseButton &&
      event.mouse_button == MouseButton::Left) {

    if (selected_item_ >= 0 &&
        selected_item_ < static_cast<int>(items_.size())) {
      execute_item(selected_item_);
    }
    hide();
    return true;
  }

  return false;
}

void ContextMenu::rebuild_geometry() {
  // Rebuild vertex buffer for menu rendering
}

void ContextMenu::execute_item(size_t index) {
  if (index < items_.size() && items_[index].action) {
    items_[index].action();
  }
}

// ============================================================================
// Tooltip Manager Implementation
// ============================================================================

TooltipManager::TooltipManager() {}

TooltipManager::~TooltipManager() {
  // Cleanup Vulkan resources
}

void TooltipManager::show_tooltip(const glm::vec2 &position,
                                  const TooltipData &data) {
  tooltip_position_ = position;
  current_tooltip_ = data;
  pending_ = true;
  show_timer_ = 0.0f;
  visible_ = false;
}

void TooltipManager::hide_tooltip() {
  visible_ = false;
  pending_ = false;
  show_timer_ = 0.0f;
}

void TooltipManager::update(float delta_time) {
  if (pending_) {
    show_timer_ += delta_time * 1000.0f; // Convert to milliseconds

    if (show_timer_ >= current_tooltip_.delay_ms) {
      visible_ = true;
      pending_ = false;
      rebuild_geometry();
    }
  }
}

void TooltipManager::render(VkCommandBuffer /*cmd*/) {
  if (!visible_)
    return;

  // Render tooltip background and text
  // Implementation would use Vulkan text rendering pipeline
}

void TooltipManager::rebuild_geometry() {
  // Rebuild vertex buffer for tooltip rendering
}

// ============================================================================
// Drag and Drop Manager Implementation
// ============================================================================

DragDropManager::DragDropManager() {}

DragDropManager::~DragDropManager() {
  // Cleanup Vulkan resources
}

void DragDropManager::start_drag(const DragDropData &data,
                                 const glm::vec2 &start_position) {
  drag_data_ = data;
  start_position_ = start_position;
  drag_position_ = start_position;
  dragging_ = true;
}

void DragDropManager::update_drag(const glm::vec2 &position) {
  if (!dragging_)
    return;

  drag_position_ = position;
}

void DragDropManager::end_drag(const glm::vec2 &position) {
  if (!dragging_)
    return;

  UIComponent *target = find_drop_target(position);
  if (target) {
    auto it = drop_targets_.find(target);
    if (it != drop_targets_.end()) {
      it->second(drag_data_, position);
    }
  }

  dragging_ = false;
}

void DragDropManager::cancel_drag() { dragging_ = false; }

void DragDropManager::register_drop_target(UIComponent *component,
                                           DropCallback callback) {
  drop_targets_[component] = callback;
}

void DragDropManager::unregister_drop_target(UIComponent *component) {
  drop_targets_.erase(component);
}

void DragDropManager::render(VkCommandBuffer /*cmd*/) {
  if (!dragging_)
    return;

  render_drag_visual();
}

void DragDropManager::render_drag_visual() {
  // Render drag visual feedback
}

UIComponent *DragDropManager::find_drop_target(const glm::vec2 & /*position*/) {
  // Find component at position that accepts drops
  return nullptr; // Placeholder
}

// ============================================================================
// Gesture Recognizer Implementation
// ============================================================================

GestureRecognizer::GestureRecognizer() {}

GestureRecognizer::~GestureRecognizer() {}

void GestureRecognizer::add_touch_point(const TouchPoint &point) {
  active_touches_[point.id] = point;
}

void GestureRecognizer::update_touch_point(const TouchPoint &point) {
  auto it = active_touches_.find(point.id);
  if (it != active_touches_.end()) {
    it->second = point;
  }
}

void GestureRecognizer::remove_touch_point(int touch_id) {
  active_touches_.erase(touch_id);
}

void GestureRecognizer::set_gesture_callback(GestureType type,
                                             GestureCallback callback) {
  gesture_callbacks_[type] = callback;
}

void GestureRecognizer::update(float /*delta_time*/) { detect_gestures(); }

void GestureRecognizer::detect_gestures() {
  if (active_touches_.size() >= 2) {
    detect_pinch();
    detect_rotation();
  }

  if (active_touches_.size() == 1) {
    detect_tap();
    detect_long_press();
    detect_swipe();
  }
}

void GestureRecognizer::detect_pinch() {
  if (active_touches_.size() < 2)
    return;

  float current_distance = calculate_distance();
  auto &state = gesture_states_[GestureType::Pinch];

  if (!state.active) {
    state.active = true;
    state.start_distance = current_distance;
    state.start_time = std::chrono::high_resolution_clock::now();
  }

  float scale = current_distance / state.start_distance;
  if (std::abs(scale - 1.0f) > 0.1f) { // Threshold check
    GestureEvent event;
    event.type = GestureType::Pinch;
    event.center = calculate_center();
    event.scale = scale;

    auto callback_it = gesture_callbacks_.find(GestureType::Pinch);
    if (callback_it != gesture_callbacks_.end()) {
      callback_it->second(event);
    }
  }
}

void GestureRecognizer::detect_rotation() {
  if (active_touches_.size() < 2)
    return;

  float current_angle = calculate_angle();
  auto &state = gesture_states_[GestureType::Rotate];

  if (!state.active) {
    state.active = true;
    state.start_angle = current_angle;
    state.start_time = std::chrono::high_resolution_clock::now();
  }

  float rotation = current_angle - state.start_angle;
  if (std::abs(rotation) > rotation_threshold_) {
    GestureEvent event;
    event.type = GestureType::Rotate;
    event.center = calculate_center();
    event.rotation = rotation;

    auto callback_it = gesture_callbacks_.find(GestureType::Rotate);
    if (callback_it != gesture_callbacks_.end()) {
      callback_it->second(event);
    }
  }
}

void GestureRecognizer::detect_swipe() {
  // Implementation for swipe detection
}

void GestureRecognizer::detect_tap() {
  // Implementation for tap detection
}

void GestureRecognizer::detect_long_press() {
  // Implementation for long press detection
}

glm::vec2 GestureRecognizer::calculate_center() {
  glm::vec2 center{0.0f};
  for (const auto &touch : active_touches_) {
    center += touch.second.position;
  }
  return center / static_cast<float>(active_touches_.size());
}

float GestureRecognizer::calculate_distance() {
  if (active_touches_.size() < 2)
    return 0.0f;

  auto it = active_touches_.begin();
  glm::vec2 pos1 = it->second.position;
  ++it;
  glm::vec2 pos2 = it->second.position;

  return glm::length(pos2 - pos1);
}

float GestureRecognizer::calculate_angle() {
  if (active_touches_.size() < 2)
    return 0.0f;

  auto it = active_touches_.begin();
  glm::vec2 pos1 = it->second.position;
  ++it;
  glm::vec2 pos2 = it->second.position;

  return std::atan2(pos2.y - pos1.y, pos2.x - pos1.x);
}

// ============================================================================
// Selection Manager Implementation
// ============================================================================

SelectionManager::SelectionManager() {}

SelectionManager::~SelectionManager() {
  // Cleanup Vulkan resources
}

void SelectionManager::select_component(UIComponent *component,
                                        bool multi_select) {
  if (!multi_select) {
    selected_components_.clear();
  }

  selected_components_.insert(component);
  rebuild_selection_geometry();
}

void SelectionManager::deselect_component(UIComponent *component) {
  selected_components_.erase(component);
  rebuild_selection_geometry();
}

void SelectionManager::clear_selection() {
  selected_components_.clear();
  rebuild_selection_geometry();
}

void SelectionManager::select_all() {
  // Implementation would select all selectable components
  rebuild_selection_geometry();
}

bool SelectionManager::is_selected(UIComponent *component) const {
  return selected_components_.find(component) != selected_components_.end();
}

const std::unordered_set<UIComponent *> &
SelectionManager::get_selected_components() const {
  return selected_components_;
}

void SelectionManager::render_selection_indicators(
    [[maybe_unused]] VkCommandBuffer cmd) {
  if (selected_components_.empty())
    return;

  // Render selection indicators around selected components
}

void SelectionManager::rebuild_selection_geometry() {
  // Rebuild vertex buffer for selection indicators
}

// ============================================================================
// Zoom and Pan Controller Implementation
// ============================================================================

ZoomPanController::ZoomPanController() {}

ZoomPanController::~ZoomPanController() {}

void ZoomPanController::set_bounds(const glm::vec2 &min_bounds,
                                   const glm::vec2 &max_bounds) {
  min_bounds_ = min_bounds;
  max_bounds_ = max_bounds;
  clamp_pan();
}

void ZoomPanController::set_zoom_limits(float min_zoom, float max_zoom) {
  min_zoom_ = min_zoom;
  max_zoom_ = max_zoom;
  clamp_zoom();
}

void ZoomPanController::pan(const glm::vec2 &delta) {
  pan_offset_ += delta / zoom_factor_;
  clamp_pan();
}

void ZoomPanController::zoom(float factor, const glm::vec2 &center) {
  glm::vec2 world_center = screen_to_world(center);
  zoom_factor_ *= factor;
  clamp_zoom();

  // Adjust pan to keep zoom center fixed
  glm::vec2 new_world_center = screen_to_world(center);
  pan_offset_ += world_center - new_world_center;
  clamp_pan();
}

void ZoomPanController::zoom_to_fit(const glm::vec2 &content_size) {
  // Calculate zoom to fit content
  glm::vec2 viewport_size = max_bounds_ - min_bounds_;
  float zoom_x = viewport_size.x / content_size.x;
  float zoom_y = viewport_size.y / content_size.y;

  zoom_factor_ = std::min(zoom_x, zoom_y);
  clamp_zoom();

  // Center content
  pan_offset_ = (content_size - viewport_size / zoom_factor_) * 0.5f;
  clamp_pan();
}

void ZoomPanController::reset() {
  pan_offset_ = glm::vec2{0.0f};
  zoom_factor_ = 1.0f;
}

glm::mat4 ZoomPanController::get_transform_matrix() const {
  glm::mat4 transform = glm::mat4(1.0f);
  transform =
      glm::scale(transform, glm::vec3(zoom_factor_, zoom_factor_, 1.0f));
  transform = glm::translate(transform, glm::vec3(-pan_offset_, 0.0f));
  return transform;
}

glm::vec2
ZoomPanController::screen_to_world(const glm::vec2 &screen_pos) const {
  return screen_pos / zoom_factor_ + pan_offset_;
}

glm::vec2 ZoomPanController::world_to_screen(const glm::vec2 &world_pos) const {
  return (world_pos - pan_offset_) * zoom_factor_;
}

void ZoomPanController::clamp_pan() {
  pan_offset_.x = std::clamp(pan_offset_.x, min_bounds_.x, max_bounds_.x);
  pan_offset_.y = std::clamp(pan_offset_.y, min_bounds_.y, max_bounds_.y);
}

void ZoomPanController::clamp_zoom() {
  zoom_factor_ = std::clamp(zoom_factor_, min_zoom_, max_zoom_);
}

// ============================================================================
// Hotkey Manager Implementation
// ============================================================================

HotkeyManager::HotkeyManager() {}

HotkeyManager::~HotkeyManager() {}

void HotkeyManager::register_hotkey(const std::string &name, int key_code,
                                    uint32_t modifiers,
                                    std::function<void()> action,
                                    const std::string &description) {
  HotkeyBinding binding;
  binding.key_code = key_code;
  binding.modifiers = modifiers;
  binding.action = action;
  binding.description = description;
  binding.enabled = true;

  hotkeys_[name] = binding;
}

void HotkeyManager::unregister_hotkey(const std::string &name) {
  hotkeys_.erase(name);
}

void HotkeyManager::enable_hotkey(const std::string &name, bool enabled) {
  auto it = hotkeys_.find(name);
  if (it != hotkeys_.end()) {
    it->second.enabled = enabled;
  }
}

bool HotkeyManager::handle_key_event(const InputEvent &event) {
  if (event.type != InputEventType::KeyDown)
    return false;

  // Update pressed keys state
  if (event.type == InputEventType::KeyDown) {
    pressed_keys_.insert(event.key);
  } else {
    pressed_keys_.erase(event.key);
  }

  // Check for hotkey matches
  for (const auto &hotkey_pair : hotkeys_) {
    const auto &binding = hotkey_pair.second;

    if (!binding.enabled)
      continue;

    if (static_cast<uint32_t>(binding.key_code) ==
            static_cast<uint32_t>(event.key) &&
        binding.modifiers == event.modifiers) {

      if (binding.action) {
        binding.action();
      }
      return true;
    }
  }

  return false;
}

std::vector<std::pair<std::string, HotkeyBinding>>
HotkeyManager::get_all_hotkeys() const {
  std::vector<std::pair<std::string, HotkeyBinding>> result;
  for (const auto &pair : hotkeys_) {
    result.push_back(pair);
  }
  return result;
}

uint64_t HotkeyManager::make_key_hash(int key_code, uint32_t modifiers) const {
  return (static_cast<uint64_t>(key_code) << 32) | modifiers;
}

// ============================================================================
// Main Interaction Manager Implementation
// ============================================================================

InteractionManager::InteractionManager(VulkanDashboard *dashboard)
    : dashboard_(dashboard) {}

InteractionManager::~InteractionManager() { cleanup(); }

void InteractionManager::initialize(Display *display, Window window) {
  display_ = display;
  window_ = window;

  setup_xinput2();
  setup_default_hotkeys();
}

void InteractionManager::cleanup() {
  cleanup_xinput2();
  context_menus_.clear();
}

void InteractionManager::process_x11_event(XEvent *event) {
  switch (event->type) {
  case MotionNotify:
    process_motion_event(event);
    break;

  case ButtonPress:
  case ButtonRelease:
    process_mouse_event(event);
    break;

  case XI_KeyPress:
  case XI_KeyRelease:
    process_keyboard_event(event);
    break;

  case GenericEvent:
    if (event->xcookie.extension == xi_opcode_) {
      if (XGetEventData(display_, &event->xcookie)) {
        process_touch_event(event);
        XFreeEventData(display_, &event->xcookie);
      }
    }
    break;
  }
}

void InteractionManager::update(float delta_time) {
  tooltip_manager_.update(delta_time);
  gesture_recognizer_.update(delta_time);

  // Update context menus
  for (auto const &menu : context_menus_) {
    (void)menu;
  }
}

void InteractionManager::render(VkCommandBuffer cmd) {
  // Render interaction elements
  tooltip_manager_.render(cmd);
  drag_drop_manager_.render(cmd);
  selection_manager_.render_selection_indicators(cmd);

  // Render context menus
  for (auto &menu : context_menus_) {
    menu->render(cmd);
  }
}

void InteractionManager::register_component(UIComponent *component) {
  auto it = std::find(registered_components_.begin(),
                      registered_components_.end(), component);
  if (it == registered_components_.end()) {
    registered_components_.push_back(component);
  }
}

void InteractionManager::unregister_component(UIComponent *component) {
  auto it = std::find(registered_components_.begin(),
                      registered_components_.end(), component);
  if (it != registered_components_.end()) {
    registered_components_.erase(it);
  }

  // Remove from selection if selected
  selection_manager_.deselect_component(component);

  // Remove from drag/drop targets
  drag_drop_manager_.unregister_drop_target(component);

  // Clear focus if this component was focused
  if (focused_component_ == component) {
    focused_component_ = nullptr;
  }

  if (hovered_component_ == component) {
    hovered_component_ = nullptr;
  }
}

ContextMenu *
InteractionManager::create_context_menu(const glm::vec2 &position) {
  auto menu = std::make_unique<ContextMenu>(position);
  ContextMenu *menu_ptr = menu.get();
  context_menus_.push_back(std::move(menu));
  return menu_ptr;
}

bool InteractionManager::is_mouse_button_pressed(MouseButton button) const {
  return pressed_mouse_buttons_.find(static_cast<int>(button)) !=
         pressed_mouse_buttons_.end();
}

bool InteractionManager::is_key_pressed(int key_code) const {
  return pressed_keys_.find(key_code) != pressed_keys_.end();
}

void InteractionManager::set_focused_component(UIComponent *component) {
  focused_component_ = component;
}

void InteractionManager::process_mouse_event(XEvent *event) {
  XButtonEvent *button_event = &event->xbutton;

  mouse_position_ = glm::vec2(button_event->x, button_event->y);

  InputEvent input_event = create_input_event(InputEventType::MouseButton);
  input_event.position = mouse_position_;
  input_event.mouse_button = static_cast<MouseButton>(button_event->button);

  if (event->type == ButtonPress) {
    pressed_mouse_buttons_.insert(button_event->button);

    // Handle context menu
    if (button_event->button == static_cast<int>(MouseButton::Right)) {
      UIComponent *component = find_component_at_position(mouse_position_);
      if (component) {
        // Create context menu for component
        ContextMenu *menu = create_context_menu(mouse_position_);

        // Add common context menu items
        ContextMenuItem copy_item;
        copy_item.label = "Copy";
        copy_item.action = []() { /* Copy action */ };
        menu->add_item(copy_item);

        ContextMenuItem properties_item;
        properties_item.label = "Properties";
        properties_item.action = []() { /* Properties action */ };
        menu->add_item(properties_item);

        menu->show();
      }
    }

    // Handle selection
    if (button_event->button == static_cast<int>(MouseButton::Left)) {
      UIComponent *component = find_component_at_position(mouse_position_);
      if (component) {
        bool multi_select =
            (active_modifiers_ & static_cast<uint32_t>(KeyModifier::Ctrl)) != 0;
        selection_manager_.select_component(component, multi_select);
        set_focused_component(component);
      } else {
        if (!(active_modifiers_ & static_cast<uint32_t>(KeyModifier::Ctrl))) {
          selection_manager_.clear_selection();
        }
        set_focused_component(nullptr);
      }
    }
  } else {
    pressed_mouse_buttons_.erase(button_event->button);

    // Handle drag end
    if (drag_drop_manager_.is_dragging()) {
      drag_drop_manager_.end_drag(mouse_position_);
    }
  }

  // Dispatch to components
  UIComponent *target_component = find_component_at_position(mouse_position_);
  if (target_component) {
    dispatch_event_to_component(target_component, input_event);
  }
}

void InteractionManager::process_keyboard_event(XEvent *event) {
  XKeyEvent *key_event = &event->xkey;

  // Update modifiers
  active_modifiers_ = x11_modifiers_to_internal(key_event->state);

  KeySym keysym = XKeycodeToKeysym(display_, key_event->keycode, 0);

  InputEvent input_event =
      create_input_event(event->type == XI_KeyPress ? InputEventType::KeyDown
                                                    : InputEventType::KeyUp);
  input_event.key = keysym;
  input_event.modifiers = active_modifiers_;

  // Handle hotkeys
  if (hotkey_manager_.handle_key_event(input_event)) {
    return; // Hotkey handled the event
  }

  // Dispatch to focused component
  if (focused_component_) {
    dispatch_event_to_component(focused_component_, input_event);
  }
}

void InteractionManager::process_touch_event(XEvent *event) {
  // Handle XI2 touch events for multi-touch support
  XIDeviceEvent *touch_event = (XIDeviceEvent *)event->xcookie.data;

  TouchPoint touch;
  touch.id = touch_event->detail;
  touch.position = glm::vec2(touch_event->event_x, touch_event->event_y);
  touch.timestamp = std::chrono::high_resolution_clock::now();

  switch (touch_event->evtype) {
  case XI_TouchBegin:
    gesture_recognizer_.add_touch_point(touch);
    break;

  case XI_TouchUpdate:
    gesture_recognizer_.update_touch_point(touch);
    break;

  case XI_TouchEnd:
    gesture_recognizer_.remove_touch_point(touch.id);
    break;
  }
}

void InteractionManager::process_motion_event(XEvent *event) {
  XMotionEvent *motion_event = &event->xmotion;

  glm::vec2 new_position(motion_event->x, motion_event->y);
  glm::vec2 delta = new_position - mouse_position_;
  mouse_position_ = new_position;

  // Update hover state
  update_hover_state(mouse_position_);

  // Handle drag update
  if (drag_drop_manager_.is_dragging()) {
    drag_drop_manager_.update_drag(mouse_position_);
  }

  InputEvent input_event = create_input_event(InputEventType::MouseMove);
  input_event.position = mouse_position_;
  input_event.delta = delta;

  // Dispatch to hovered component
  if (hovered_component_) {
    dispatch_event_to_component(hovered_component_, input_event);
  }
}

UIComponent *
InteractionManager::find_component_at_position(const glm::vec2 &position) {
  // Find the topmost component at the given position
  for (auto it = registered_components_.rbegin();
       it != registered_components_.rend(); ++it) {
    UIComponent *component = *it;
    if (!component->is_visible())
      continue;

    glm::vec2 comp_pos = component->get_position();
    glm::vec2 comp_size = component->get_size();

    if (position.x >= comp_pos.x && position.x <= comp_pos.x + comp_size.x &&
        position.y >= comp_pos.y && position.y <= comp_pos.y + comp_size.y) {
      return component;
    }
  }

  return nullptr;
}

void InteractionManager::update_hover_state(const glm::vec2 &position) {
  UIComponent *new_hovered = find_component_at_position(position);

  if (new_hovered != hovered_component_) {
    // Hide tooltip when leaving component
    if (hovered_component_) {
      tooltip_manager_.hide_tooltip();
    }

    hovered_component_ = new_hovered;

    // Show tooltip for new component
    if (hovered_component_) {
      TooltipManager::TooltipData tooltip;
      tooltip.title = "Component";
      tooltip.content = "Interactive UI component";
      tooltip_manager_.show_tooltip(position, tooltip);
    }
  }
}

void InteractionManager::dispatch_event_to_component(UIComponent *component,
                                                     const InputEvent &event) {
  if (component && component->is_visible()) {
    component->handle_input(event);
  }
}

void InteractionManager::setup_xinput2() {
  // Setup XI2 for multi-touch support
  int major = 2, minor = 2;
  if (XIQueryVersion(display_, &major, &minor) != X11_Success) {
    std::cerr << "XI2 not available\n";
    return;
  }

  // Get XI2 opcode
  int first_event, first_error;
  if (!XQueryExtension(display_, "XInputExtension", &xi_opcode_, &first_event,
                       &first_error)) {
    std::cerr << "XInput extension not available\n";
    return;
  }

  // Select touch events
  XIEventMask mask;
  mask.deviceid = XIAllMasterDevices;
  mask.mask_len = XIMaskLen(XI_LASTEVENT);
  mask.mask = (unsigned char *)calloc(mask.mask_len, sizeof(char));

  XISetMask(mask.mask, XI_TouchBegin);
  XISetMask(mask.mask, XI_TouchUpdate);
  XISetMask(mask.mask, XI_TouchEnd);

  XISelectEvents(display_, window_, &mask, 1);

  free(mask.mask);
  multi_touch_enabled_ = true;
}

void InteractionManager::cleanup_xinput2() {
  if (multi_touch_enabled_) {
    // Cleanup XI2 resources
    multi_touch_enabled_ = false;
  }
}

uint32_t InteractionManager::x11_modifiers_to_internal(unsigned int x11_mods) {
  uint32_t internal_mods = 0;

  if (x11_mods & ShiftMask)
    internal_mods |= static_cast<uint32_t>(KeyModifier::Shift);
  if (x11_mods & ControlMask)
    internal_mods |= static_cast<uint32_t>(KeyModifier::Ctrl);
  if (x11_mods & Mod1Mask)
    internal_mods |= static_cast<uint32_t>(KeyModifier::Alt);
  if (x11_mods & Mod4Mask)
    internal_mods |= static_cast<uint32_t>(KeyModifier::Super);

  return internal_mods;
}

InputEvent InteractionManager::create_input_event(InputEventType type) {
  InputEvent event;
  event.type = type;
  event.timestamp = std::chrono::high_resolution_clock::now();
  event.modifiers = active_modifiers_;
  return event;
}

void InteractionManager::setup_default_hotkeys() {
  // Setup common trading dashboard hotkeys

  // File operations
  hotkey_manager_.register_hotkey(
      "save_layout", XK_s, static_cast<uint32_t>(KeyModifier::Ctrl),
      []() { /* Save layout */ }, "Save current layout");

  hotkey_manager_.register_hotkey(
      "load_layout", XK_o, static_cast<uint32_t>(KeyModifier::Ctrl),
      []() { /* Load layout */ }, "Load layout");

  // View operations
  hotkey_manager_.register_hotkey(
      "zoom_in", XK_plus, static_cast<uint32_t>(KeyModifier::Ctrl),
      [this]() { zoom_pan_controller_.zoom(1.2f, mouse_position_); },
      "Zoom in");

  hotkey_manager_.register_hotkey(
      "zoom_out", XK_minus, static_cast<uint32_t>(KeyModifier::Ctrl),
      [this]() { zoom_pan_controller_.zoom(0.8f, mouse_position_); },
      "Zoom out");

  hotkey_manager_.register_hotkey(
      "zoom_fit", XK_0, static_cast<uint32_t>(KeyModifier::Ctrl),
      [this]() { zoom_pan_controller_.reset(); }, "Zoom to fit");

  // Selection operations
  hotkey_manager_.register_hotkey(
      "select_all", XK_a, static_cast<uint32_t>(KeyModifier::Ctrl),
      [this]() { selection_manager_.select_all(); }, "Select all");

  hotkey_manager_.register_hotkey(
      "deselect_all", XK_d, static_cast<uint32_t>(KeyModifier::Ctrl),
      [this]() { selection_manager_.clear_selection(); }, "Deselect all");

  // Trading operations
  hotkey_manager_.register_hotkey(
      "quick_buy", XK_F1, 0, []() { /* Quick buy */ }, "Quick buy order");

  hotkey_manager_.register_hotkey(
      "quick_sell", XK_F2, 0, []() { /* Quick sell */ }, "Quick sell order");

  hotkey_manager_.register_hotkey(
      "cancel_orders", XK_Escape, 0, []() { /* Cancel all orders */ },
      "Cancel all orders");

  // View switching
  hotkey_manager_.register_hotkey(
      "switch_to_chart", XK_1, static_cast<uint32_t>(KeyModifier::Alt),
      []() { /* Switch to chart view */ }, "Switch to chart view");

  hotkey_manager_.register_hotkey(
      "switch_to_orderbook", XK_2, static_cast<uint32_t>(KeyModifier::Alt),
      []() { /* Switch to orderbook view */ }, "Switch to orderbook view");

  hotkey_manager_.register_hotkey(
      "switch_to_trades", XK_3, static_cast<uint32_t>(KeyModifier::Alt),
      []() { /* Switch to trades view */ }, "Switch to trades view");
}

} // namespace BTQuant