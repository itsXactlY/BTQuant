#pragma once

/**
 * BTQuant Advanced Interaction Manager
 * 
 * Comprehensive input handling system for professional trading dashboard
 * with multi-touch, gesture recognition, and context-aware interactions.
 */

#include "vulkan_dashboard_advanced.hpp"
#include <X11/Xlib.h>
#include <X11/extensions/XInput2.h>
#include <unordered_set>
#include <functional>

namespace BTQuant {

// Forward declarations
class UIComponent;
class VulkanDashboard;

// ============================================================================
// Input Event System (defined in vulkan_dashboard_advanced.hpp)
// ============================================================================

// ============================================================================
// Context Menu System
// ============================================================================

struct ContextMenuItem {
    std::string label;
    std::string icon;
    std::function<void()> action;
    bool enabled = true;
    bool separator = false;
    std::vector<ContextMenuItem> submenu;
};

class ContextMenu {
public:
    ContextMenu(const glm::vec2& position);
    ~ContextMenu();
    
    void add_item(const ContextMenuItem& item);
    void add_separator();
    void show();
    void hide();
    bool is_visible() const { return visible_; }
    
    void render(VkCommandBuffer cmd);
    bool handle_input(const InputEvent& event);
    
private:
    glm::vec2 position_;
    std::vector<ContextMenuItem> items_;
    bool visible_ = false;
    int selected_item_ = -1;
    
    // Rendering resources
    BufferAllocation vertex_buffer_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    
    void rebuild_geometry();
    void execute_item(size_t index);
};

// ============================================================================
// Tooltip System
// ============================================================================

class TooltipManager {
public:
    struct TooltipData {
        std::string title;
        std::string content;
        glm::vec4 background_color{0.1f, 0.1f, 0.1f, 0.9f};
        glm::vec4 text_color{1.0f, 1.0f, 1.0f, 1.0f};
        float delay_ms = 500.0f;
        float max_width = 300.0f;
    };
    
    TooltipManager();
    ~TooltipManager();
    
    void show_tooltip(const glm::vec2& position, const TooltipData& data);
    void hide_tooltip();
    void update(float delta_time);
    void render(VkCommandBuffer cmd);
    
private:
    TooltipData current_tooltip_;
    glm::vec2 tooltip_position_;
    bool visible_ = false;
    bool pending_ = false;
    float show_timer_ = 0.0f;
    
    // Rendering resources
    BufferAllocation vertex_buffer_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    
    void rebuild_geometry();
};

// ============================================================================
// Drag and Drop System
// ============================================================================

enum class DragDropType {
    Symbol,
    Panel,
    Chart,
    Indicator,
    Layout
};

struct DragDropData {
    DragDropType type;
    std::string data;
    glm::vec2 offset{0.0f};
    UIComponent* source_component = nullptr;
    void* user_data = nullptr;
};

class DragDropManager {
public:
    using DropCallback = std::function<bool(const DragDropData&, const glm::vec2&)>;
    
    DragDropManager();
    ~DragDropManager();
    
    void start_drag(const DragDropData& data, const glm::vec2& start_position);
    void update_drag(const glm::vec2& position);
    void end_drag(const glm::vec2& position);
    void cancel_drag();
    
    bool is_dragging() const { return dragging_; }
    const DragDropData& get_drag_data() const { return drag_data_; }
    
    void register_drop_target(UIComponent* component, DropCallback callback);
    void unregister_drop_target(UIComponent* component);
    
    void render(VkCommandBuffer cmd);
    
private:
    bool dragging_ = false;
    DragDropData drag_data_;
    glm::vec2 drag_position_;
    glm::vec2 start_position_;
    
    std::unordered_map<UIComponent*, DropCallback> drop_targets_;
    
    // Visual feedback
    BufferAllocation drag_visual_buffer_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    
    void render_drag_visual();
    UIComponent* find_drop_target(const glm::vec2& position);
};

// ============================================================================
// Gesture Recognition System
// ============================================================================

class GestureRecognizer {
public:
    using GestureCallback = std::function<void(const GestureEvent&)>;
    
    GestureRecognizer();
    ~GestureRecognizer();
    
    void add_touch_point(const TouchPoint& point);
    void update_touch_point(const TouchPoint& point);
    void remove_touch_point(int touch_id);
    
    void set_gesture_callback(GestureType type, GestureCallback callback);
    void update(float delta_time);
    
private:
    std::unordered_map<int, TouchPoint> active_touches_;
    std::unordered_map<GestureType, GestureCallback> gesture_callbacks_;
    
    // Gesture state
    struct GestureState {
        bool active = false;
        std::chrono::high_resolution_clock::time_point start_time;
        glm::vec2 start_center{0.0f};
        float start_distance = 0.0f;
        float start_angle = 0.0f;
    };
    
    std::unordered_map<GestureType, GestureState> gesture_states_;
    
    // Recognition parameters
    float pinch_threshold_ = 10.0f;
    float rotation_threshold_ = 0.1f;
    float swipe_threshold_ = 50.0f;
    float tap_threshold_ = 10.0f;
    float long_press_duration_ = 1000.0f;
    
    void detect_gestures();
    void detect_pinch();
    void detect_rotation();
    void detect_swipe();
    void detect_tap();
    void detect_long_press();
    
    glm::vec2 calculate_center();
    float calculate_distance();
    float calculate_angle();
};

// ============================================================================
// Selection System
// ============================================================================

class SelectionManager {
public:
    SelectionManager();
    ~SelectionManager();
    
    void select_component(UIComponent* component, bool multi_select = false);
    void deselect_component(UIComponent* component);
    void clear_selection();
    void select_all();
    
    bool is_selected(UIComponent* component) const;
    const std::unordered_set<UIComponent*>& get_selected_components() const;
    size_t get_selection_count() const { return selected_components_.size(); }
    
    void render_selection_indicators(VkCommandBuffer cmd);
    
private:
    std::unordered_set<UIComponent*> selected_components_;
    
    // Visual feedback
    BufferAllocation selection_buffer_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    
    void rebuild_selection_geometry();
};

// ============================================================================
// Zoom and Pan Controller
// ============================================================================

class ZoomPanController {
public:
    ZoomPanController();
    ~ZoomPanController();
    
    void set_bounds(const glm::vec2& min_bounds, const glm::vec2& max_bounds);
    void set_zoom_limits(float min_zoom, float max_zoom);
    
    void pan(const glm::vec2& delta);
    void zoom(float factor, const glm::vec2& center);
    void zoom_to_fit(const glm::vec2& content_size);
    void reset();
    
    glm::mat4 get_transform_matrix() const;
    glm::vec2 get_pan_offset() const { return pan_offset_; }
    float get_zoom_factor() const { return zoom_factor_; }
    
    // Screen to world coordinate conversion
    glm::vec2 screen_to_world(const glm::vec2& screen_pos) const;
    glm::vec2 world_to_screen(const glm::vec2& world_pos) const;
    
private:
    glm::vec2 pan_offset_{0.0f};
    float zoom_factor_ = 1.0f;
    
    glm::vec2 min_bounds_{-1000.0f};
    glm::vec2 max_bounds_{1000.0f};
    float min_zoom_ = 0.1f;
    float max_zoom_ = 10.0f;
    
    void clamp_pan();
    void clamp_zoom();
};

// ============================================================================
// Hotkey System
// ============================================================================

struct HotkeyBinding {
    int key_code;
    uint32_t modifiers;
    std::function<void()> action;
    std::string description;
    bool enabled = true;
};

class HotkeyManager {
public:
    HotkeyManager();
    ~HotkeyManager();
    
    void register_hotkey(const std::string& name, int key_code, uint32_t modifiers, 
                        std::function<void()> action, const std::string& description = "");
    void unregister_hotkey(const std::string& name);
    void enable_hotkey(const std::string& name, bool enabled);
    
    bool handle_key_event(const InputEvent& event);
    
    std::vector<std::pair<std::string, HotkeyBinding>> get_all_hotkeys() const;
    
private:
    std::unordered_map<std::string, HotkeyBinding> hotkeys_;
    std::unordered_set<int> pressed_keys_;
    
    uint64_t make_key_hash(int key_code, uint32_t modifiers) const;
};

// ============================================================================
// Main Interaction Manager
// ============================================================================

class InteractionManager {
public:
    InteractionManager(VulkanDashboard* dashboard);
    ~InteractionManager();
    
    // Initialization
    void initialize(Display* display, Window window);
    void cleanup();
    
    // Event processing
    void process_x11_event(XEvent* event);
    void update(float delta_time);
    void render(VkCommandBuffer cmd);
    
    // Component registration
    void register_component(UIComponent* component);
    void unregister_component(UIComponent* component);
    
    // Subsystem access
    ContextMenu* create_context_menu(const glm::vec2& position);
    TooltipManager& get_tooltip_manager() { return tooltip_manager_; }
    DragDropManager& get_drag_drop_manager() { return drag_drop_manager_; }
    SelectionManager& get_selection_manager() { return selection_manager_; }
    ZoomPanController& get_zoom_pan_controller() { return zoom_pan_controller_; }
    HotkeyManager& get_hotkey_manager() { return hotkey_manager_; }
    
    // Input state queries
    bool is_mouse_button_pressed(MouseButton button) const;
    bool is_key_pressed(int key_code) const;
    glm::vec2 get_mouse_position() const { return mouse_position_; }
    uint32_t get_active_modifiers() const { return active_modifiers_; }
    
    // Focus management
    void set_focused_component(UIComponent* component);
    UIComponent* get_focused_component() const { return focused_component_; }
    
private:
    VulkanDashboard* dashboard_;
    Display* display_ = nullptr;
    Window window_;
    
    // Input state
    glm::vec2 mouse_position_{0.0f};
    std::unordered_set<int> pressed_mouse_buttons_;
    std::unordered_set<int> pressed_keys_;
    uint32_t active_modifiers_ = 0;
    
    // Component management
    std::vector<UIComponent*> registered_components_;
    UIComponent* focused_component_ = nullptr;
    UIComponent* hovered_component_ = nullptr;
    
    // Subsystems
    TooltipManager tooltip_manager_;
    DragDropManager drag_drop_manager_;
    SelectionManager selection_manager_;
    ZoomPanController zoom_pan_controller_;
    HotkeyManager hotkey_manager_;
    GestureRecognizer gesture_recognizer_;
    
    // Context menus
    std::vector<std::unique_ptr<ContextMenu>> context_menus_;
    
    // Multi-touch support
    bool multi_touch_enabled_ = false;
    int xi_opcode_ = 0;
    
    // Event processing
    void process_mouse_event(XEvent* event);
    void process_keyboard_event(XEvent* event);
    void process_touch_event(XEvent* event);
    void process_motion_event(XEvent* event);
    
    // Component interaction
    UIComponent* find_component_at_position(const glm::vec2& position);
    void update_hover_state(const glm::vec2& position);
    void dispatch_event_to_component(UIComponent* component, const InputEvent& event);
    
    // X11 input setup
    void setup_xinput2();
    void cleanup_xinput2();
    
    // Utility methods
    uint32_t x11_modifiers_to_internal(unsigned int x11_mods);
    InputEvent create_input_event(InputEventType type);
    
    // Default hotkeys setup
    void setup_default_hotkeys();
};

} // namespace BTQuant