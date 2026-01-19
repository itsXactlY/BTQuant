#pragma once

#include "../vulkan_dashboard_advanced.hpp"

namespace BTQuant {

class KeyboardShortcutsComponent : public UIComponent {
public:
  KeyboardShortcutsComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~KeyboardShortcutsComponent() override;

  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void handle_input(const InputEvent &event) override;

  struct Shortcut {
    std::string name;
    std::string description;
    std::string keys;
    std::function<void()> callback;
    bool active;
  };

  void add_shortcut(const Shortcut &shortcut);
  void remove_shortcut(const std::string &name);
  void enable_shortcut(const std::string &name, bool enabled);
  bool is_shortcut_active(const std::string &name) const;

private:
  void initialize_default_shortcuts();
  void execute_shortcut(const std::string &name);
  bool handle_key_event(int keycode, bool pressed, uint32_t modifiers);

  std::unordered_map<std::string, Shortcut> shortcuts_;
  std::unordered_map<int, bool> key_states_;
  uint32_t current_modifiers_ = 0;
  DashboardTheme theme_;
  bool show_help_ = false;
  mutable std::mutex data_mutex_;
};

} // namespace BTQuant
