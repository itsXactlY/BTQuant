#include "../../include/components/keyboard_shortcuts_component.hpp"
#include <imgui.h>
#include <imgui_internal.h>

namespace BTQuant {

KeyboardShortcutsComponent::KeyboardShortcutsComponent(const glm::vec2 &position,
                                                       const glm::vec2 &size)
    : UIComponent(position, size) {
  initialize_default_shortcuts();
}

KeyboardShortcutsComponent::~KeyboardShortcutsComponent() {}

void KeyboardShortcutsComponent::initialize_vulkan_resources(VulkanCore *) {}

void KeyboardShortcutsComponent::initialize_default_shortcuts() {
  std::lock_guard<std::mutex> lock(data_mutex_);

  // Trading shortcuts
  shortcuts_["buy_market"] = {"Buy Market", "Place a market buy order", "Ctrl+B",
                              []() { /* Implement buy market order */ }, true};
  shortcuts_["sell_market"] = {"Sell Market", "Place a market sell order", "Ctrl+S",
                               []() { /* Implement sell market order */ }, true};
  shortcuts_["cancel_all"] = {"Cancel All", "Cancel all active orders", "Ctrl+X",
                              []() { /* Implement cancel all orders */ }, true};
  shortcuts_["flatten"] = {"Flatten", "Close all positions", "Ctrl+F",
                            []() { /* Implement flatten positions */ }, true};

  // Chart shortcuts
  shortcuts_["zoom_in"] = {"Zoom In", "Zoom in on chart", "Ctrl++",
                           []() { /* Implement zoom in */ }, true};
  shortcuts_["zoom_out"] = {"Zoom Out", "Zoom out on chart", "Ctrl+-",
                            []() { /* Implement zoom out */ }, true};
  shortcuts_["pan_left"] = {"Pan Left", "Pan chart to the left", "Left",
                            []() { /* Implement pan left */ }, true};
  shortcuts_["pan_right"] = {"Pan Right", "Pan chart to the right", "Right",
                             []() { /* Implement pan right */ }, true};

  // Timeframe shortcuts
  shortcuts_["tf_1m"] = {"1 Minute", "Change to 1-minute timeframe", "1",
                         []() { /* Implement timeframe change */ }, true};
  shortcuts_["tf_5m"] = {"5 Minutes", "Change to 5-minute timeframe", "5",
                         []() { /* Implement timeframe change */ }, true};
  shortcuts_["tf_15m"] = {"15 Minutes", "Change to 15-minute timeframe", "15",
                          []() { /* Implement timeframe change */ }, true};
  shortcuts_["tf_1h"] = {"1 Hour", "Change to 1-hour timeframe", "H",
                         []() { /* Implement timeframe change */ }, true};
  shortcuts_["tf_4h"] = {"4 Hours", "Change to 4-hour timeframe", "4",
                         []() { /* Implement timeframe change */ }, true};
  shortcuts_["tf_1d"] = {"1 Day", "Change to 1-day timeframe", "D",
                         []() { /* Implement timeframe change */ }, true};

  // UI shortcuts
  shortcuts_["toggle_fullscreen"] = {"Toggle Fullscreen", "Toggle fullscreen mode", "F11",
                                     []() { /* Implement fullscreen toggle */ }, true};
  shortcuts_["toggle_help"] = {"Toggle Help", "Show/hide help menu", "F1",
                               [this]() { show_help_ = !show_help_; }, true};
  shortcuts_["toggle_dark_mode"] = {"Toggle Dark Mode", "Switch between light/dark themes", "Ctrl+D",
                                     []() { /* Implement theme toggle */ }, true};
  shortcuts_["reset_layout"] = {"Reset Layout", "Reset dashboard to default layout", "Ctrl+R",
                                []() { /* Implement layout reset */ }, true};
  shortcuts_["save_layout"] = {"Save Layout", "Save current layout", "Ctrl+Shift+S",
                               []() { /* Implement layout save */ }, true};
  shortcuts_["load_layout"] = {"Load Layout", "Load saved layout", "Ctrl+Shift+L",
                               []() { /* Implement layout load */ }, true};

  // Indicators shortcuts
  shortcuts_["toggle_sma"] = {"Toggle SMA", "Show/hide simple moving average", "S",
                              []() { /* Implement SMA toggle */ }, true};
  shortcuts_["toggle_ema"] = {"Toggle EMA", "Show/hide exponential moving average", "E",
                              []() { /* Implement EMA toggle */ }, true};
  shortcuts_["toggle_rsi"] = {"Toggle RSI", "Show/hide RSI indicator", "R",
                              []() { /* Implement RSI toggle */ }, true};
  shortcuts_["toggle_macd"] = {"Toggle MACD", "Show/hide MACD indicator", "M",
                               []() { /* Implement MACD toggle */ }, true};
  shortcuts_["toggle_bb"] = {"Toggle Bollinger Bands", "Show/hide Bollinger Bands", "B",
                             []() { /* Implement BB toggle */ }, true};
  shortcuts_["toggle_volume"] = {"Toggle Volume", "Show/hide volume indicator", "V",
                                 []() { /* Implement volume toggle */ }, true};

  // Risk management shortcuts
  shortcuts_["emergency_stop"] = {"Emergency Stop", "Trigger emergency de-leverage", "Ctrl+E",
                                  []() { /* Implement emergency stop */ }, true};
  shortcuts_["toggle_risk_limits"] = {"Toggle Risk Limits", "Show/hide risk management panel", "Ctrl+L",
                                      []() { /* Implement risk panel toggle */ }, true};

  // System shortcuts
  shortcuts_["toggle_performance"] = {"Toggle Performance", "Show/hide performance monitor", "Ctrl+P",
                                      []() { /* Implement performance toggle */ }, true};
  shortcuts_["toggle_logs"] = {"Toggle Logs", "Show/hide system logs", "Ctrl+O",
                               []() { /* Implement logs toggle */ }, true};
  shortcuts_["take_screenshot"] = {"Take Screenshot", "Capture screenshot of dashboard", "F12",
                                    []() { /* Implement screenshot */ }, true};
  shortcuts_["toggle_recording"] = {"Toggle Recording", "Start/stop recording", "F9",
                                     []() { /* Implement recording toggle */ }, true};
}

void KeyboardShortcutsComponent::update(float dt) {}

void KeyboardShortcutsComponent::render_gui() {
  if (show_help_) {
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2 - 250, ImGui::GetIO().DisplaySize.y / 2 - 300),
                           ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_Appearing);
    
    if (ImGui::Begin("Keyboard Shortcuts Help", &show_help_, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_Modal)) {
      ImGui::TextColored(ImVec4(theme_.accent_primary.x, theme_.accent_primary.y, theme_.accent_primary.z, 1),
                        "Keyboard Shortcuts");
      ImGui::Separator();

      ImGui::BeginChild("ShortcutsList", ImVec2(0, 500), true);
      
      std::lock_guard<std::mutex> lock(data_mutex_);

      if (ImGui::CollapsingHeader("Trading")) {
        auto render_shortcut = [](const Shortcut &s) {
          ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "%-20s", s.keys.c_str());
          ImGui::SameLine();
          ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "%-30s", s.name.c_str());
          ImGui::SameLine();
          ImGui::TextDisabled("%s", s.description.c_str());
        };

        for (const auto &[name, shortcut] : shortcuts_) {
          if (name.find("buy_") == 0 || name.find("sell_") == 0 ||
              name.find("cancel_") == 0 || name.find("flatten") == 0) {
            render_shortcut(shortcut);
          }
        }
      }

      if (ImGui::CollapsingHeader("Charts")) {
        for (const auto &[name, shortcut] : shortcuts_) {
          if (name.find("zoom_") == 0 || name.find("pan_") == 0) {
            ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "%-20s", shortcut.keys.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "%-30s", shortcut.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%s", shortcut.description.c_str());
          }
        }
      }

      if (ImGui::CollapsingHeader("Timeframes")) {
        for (const auto &[name, shortcut] : shortcuts_) {
          if (name.find("tf_") == 0) {
            ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "%-20s", shortcut.keys.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "%-30s", shortcut.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%s", shortcut.description.c_str());
          }
        }
      }

      if (ImGui::CollapsingHeader("Indicators")) {
        for (const auto &[name, shortcut] : shortcuts_) {
          if (name.find("toggle_") == 0 && (name.find("sma") != std::string::npos ||
                                           name.find("ema") != std::string::npos ||
                                           name.find("rsi") != std::string::npos ||
                                           name.find("macd") != std::string::npos ||
                                           name.find("bb") != std::string::npos ||
                                           name.find("volume") != std::string::npos)) {
            ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "%-20s", shortcut.keys.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "%-30s", shortcut.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%s", shortcut.description.c_str());
          }
        }
      }

      if (ImGui::CollapsingHeader("UI")) {
        for (const auto &[name, shortcut] : shortcuts_) {
          if ((name.find("toggle_") == 0 && (name.find("fullscreen") != std::string::npos ||
                                           name.find("help") != std::string::npos ||
                                           name.find("dark") != std::string::npos)) ||
              name.find("reset_layout") != std::string::npos ||
              name.find("save_layout") != std::string::npos ||
              name.find("load_layout") != std::string::npos) {
            ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "%-20s", shortcut.keys.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "%-30s", shortcut.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%s", shortcut.description.c_str());
          }
        }
      }

      if (ImGui::CollapsingHeader("Risk Management")) {
        for (const auto &[name, shortcut] : shortcuts_) {
          if (name.find("emergency") != std::string::npos ||
              name.find("toggle_risk") != std::string::npos) {
            ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "%-20s", shortcut.keys.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "%-30s", shortcut.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%s", shortcut.description.c_str());
          }
        }
      }

      if (ImGui::CollapsingHeader("System")) {
        for (const auto &[name, shortcut] : shortcuts_) {
          if (name.find("toggle_performance") != std::string::npos ||
              name.find("toggle_logs") != std::string::npos ||
              name.find("take_screenshot") != std::string::npos ||
              name.find("toggle_recording") != std::string::npos) {
            ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "%-20s", shortcut.keys.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "%-30s", shortcut.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%s", shortcut.description.c_str());
          }
        }
      }

      ImGui::EndChild();
    }
    ImGui::End();
  }
}

void KeyboardShortcutsComponent::clear_data() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  shortcuts_.clear();
  key_states_.clear();
  current_modifiers_ = 0;
  show_help_ = false;
}

void KeyboardShortcutsComponent::handle_input(const InputEvent &event) {
  if (event.type == InputEventType::KeyDown || event.type == InputEventType::KeyUp) {
    key_states_[event.keycode] = (event.type == InputEventType::KeyDown);
    current_modifiers_ = event.modifiers;

    if (event.type == InputEventType::KeyDown) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      
      if (current_modifiers_ == 0) {
        if (event.keycode == 256) {
          show_help_ = false;
        } else if (event.keycode == '1') {
          execute_shortcut("tf_1m");
        } else if (event.keycode == '5') {
          execute_shortcut("tf_5m");
        } else if (event.keycode == 'S') {
          execute_shortcut("toggle_sma");
        } else if (event.keycode == 'E') {
          execute_shortcut("toggle_ema");
        } else if (event.keycode == 'R') {
          execute_shortcut("toggle_rsi");
        } else if (event.keycode == 'M') {
          execute_shortcut("toggle_macd");
        } else if (event.keycode == 'B') {
          execute_shortcut("toggle_bb");
        } else if (event.keycode == 'V') {
          execute_shortcut("toggle_volume");
        } else if (event.keycode == GLFW_KEY_LEFT) {
          execute_shortcut("pan_left");
        } else if (event.keycode == GLFW_KEY_RIGHT) {
          execute_shortcut("pan_right");
        }
      }

      if (current_modifiers_ & (1 << 1)) {
        if (event.keycode == 'B') {
          execute_shortcut("buy_market");
        } else if (event.keycode == 'S') {
          execute_shortcut("sell_market");
        } else if (event.keycode == 'X') {
          execute_shortcut("cancel_all");
        } else if (event.keycode == 'F') {
          execute_shortcut("flatten");
        } else if (event.keycode == 'D') {
          execute_shortcut("toggle_dark_mode");
        } else if (event.keycode == 'R') {
          execute_shortcut("reset_layout");
        } else if (event.keycode == 'L') {
          execute_shortcut("toggle_risk_limits");
        } else if (event.keycode == 'P') {
          execute_shortcut("toggle_performance");
        } else if (event.keycode == 'O') {
          execute_shortcut("toggle_logs");
        } else if (event.keycode == 'E') {
          execute_shortcut("emergency_stop");
        } else if (event.keycode == '+') {
          execute_shortcut("zoom_in");
        } else if (event.keycode == '-') {
          execute_shortcut("zoom_out");
        }
      }

      if (current_modifiers_ & (1 << 1) && current_modifiers_ & (1 << 0)) {
        if (event.keycode == 'S') {
          execute_shortcut("save_layout");
        } else if (event.keycode == 'L') {
          execute_shortcut("load_layout");
        }
      }

      if (event.keycode == GLFW_KEY_F1) {
        execute_shortcut("toggle_help");
      } else if (event.keycode == GLFW_KEY_F9) {
        execute_shortcut("toggle_recording");
      } else if (event.keycode == GLFW_KEY_F11) {
        execute_shortcut("toggle_fullscreen");
      } else if (event.keycode == GLFW_KEY_F12) {
        execute_shortcut("take_screenshot");
      }
    }
  }
}

void KeyboardShortcutsComponent::add_shortcut(const Shortcut &shortcut) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  shortcuts_[shortcut.name] = shortcut;
}

void KeyboardShortcutsComponent::remove_shortcut(const std::string &name) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = shortcuts_.find(name);
  if (it != shortcuts_.end()) {
    shortcuts_.erase(it);
  }
}

void KeyboardShortcutsComponent::enable_shortcut(const std::string &name, bool enabled) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = shortcuts_.find(name);
  if (it != shortcuts_.end()) {
    it->second.active = enabled;
  }
}

bool KeyboardShortcutsComponent::is_shortcut_active(const std::string &name) const {
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = shortcuts_.find(name);
  return it != shortcuts_.end() && it->second.active;
}

void KeyboardShortcutsComponent::execute_shortcut(const std::string &name) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = shortcuts_.find(name);
  if (it != shortcuts_.end() && it->second.active && it->second.callback) {
    it->second.callback();
  }
}

bool KeyboardShortcutsComponent::handle_key_event(int keycode, bool pressed, uint32_t modifiers) {
  if (pressed) {
    key_states_[keycode] = true;
    current_modifiers_ = modifiers;
  } else {
    key_states_[keycode] = false;
  }
  return true;
}

} // namespace BTQuant
