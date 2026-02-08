#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <fstream>
#include <sstream>

#include "../../include/components/keyboard_shortcuts_component.hpp"
#include "../../include/ui/settings_manager.hpp"

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace BTQuant {

// Constructor
KeyboardShortcutsComponent::KeyboardShortcutsComponent(const glm::vec2& position,
                                                       const glm::vec2& size)
    : UIComponent(position, size) {
  initialize_default_shortcuts();
}

// Destructor
KeyboardShortcutsComponent::~KeyboardShortcutsComponent() = default;

void KeyboardShortcutsComponent::initialize_vulkan_resources(VulkanCore* core) {
  // Initialize any Vulkan resources needed for the component
}

void KeyboardShortcutsComponent::update(float dt) {
  // Update logic if needed
}

void KeyboardShortcutsComponent::clear_data() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  shortcuts_.clear();
}

void KeyboardShortcutsComponent::handle_input(const InputEvent& event) {
  if (event.type == InputEventType::KeyDown) {
    handle_key_event(event.keycode, true, event.modifiers);
  } else if (event.type == InputEventType::KeyUp) {
    handle_key_event(event.keycode, false, event.modifiers);
  }
}

void KeyboardShortcutsComponent::add_shortcut(const Shortcut& shortcut) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  shortcuts_[shortcut.name] = shortcut;
}

void KeyboardShortcutsComponent::remove_shortcut(const std::string& name) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  shortcuts_.erase(name);
}

void KeyboardShortcutsComponent::enable_shortcut(const std::string& name, bool enabled) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = shortcuts_.find(name);
  if (it != shortcuts_.end()) {
    it->second.active = enabled;
  }
}

bool KeyboardShortcutsComponent::is_shortcut_active(const std::string& name) const {
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = shortcuts_.find(name);
  if (it != shortcuts_.end()) {
    return it->second.active;
  }
  return false;
}

void KeyboardShortcutsComponent::initialize_default_shortcuts() {
  std::lock_guard<std::mutex> lock(data_mutex_);

  // Define default shortcuts
  shortcuts_["toggle_fullscreen"] = {"toggle_fullscreen", "Toggle Fullscreen Mode", "F11",
                                     []() { /* callback */ }, true};
  shortcuts_["save_layout"] = {"save_layout", "Save Current Layout", "Ctrl+S",
                               []() { /* callback */ }, true};
  shortcuts_["load_layout"] = {"load_layout", "Load Layout", "Ctrl+L", []() { /* callback */ },
                               true};
  shortcuts_["take_screenshot"] = {"take_screenshot", "Take Screenshot", "Ctrl+Shift+S",
                                   []() { /* callback */ }, true};
  shortcuts_["open_settings"] = {"open_settings", "Open Settings Dialog", "Ctrl+,",
                                 []() { /* callback */ }, true};
  shortcuts_["new_workspace"] = {"new_workspace", "Create New Workspace", "Ctrl+N",
                                 []() { /* callback */ }, true};
  shortcuts_["close_tab"] = {"close_tab", "Close Current Tab", "Ctrl+W", []() { /* callback */ },
                             true};
  shortcuts_["zoom_in"] = {"zoom_in", "Zoom In", "Ctrl++", []() { /* callback */ }, true};
  shortcuts_["zoom_out"] = {"zoom_out", "Zoom Out", "Ctrl+-", []() { /* callback */ }, true};
  shortcuts_["reset_zoom"] = {"reset_zoom", "Reset Zoom Level", "Ctrl+0", []() { /* callback */ },
                              true};
  shortcuts_["undo"] = {"undo", "Undo Last Action", "Ctrl+Z", []() { /* callback */ }, true};
  shortcuts_["redo"] = {"redo", "Redo Last Action", "Ctrl+Y", []() { /* callback */ }, true};
  shortcuts_["copy"] = {"copy", "Copy Selection", "Ctrl+C", []() { /* callback */ }, true};
  shortcuts_["paste"] = {"paste", "Paste from Clipboard", "Ctrl+V", []() { /* callback */ }, true};
  shortcuts_["cut"] = {"cut", "Cut Selection", "Ctrl+X", []() { /* callback */ }, true};
  shortcuts_["duplicate"] = {"duplicate", "Duplicate Selection", "Ctrl+D", []() { /* callback */ },
                             true};
  shortcuts_["search"] = {"search", "Open Search Dialog", "Ctrl+F", []() { /* callback */ }, true};
  shortcuts_["find_next"] = {"find_next", "Find Next Occurrence", "F3", []() { /* callback */ },
                             true};
  shortcuts_["find_prev"] = {"find_prev", "Find Previous Occurrence", "Shift+F3",
                             []() { /* callback */ }, true};
  shortcuts_["toggle_console"] = {"toggle_console", "Toggle Console Panel", "`",
                                  []() { /* callback */ }, true};
  shortcuts_["toggle_watchlist"] = {"toggle_watchlist", "Toggle Watchlist Panel", "Ctrl+Shift+W",
                                    []() { /* callback */ }, true};
  shortcuts_["toggle_orders"] = {"toggle_orders", "Toggle Orders Panel", "Ctrl+Shift+O",
                                 []() { /* callback */ }, true};
  shortcuts_["toggle_positions"] = {"toggle_positions", "Toggle Positions Panel", "Ctrl+Shift+P",
                                    []() { /* callback */ }, true};
  shortcuts_["quick_buy"] = {"quick_buy", "Quick Buy Order", "F9", []() { /* callback */ }, true};
  shortcuts_["quick_sell"] = {"quick_sell", "Quick Sell Order", "F10", []() { /* callback */ },
                              true};
  shortcuts_["cancel_all_orders"] = {"cancel_all_orders", "Cancel All Orders", "Ctrl+Shift+C",
                                     []() { /* callback */ }, true};
}

void KeyboardShortcutsComponent::execute_shortcut(const std::string& name) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = shortcuts_.find(name);
  if (it != shortcuts_.end() && it->second.active && it->second.callback) {
    it->second.callback();
  }
}

bool KeyboardShortcutsComponent::handle_key_event(int keycode, bool pressed, uint32_t modifiers) {
  if (!pressed) return false;  // Only handle key press events

  // Build a string representation of the key combination
  std::string key_combo = "";

  if (modifiers & static_cast<uint32_t>(KeyModifier::Ctrl)) {
    key_combo += "Ctrl+";
  }
  if (modifiers & static_cast<uint32_t>(KeyModifier::Alt)) {
    key_combo += "Alt+";
  }
  if (modifiers & static_cast<uint32_t>(KeyModifier::Shift)) {
    key_combo += "Shift+";
  }

  // Map keycode to string representation
  std::string key_name = "";
  switch (keycode) {
    case 32:
      key_name = "Space";
      break;  // Space
    case 39:
      key_name = "'";
      break;  // Apostrophe
    case 44:
      key_name = ",";
      break;  // Comma
    case 45:
      key_name = "-";
      break;  // Minus
    case 46:
      key_name = ".";
      break;  // Period
    case 47:
      key_name = "/";
      break;  // Slash
    case 48:
      key_name = "0";
      break;
    case 49:
      key_name = "1";
      break;
    case 50:
      key_name = "2";
      break;
    case 51:
      key_name = "3";
      break;
    case 52:
      key_name = "4";
      break;
    case 53:
      key_name = "5";
      break;
    case 54:
      key_name = "6";
      break;
    case 55:
      key_name = "7";
      break;
    case 56:
      key_name = "8";
      break;
    case 57:
      key_name = "9";
      break;
    case 59:
      key_name = ";";
      break;  // Semicolon
    case 61:
      key_name = "=";
      break;  // Equal
    case 65:
      key_name = "A";
      break;
    case 66:
      key_name = "B";
      break;
    case 67:
      key_name = "C";
      break;
    case 68:
      key_name = "D";
      break;
    case 69:
      key_name = "E";
      break;
    case 70:
      key_name = "F";
      break;
    case 71:
      key_name = "G";
      break;
    case 72:
      key_name = "H";
      break;
    case 73:
      key_name = "I";
      break;
    case 74:
      key_name = "J";
      break;
    case 75:
      key_name = "K";
      break;
    case 76:
      key_name = "L";
      break;
    case 77:
      key_name = "M";
      break;
    case 78:
      key_name = "N";
      break;
    case 79:
      key_name = "O";
      break;
    case 80:
      key_name = "P";
      break;
    case 81:
      key_name = "Q";
      break;
    case 82:
      key_name = "R";
      break;
    case 83:
      key_name = "S";
      break;
    case 84:
      key_name = "T";
      break;
    case 85:
      key_name = "U";
      break;
    case 86:
      key_name = "V";
      break;
    case 87:
      key_name = "W";
      break;
    case 88:
      key_name = "X";
      break;
    case 89:
      key_name = "Y";
      break;
    case 90:
      key_name = "Z";
      break;
    case 91:
      key_name = "[";
      break;  // Left bracket
    case 92:
      key_name = "\\";
      break;  // Backslash
    case 93:
      key_name = "]";
      break;  // Right bracket
    case 96:
      key_name = "`";
      break;  // Grave accent
    case 256:
      key_name = "Esc";
      break;
    case 257:
      key_name = "Enter";
      break;
    case 258:
      key_name = "Tab";
      break;
    case 259:
      key_name = "Backspace";
      break;
    case 260:
      key_name = "Insert";
      break;
    case 261:
      key_name = "Del";
      break;
    case 262:
      key_name = "Right";
      break;
    case 263:
      key_name = "Left";
      break;
    case 264:
      key_name = "Down";
      break;
    case 265:
      key_name = "Up";
      break;
    case 266:
      key_name = "Page Up";
      break;
    case 267:
      key_name = "Page Down";
      break;
    case 268:
      key_name = "Home";
      break;
    case 269:
      key_name = "End";
      break;
    case 280:
      key_name = "Caps Lock";
      break;
    case 281:
      key_name = "Scroll Lock";
      break;
    case 282:
      key_name = "Num Lock";
      break;
    case 283:
      key_name = "Print Screen";
      break;
    case 284:
      key_name = "Pause";
      break;
    case 290:
      key_name = "F1";
      break;
    case 291:
      key_name = "F2";
      break;
    case 292:
      key_name = "F3";
      break;
    case 293:
      key_name = "F4";
      break;
    case 294:
      key_name = "F5";
      break;
    case 295:
      key_name = "F6";
      break;
    case 296:
      key_name = "F7";
      break;
    case 297:
      key_name = "F8";
      break;
    case 298:
      key_name = "F9";
      break;
    case 299:
      key_name = "F10";
      break;
    case 300:
      key_name = "F11";
      break;
    case 301:
      key_name = "F12";
      break;
    case 320:
      key_name = "KP 0";
      break;  // Keypad 0
    case 321:
      key_name = "KP 1";
      break;
    case 322:
      key_name = "KP 2";
      break;
    case 323:
      key_name = "KP 3";
      break;
    case 324:
      key_name = "KP 4";
      break;
    case 325:
      key_name = "KP 5";
      break;
    case 326:
      key_name = "KP 6";
      break;
    case 327:
      key_name = "KP 7";
      break;
    case 328:
      key_name = "KP 8";
      break;
    case 329:
      key_name = "KP 9";
      break;
    case 330:
      key_name = "KP .";
      break;  // Keypad .
    case 331:
      key_name = "KP /";
      break;  // Keypad /
    case 332:
      key_name = "KP *";
      break;  // Keypad *
    case 333:
      key_name = "KP -";
      break;  // Keypad -
    case 334:
      key_name = "KP +";
      break;  // Keypad +
    case 335:
      key_name = "KP Enter";
      break;
    case 336:
      key_name = "KP =";
      break;  // Keypad =
    default:
      key_name = "Unknown";
      break;
  }

  key_combo += key_name;

  // Find and execute the matching shortcut
  std::lock_guard<std::mutex> lock(data_mutex_);
  for (auto& pair : shortcuts_) {
    if (pair.second.keys == key_combo && pair.second.active) {
      if (pair.second.callback) {
        pair.second.callback();
        return true;
      }
    }
  }

  return false;
}

void KeyboardShortcutsComponent::render_gui() {
  // Check if we're in a valid ImGui frame scope to prevent assertion errors
  // We can check this by attempting to get the current context and checking if it's valid
  ImGuiContext* g = ImGui::GetCurrentContext();
  if (g == nullptr) {
    // If there's no valid ImGui context, skip rendering this frame
    return;
  }

  // In newer versions of ImGui, we can't directly access WithinFrameScope
  // Instead, we'll just check if the context is valid and proceed with rendering
  // If we're not in a proper frame, ImGui will handle the error internally

  ImGui::Begin("Keyboard Shortcuts Editor", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

  static std::string filter = "";
  ImGui::Text("Filter shortcuts:");
  ImGui::InputText("##filter", &filter);

  ImGui::Separator();

  // Show import/export/reset buttons
  if (ImGui::Button("Import Profile")) {
    import_shortcut_profile();
  }
  ImGui::SameLine();
  if (ImGui::Button("Export Profile")) {
    export_shortcut_profile();
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset to Defaults")) {
    reset_to_defaults();
  }

  ImGui::Separator();

  // Table of shortcuts
  if (ImGui::BeginTable(
          "shortcuts_table", 4,
          ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 200.0f);
    ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Shortcut", ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableHeadersRow();

    std::lock_guard<std::mutex> lock(data_mutex_);
    for (auto& pair : shortcuts_) {
      const auto& name = pair.first;
      auto& shortcut = pair.second;

      // Apply filter
      if (!filter.empty()) {
        std::string lower_name = name;
        std::string lower_desc = shortcut.description;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        std::transform(lower_desc.begin(), lower_desc.end(), lower_desc.begin(), ::tolower);
        std::string lower_filter = filter;
        std::transform(lower_filter.begin(), lower_filter.end(), lower_filter.begin(), ::tolower);

        if (lower_name.find(lower_filter) == std::string::npos &&
            lower_desc.find(lower_filter) == std::string::npos) {
          continue;
        }
      }

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%s", name.c_str());

      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", shortcut.description.c_str());

      ImGui::TableSetColumnIndex(2);
      static std::string editing_shortcut = "";
      static bool is_editing = false;
      static std::string editing_name = "";

      if (is_editing && editing_name == name) {
        ImGui::InputText("##edit_shortcut", &editing_shortcut);
        if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) {
          shortcut.keys = editing_shortcut;
          is_editing = false;
          editing_name = "";
        } else if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
          is_editing = false;
          editing_name = "";
        }
      } else {
        ImGui::Text("%s", shortcut.keys.c_str());
        if (ImGui::IsItemClicked()) {
          editing_shortcut = shortcut.keys;
          is_editing = true;
          editing_name = name;
        }
      }

      ImGui::TableSetColumnIndex(3);
      ImGui::PushID(name.c_str());
      if (ImGui::Checkbox("##active", &shortcut.active)) {
        // Checkbox state changed
      }
      ImGui::SameLine();
      if (ImGui::Button("Reset")) {
        // Reset to default shortcut
        reset_shortcut_to_default(name);
      }
      ImGui::PopID();
    }

    ImGui::EndTable();
  }

  ImGui::End();
}

void KeyboardShortcutsComponent::import_shortcut_profile() {
#ifdef HAS_NLOHMANN_JSON
  try {
    // In a real implementation, this would open a file dialog
    // For now, we'll simulate loading from a file named "shortcuts.json"
    std::ifstream file("shortcuts.json");
    if (!file.is_open()) {
      // Could not open file
      return;
    }

    nlohmann::json shortcuts_json;
    file >> shortcuts_json;
    file.close();

    std::lock_guard<std::mutex> lock(data_mutex_);
    for (auto& pair : shortcuts_json.items()) {
      std::string name = pair.key();
      auto shortcut_json = pair.value();

      auto it = shortcuts_.find(name);
      if (it != shortcuts_.end()) {
        // Update the shortcut keys
        if (shortcut_json.contains("keys")) {
          it->second.keys = shortcut_json["keys"].get<std::string>();
        }
        if (shortcut_json.contains("active")) {
          it->second.active = shortcut_json["active"].get<bool>();
        }
      }
    }
  } catch (const std::exception& e) {
    // Error importing shortcuts
  }
#endif
}

void KeyboardShortcutsComponent::export_shortcut_profile() {
#ifdef HAS_NLOHMANN_JSON
  try {
    nlohmann::json shortcuts_json;

    std::lock_guard<std::mutex> lock(data_mutex_);
    for (const auto& pair : shortcuts_) {
      const auto& name = pair.first;
      const auto& shortcut = pair.second;

      nlohmann::json shortcut_json;
      shortcut_json["name"] = shortcut.name;
      shortcut_json["description"] = shortcut.description;
      shortcut_json["keys"] = shortcut.keys;
      shortcut_json["active"] = shortcut.active;

      shortcuts_json[name] = shortcut_json;
    }

    std::ofstream file("shortcuts.json");
    if (file.is_open()) {
      file << shortcuts_json.dump(4);
      file.close();
    }
  } catch (const std::exception& e) {
    // Error exporting shortcuts
  }
#endif
}

void KeyboardShortcutsComponent::reset_to_defaults() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  initialize_default_shortcuts();
}

void KeyboardShortcutsComponent::reset_shortcut_to_default(const std::string& name) {
  std::lock_guard<std::mutex> lock(data_mutex_);

  // Create a temporary component to get the default shortcut
  KeyboardShortcutsComponent temp_component(glm::vec2(0, 0), glm::vec2(0, 0));
  auto it = temp_component.shortcuts_.find(name);
  if (it != temp_component.shortcuts_.end()) {
    auto target_it = shortcuts_.find(name);
    if (target_it != shortcuts_.end()) {
      target_it->second.keys = it->second.keys;
      target_it->second.active = it->second.active;
    }
  }
}

}  // namespace BTQuant