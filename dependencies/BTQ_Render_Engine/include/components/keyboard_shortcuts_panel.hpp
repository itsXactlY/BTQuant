#pragma once

#include "panel_base.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace BTQuant {

/**
 * KeyboardShortcutsPanel - Keyboard shortcut configuration and display
 * 
 * Features (Stub):
 * - View all keyboard shortcuts by category
 * - Customize key bindings
 * - Import/Export shortcut profiles
 * - Search functionality
 * - Conflict detection
 */
class KeyboardShortcutsPanel : public PanelBase {
public:
    explicit KeyboardShortcutsPanel(const PanelConfig& config);
    ~KeyboardShortcutsPanel() override = default;

    void initialize() override;
    void render() override;

    // Shortcut management
    void reset_to_defaults();
    bool export_shortcuts(const std::string& filename);
    bool import_shortcuts(const std::string& filename);

private:
    struct Shortcut {
        std::string action;
        std::string key_combo;
        int category;  // 0: General, 1: Trading, 2: Chart, etc.
        bool is_recording;
    };
    
    std::vector<Shortcut> shortcuts_;
    char search_filter_[64] = "";
    int selected_category_ = 0;  // 0 = All
    int recording_index_ = -1;
    
    void render_search_filter();
    void render_shortcut_list();
    void render_actions();
    void start_recording(int index);
    void stop_recording();
};

}  // namespace BTQuant
