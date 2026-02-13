#pragma once

#include <memory>
#include <string>
#include <vector>

#include "layout/layout_presets.hpp"

namespace BTQuant {

namespace UI {

// ============================================================================
// Layout Manager System
// ============================================================================

class LayoutManager {
public:
    // Singleton access
    static LayoutManager& getInstance() {
        static LayoutManager instance;
        return instance;
    }

    // Delete copy and move constructors and assignment operators
    LayoutManager(const LayoutManager&) = delete;
    LayoutManager& operator=(const LayoutManager&) = delete;
    LayoutManager(LayoutManager&&) = delete;
    LayoutManager& operator=(LayoutManager&&) = delete;

private:
    LayoutManager();  // Private constructor for singleton
    ~LayoutManager(); // Private destructor for singleton

public:

    // Save current panel arrangement as named preset
    bool save_current_layout_as_preset(const std::string& preset_name,
                                      const std::string& description = "",
                                      const std::string& category = "Custom");

    // Load preset to restore layout
    bool load_preset_layout(const std::string& preset_name);

    // Delete preset
    bool delete_preset(const std::string& preset_name);

    // Get all available presets
    std::vector<Layout::LayoutPreset> get_all_presets() const;

    // Get presets by category
    std::vector<Layout::LayoutPreset> get_presets_by_category(const std::string& category) const;

    // Check if a preset exists
    bool preset_exists(const std::string& preset_name) const;


    // Quick-save functionality for F5-F8 hotkeys
    bool quick_save_layout(int slot_num);
    bool quick_load_layout(int slot_num);
    std::string get_quick_save_name(int slot_num) const;

    // Get the currently active layout name
    std::string get_active_layout_name() const;

    // Track which quick save slot is currently loaded (0 if none)
    void set_active_quick_slot(int slot_num);
    int get_active_quick_slot() const;

private:
    std::unique_ptr<Layout::LayoutPresetManager> preset_manager_;
    std::string presets_directory_;
    int active_quick_slot_ = 0;  // 0 means no quick save slot is active, 1-4 for slots

    void initialize_presets_directory();
    std::string get_current_layout_json();
    bool apply_layout_from_preset(const Layout::LayoutPreset& preset);
    std::string get_preset_file_path(const std::string& preset_name) const;

    // Helper methods for quick-save functionality
    std::string get_quick_save_filename(int slot_num) const;
};

}  // namespace UI
}  // namespace BTQuant