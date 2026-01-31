#pragma once

#include <string>
#include <vector>

namespace BTQuant {
namespace Layout {

// ============================================================================
// Layout Presets System
// ============================================================================

struct LayoutPreset {
    std::string name;
    std::string description;
    std::string category;  // "Trading", "Analysis", "Monitoring", etc.
    std::string json_data; // Serialized layout data
    std::string thumbnail_path; // Path to thumbnail image
    bool is_builtin;       // Whether this is a built-in preset
    std::string author;    // Author of the preset
    std::string version;   // Version of the preset
};

class LayoutPresetManager {
public:
    LayoutPresetManager();
    ~LayoutPresetManager();

    // Load all available presets
    void load_presets();
    
    // Save a custom preset
    bool save_preset(const std::string& name, const std::string& description, 
                     const std::string& json_data, const std::string& category = "Custom");
    
    // Apply a preset to the current layout
    bool apply_preset(const std::string& preset_name);
    
    // Delete a custom preset
    bool delete_preset(const std::string& preset_name);
    
    // Get all available presets
    std::vector<LayoutPreset> get_all_presets() const;
    
    // Get presets by category
    std::vector<LayoutPreset> get_presets_by_category(const std::string& category) const;
    
    // Export a preset to file
    bool export_preset(const std::string& preset_name, const std::string& file_path);
    
    // Import a preset from file
    bool import_preset(const std::string& file_path);
    
    // Create thumbnail for a preset
    bool create_thumbnail(const std::string& preset_name, const std::string& thumbnail_path);
    
    // Validate a preset
    bool validate_preset(const LayoutPreset& preset) const;

private:
    std::vector<LayoutPreset> presets_;
    std::string presets_directory_;
    
    void initialize_presets_directory();
    void load_builtin_presets();
    void load_user_presets();
    LayoutPreset load_preset_from_file(const std::string& file_path);
    bool save_preset_to_file(const LayoutPreset& preset);
    std::string get_preset_file_path(const std::string& preset_name) const;
};

} // namespace Layout
} // namespace BTQuant