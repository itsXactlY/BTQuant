#include "../../include/components/theme_customization_component.hpp"
#include <iostream>

namespace BTQuant {

ThemeCustomizationComponent::ThemeCustomizationComponent(RenderEngine::DashboardConfig& config)
    : UIComponent({0, 0}, {0, 0}), config_(config) {
    // Initialize custom theme with current theme settings
    custom_theme_ = config_.getThemeConfig();
    load_available_layouts();
}

void ThemeCustomizationComponent::update(float dt) {
    (void)dt;
}

void ThemeCustomizationComponent::render_gui() {
    // Render theme and layout management in menu bar
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Order Flow", nullptr, false);
            ImGui::MenuItem("Market Depth", nullptr, false);
            ImGui::Separator();
            ImGui::MenuItem("Theme Settings", nullptr, &show_theme_window_);
            ImGui::MenuItem("Layout Manager", nullptr, &show_layout_window_);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // Render theme customization window
    if (show_theme_window_) {
        render_theme_window();
    }

    // Render layout management window
    if (show_layout_window_) {
        render_layout_window();
    }
}

void ThemeCustomizationComponent::render_theme_window() {
    ImGui::Begin("Theme Settings", &show_theme_window_, ImGuiWindowFlags_AlwaysAutoResize);
    
    // Theme selection
    ImGui::Text("Theme Presets");
    ImGui::Separator();
    
    auto available_themes = config_.getAvailableThemes();
    for (const auto& theme_name : available_themes) {
        if (ImGui::RadioButton(theme_name.c_str(), config_.getThemeConfig().name == theme_name)) {
            config_.applyTheme(theme_name);
            if (theme_name == "custom") {
                custom_theme_ = config_.getThemeConfig();
            }
        }
    }
    
    ImGui::Separator();
    ImGui::Text("Custom Theme");
    ImGui::Separator();
    
    // Color customization
    ImGui::ColorEdit4("Background", &custom_theme_.background_color.r, ImGuiColorEditFlags_NoAlpha);
    ImGui::ColorEdit4("Text", &custom_theme_.text_color.r, ImGuiColorEditFlags_NoAlpha);
    ImGui::ColorEdit4("Accent", &custom_theme_.accent_color.r, ImGuiColorEditFlags_NoAlpha);
    ImGui::ColorEdit4("Positive", &custom_theme_.positive_color.r, ImGuiColorEditFlags_NoAlpha);
    ImGui::ColorEdit4("Negative", &custom_theme_.negative_color.r, ImGuiColorEditFlags_NoAlpha);
    ImGui::ColorEdit4("Neutral", &custom_theme_.neutral_color.r, ImGuiColorEditFlags_NoAlpha);
    ImGui::ColorEdit4("Grid Lines", &custom_theme_.grid_line_color.r, ImGuiColorEditFlags_NoAlpha);
    
    // Font settings
    ImGui::SliderInt("Font Size", &custom_theme_.font_size, 8, 24);
    ImGui::SliderFloat("Line Height", &custom_theme_.line_height, 0.8f, 2.0f);
    
    // Theme name
    ImGui::InputText("Theme Name", custom_theme_name_, sizeof(custom_theme_name_));
    
    // Apply and save buttons
    if (ImGui::Button("Apply Custom Theme")) {
        custom_theme_.name = custom_theme_name_;
        config_.setThemeConfig(custom_theme_);
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("Save Custom Theme")) {
        save_custom_theme();
    }
    
    ImGui::Separator();
    
    if (ImGui::Button("Reset to Default")) {
        config_.resetToDefaults();
        custom_theme_ = config_.getThemeConfig();
        strncpy(custom_theme_name_, "Custom", sizeof(custom_theme_name_));
    }
    
    ImGui::End();
}

void ThemeCustomizationComponent::render_layout_window() {
    ImGui::Begin("Layout Manager", &show_layout_window_, ImGuiWindowFlags_AlwaysAutoResize);
    
    // Load available layouts
    if (ImGui::Button("Refresh Layouts")) {
        load_available_layouts();
    }
    
    ImGui::Separator();
    
    // Layout selection
    ImGui::Text("Available Layouts");
    ImGui::Separator();
    
    if (ImGui::ListBox("Layouts", &selected_layout_index_, 
                      [](void* data, int idx, const char** out_text) {
                          auto& layouts = *static_cast<std::vector<std::string>*>(data);
                          if (idx >= 0 && idx < static_cast<int>(layouts.size())) {
                              *out_text = layouts[idx].c_str();
                          }
                          return true;
                      }, 
                      &available_layouts_, available_layouts_.size(), 10)) {
        if (selected_layout_index_ >= 0 && selected_layout_index_ < static_cast<int>(available_layouts_.size())) {
            config_.loadLayout(available_layouts_[selected_layout_index_]);
        }
    }
    
    ImGui::Separator();
    
    // Save new layout
    ImGui::Text("Save Current Layout");
    ImGui::Separator();
    
    ImGui::InputText("Layout Name", new_layout_name_, sizeof(new_layout_name_));
    
    if (ImGui::Button("Save Layout")) {
        if (strlen(new_layout_name_) > 0) {
            config_.saveLayout(new_layout_name_);
            load_available_layouts();
            // Select the newly saved layout
            for (size_t i = 0; i < available_layouts_.size(); ++i) {
                if (available_layouts_[i] == new_layout_name_) {
                    selected_layout_index_ = static_cast<int>(i);
                    break;
                }
            }
            new_layout_name_[0] = '\0';
        }
    }
    
    ImGui::Separator();
    
    // Delete selected layout
    if (ImGui::Button("Delete Selected Layout")) {
        if (selected_layout_index_ >= 0 && selected_layout_index_ < static_cast<int>(available_layouts_.size())) {
            std::string layout_to_delete = available_layouts_[selected_layout_index_];
            if (config_.deleteLayout(layout_to_delete)) {
                load_available_layouts();
                selected_layout_index_ = 0;
            }
        }
    }
    
    ImGui::Separator();
    
    if (ImGui::Button("Reset to Default Layout")) {
        config_.loadLayout("default");
        load_available_layouts();
        selected_layout_index_ = 0;
    }
    
    ImGui::End();
}

void ThemeCustomizationComponent::apply_current_theme() {
    config_.setThemeConfig(custom_theme_);
}

void ThemeCustomizationComponent::save_custom_theme() {
    custom_theme_.name = custom_theme_name_;
    config_.setThemeConfig(custom_theme_);
    config_.saveConfiguration();
}

void ThemeCustomizationComponent::load_available_layouts() {
    available_layouts_ = config_.getAvailableLayouts();
    selected_layout_index_ = 0;
}

} // namespace BTQuant
