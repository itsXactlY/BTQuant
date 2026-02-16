#include "../../include/ui/font_manager.hpp"

#include <imgui.h>
// Note: misc/fonts/imgui_fonts/imgui_fonts_droid_sans.h not available in all ImGui builds
// Using AddFontDefault() instead which doesn't require external font data
#include <algorithm>
#include <iostream>
#include <filesystem>

// FontAwesome 6 icon ranges (solid style)
// FontAwesome 6 Free Solid icons use Unicode range U+F000 to U+F2E0
#ifndef ICON_MIN_FA
#define ICON_MIN_FA 0xF000
#endif
#ifndef ICON_MAX_FA
#define ICON_MAX_FA 0xF2E0
#endif

static const ImWchar fa_icon_ranges[] = {
    ICON_MIN_FA, ICON_MAX_FA, 0
};

namespace BTQuant {
namespace UI {

FontManager::FontManager()
    : main_font_(nullptr),
      monospace_font_(nullptr),
      header_font_(nullptr),
      icon_font_(nullptr),
      is_initialized_(false) {
}

FontManager& FontManager::getInstance() {
    static FontManager instance;
    return instance;
}

bool FontManager::initialize() {
    if (is_initialized_) {
        return true;  // Already initialized
    }

    ImGuiIO& io = ImGui::GetIO();

    // Clear any existing fonts to start fresh
    io.Fonts->Clear();

    // Configure main font
    ImFontConfig main_config;
    main_config.SizePixels = 16.0f;  // Standard size for main text
    main_config.OversampleH = 3;
    main_config.OversampleV = 3;
    main_config.PixelSnapH = true;

    // Load main font (using default for now, could be customized)
    main_font_ = io.Fonts->AddFontDefault(&main_config);
    if (!main_font_) {
        std::cerr << "Failed to load main font" << std::endl;
        return false;
    }

    // Configure monospace font for numerical displays (JetBrains Mono / Berkeley Mono)
    ImFontConfig mono_config;
    mono_config.SizePixels = 14.0f;  // Slightly smaller for dense info
    mono_config.OversampleH = 4;
    mono_config.OversampleV = 4;
    mono_config.PixelSnapH = false;
    strcpy(mono_config.Name, "Monospace##Custom");

    // Attempt to load a monospace font (fallback to default if unavailable)
    // Using a common monospace font that should be available
    monospace_font_ = io.Fonts->AddFontDefault(&mono_config);

    // If we wanted to load a specific TTF font file, we would do:
    // monospace_font_ = io.Fonts->AddFontFromFileTTF("path/to/monospace.ttf", 14.0f, &mono_config);

    if (!monospace_font_) {
        std::cerr << "Failed to load monospace font, falling back to default" << std::endl;
        // Fallback: use the main font if monospace failed
        monospace_font_ = main_font_;
    }

    // Configure header font
    ImFontConfig header_config;
    header_config.SizePixels = 20.0f;  // Larger for headers
    header_config.OversampleH = 3;
    header_config.OversampleV = 3;
    header_config.PixelSnapH = true;
    strcpy(header_config.Name, "Header##Custom");

    header_font_ = io.Fonts->AddFontDefault(&header_config);
    if (!header_font_) {
        std::cerr << "Failed to load header font, using main font as fallback" << std::endl;
        header_font_ = main_font_;
    }

    // Load FontAwesome 6 icons merged into the same font atlas
    // This prevents texture swapping during render passes
    ImFontConfig icon_config;
    icon_config.MergeMode = true;  // Merge into the same atlas as the main font
    icon_config.PixelSnapH = true;
    icon_config.OversampleH = 3;
    icon_config.OversampleV = 3;

    // Try multiple possible paths for the FontAwesome 6 font file
    std::vector<std::string> possible_paths = {
        "fonts/fa-solid-6.ttf",
        "../fonts/fa-solid-6.ttf",
        "../../fonts/fa-solid-6.ttf",
        "../../../fonts/fa-solid-6.ttf"
#ifdef BTQ_FONT_PATH
        , BTQ_FONT_PATH "/fa-solid-6.ttf"
#endif
    };

    std::string found_path;
    for (const auto& path : possible_paths) {
        if (std::filesystem::exists(path)) {
            found_path = path;
            break;
        }
    }

    if (!found_path.empty()) {
        // Load FontAwesome 6 icons merged into the same atlas
        icon_font_ = io.Fonts->AddFontFromFileTTF(
            found_path.c_str(),
            16.0f,  // Match main font size for consistent rendering
            &icon_config,
            fa_icon_ranges
        );

        if (icon_font_) {
            std::cout << "FontAwesome 6 icons loaded from: " << found_path << std::endl;
        } else {
            std::cerr << "Failed to load FontAwesome 6 font, icons will not be available" << std::endl;
        }
    } else {
        std::cerr << "FontAwesome 6 font file not found. Icons will not be available." << std::endl;
        std::cerr << "Expected font at one of: fonts/fa-solid-6.ttf" << std::endl;
    }

    // Build the font atlas (all fonts including FontAwesome are now in the same texture)
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    is_initialized_ = true;
    return true;
}

ImFont* FontManager::getMainFont() const {
    return main_font_;
}

ImFont* FontManager::getMonospaceFont() const {
    return monospace_font_;
}

ImFont* FontManager::getHeaderFont() const {
    return header_font_;
}

ImFont* FontManager::getIconFont() const {
    return icon_font_;
}

void FontManager::pushMonospaceFont() const {
    if (monospace_font_) {
        ImGui::PushFont(monospace_font_);
    }
}

void FontManager::popFont() const {
    ImGui::PopFont();
}

void FontManager::renderWithMonospaceFont(const std::function<void()>& render_fn) const {
    if (!render_fn) return;
    
    if (monospace_font_) {
        ImGui::PushFont(monospace_font_);
        render_fn();
        ImGui::PopFont();
    } else {
        render_fn();  // Execute without font change if monospace font unavailable
    }
}

void FontManager::renderWithNumericalFont(const std::function<void()>& render_fn) const {
    // For numerical displays, we use the same monospace font
    renderWithMonospaceFont(render_fn);
}

void FontManager::renderNumericalValue(float value, const char* format) const {
    if (monospace_font_) {
        ImGui::PushFont(monospace_font_);
        ImGui::Text(format, value);
        ImGui::PopFont();
    } else {
        ImGui::Text(format, value);
    }
}

void FontManager::renderNumericalValue(int value) const {
    if (monospace_font_) {
        ImGui::PushFont(monospace_font_);
        ImGui::Text("%d", value);
        ImGui::PopFont();
    } else {
        ImGui::Text("%d", value);
    }
}

void FontManager::renderFormattedNumericalValue(double value, const char* format) const {
    if (monospace_font_) {
        ImGui::PushFont(monospace_font_);
        ImGui::Text(format, value);
        ImGui::PopFont();
    } else {
        ImGui::Text(format, value);
    }
}

void FontManager::renderFormattedNumericalValue(float value, const char* format) const {
    if (monospace_font_) {
        ImGui::PushFont(monospace_font_);
        ImGui::Text(format, value);
        ImGui::PopFont();
    } else {
        ImGui::Text(format, value);
    }
}

bool FontManager::isInitialized() const {
    return is_initialized_;
}

}  // namespace UI
}  // namespace BTQuant