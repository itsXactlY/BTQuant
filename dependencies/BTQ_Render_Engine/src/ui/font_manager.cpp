#include "../../include/ui/font_manager.hpp"

#include <imgui.h>
#include <misc/freetype/imgui_freetype.h>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace BTQuant {
namespace UI {

FontManager::FontManager() 
    : main_font_(nullptr)
    , monospace_font_(nullptr)
    , header_font_(nullptr)
    , is_initialized_(false) {
}

FontManager& FontManager::getInstance() {
    static FontManager instance;
    return instance;
}

bool FontManager::initialize() {
    if (is_initialized_) {
        return true;
    }

    ImGuiIO& io = ImGui::GetIO();

    // Clear any existing fonts
    io.Fonts->Clear();

    // Enable FreeType for better font rendering if available
    #ifdef IMGUI_ENABLE_FREETYPE
        ImGuiFreeType::BuildFontAtlas(io.Fonts, 0);
    #endif

    // Configure main font
    ImFontConfig main_config;
    main_config.SizePixels = 14.0f;
    main_config.OversampleH = 3;
    main_config.OversampleV = 3;
    main_config.PixelSnapH = true;
    strcpy(main_config.Name, "MainFont");

    // Try to load a custom font, fallback to default if not available
    std::string main_font_path = "fonts/Roboto-Regular.ttf";
    if (std::filesystem::exists(main_font_path)) {
        main_font_ = io.Fonts->AddFontFromFileTTF(main_font_path.c_str(), 14.0f, &main_config);
    } else {
        main_font_ = io.Fonts->AddFontDefault(&main_config);
    }

    // Configure header font (larger size)
    ImFontConfig header_config;
    header_config.SizePixels = 16.0f;
    header_config.OversampleH = 3;
    header_config.OversampleV = 3;
    header_config.PixelSnapH = true;
    strcpy(header_config.Name, "HeaderFont");

    std::string header_font_path = "fonts/Roboto-Bold.ttf";
    if (std::filesystem::exists(header_font_path)) {
        header_font_ = io.Fonts->AddFontFromFileTTF(header_font_path.c_str(), 16.0f, &header_config);
    } else {
        header_font_ = io.Fonts->AddFontDefault(&header_config);
    }

    // Configure monospace font for numerical displays and code
    ImFontConfig mono_config;
    mono_config.SizePixels = 13.0f;
    mono_config.OversampleH = 3;
    mono_config.OversampleV = 3;
    mono_config.PixelSnapH = true;
    strcpy(mono_config.Name, "MonoFont");

    // Try to load a monospace font - prioritize common monospace fonts optimized for numerical data
    std::vector<std::string> monospace_fonts = {
        "fonts/JetBrainsMono-Regular.ttf",      // Excellent for numerical data, clear digit differentiation
        "fonts/RobotoMono-Regular.ttf",         // Good for numerical displays
        "fonts/FiraCode-Regular.ttf",           // Programming font with good digit clarity
        "fonts/SourceCodePro-Regular.ttf",      // Clean monospace for data
        "fonts/Consolas.ttf",                   // Standard programming font
        "fonts/CourierNew.ttf",                 // Classic monospace
        "fonts/DejaVuSansMono.ttf"              // Reliable fallback
    };

    bool font_loaded = false;
    for (const auto& font_path : monospace_fonts) {
        if (std::filesystem::exists(font_path)) {
            monospace_font_ = io.Fonts->AddFontFromFileTTF(font_path.c_str(), 13.0f, &mono_config);
            font_loaded = true;
            break;
        }
    }

    // If no monospace font file found, create a default one but mark it specially
    if (!font_loaded) {
        monospace_font_ = io.Fonts->AddFontDefault(&mono_config);
        std::cout << "[FontManager] Warning: No monospace font file found, using default. Consider adding JetBrainsMono or RobotoMono." << std::endl;
    }

    // Build the font atlas
    #ifdef IMGUI_ENABLE_FREETYPE
        io.Fonts->Build();  // FreeType builds the atlas differently
    #else
        io.Fonts->Build();
    #endif

    is_initialized_ = true;

    std::cout << "[FontManager] Initialized successfully with monospace font for numerical displays." << std::endl;
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

void FontManager::pushMonospaceFont() const {
    if (monospace_font_) {
        ImGui::PushFont(monospace_font_);
    }
}

void FontManager::popFont() const {
    ImGui::PopFont();
}

void FontManager::renderWithMonospaceFont(const std::function<void()>& render_fn) const {
    if (monospace_font_ && render_fn) {
        ImGui::PushFont(monospace_font_);
        render_fn();
        ImGui::PopFont();
    } else if (render_fn) {
        render_fn();
    }
}

void FontManager::renderWithNumericalFont(const std::function<void()>& render_fn) const {
    if (monospace_font_ && render_fn) {
        ImGui::PushFont(monospace_font_);
        render_fn();
        ImGui::PopFont();
    } else if (render_fn) {
        render_fn();
    }
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

bool FontManager::isInitialized() const {
    return is_initialized_;
}

}  // namespace UI
}  // namespace BTQuant