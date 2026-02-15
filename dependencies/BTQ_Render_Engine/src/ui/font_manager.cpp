#include "../../include/ui/font_manager.hpp"

#include <imgui.h>
#include "backends/imgui_impl_vulkan.h"
// Note: misc/fonts/imgui_fonts_droid_sans.h not available in all ImGui builds
// Using AddFontDefault() instead which doesn't require external font data
#include <algorithm>
#include <iostream>

namespace BTQuant {
namespace UI {

FontManager::FontManager()
    : main_font_(nullptr),
      monospace_font_(nullptr),
      header_font_(nullptr),
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

    // Calculate DPI scaling factor based on display framebuffer scale
    float dpi_scale = io.DisplayFramebufferScale.x;
    if (dpi_scale < 1.0f) dpi_scale = 1.0f;  // Minimum scale of 1.0
    
    // Apply DPI scaling to font sizes
    float base_main_size = 16.0f;
    float base_mono_size = 14.0f;
    float base_header_size = 20.0f;
    
    float scaled_main_size = base_main_size * dpi_scale;
    float scaled_mono_size = base_mono_size * dpi_scale;
    float scaled_header_size = base_header_size * dpi_scale;

    // Configure main font
    ImFontConfig main_config;
    main_config.SizePixels = scaled_main_size;  // Scaled size for high-DPI
    main_config.OversampleH = 3;
    main_config.OversampleV = 3;
    main_config.PixelSnapH = true;

    // Load main font (using default for now, could be customized)
    main_font_ = io.Fonts->AddFontDefault(&main_config);
    if (!main_font_) {
        std::cerr << "Failed to load main font" << std::endl;
        return false;
    }

    // Configure monospace font for numerical displays
    ImFontConfig mono_config;
    mono_config.SizePixels = scaled_mono_size;  // Scaled size for high-DPI
    mono_config.OversampleH = 3;
    mono_config.OversampleV = 3;
    mono_config.PixelSnapH = true;
    strcpy(mono_config.Name, "Monospace##Custom");

    // Attempt to load a monospace font (fallback to default if unavailable)
    // Using a common monospace font that should be available
    monospace_font_ = io.Fonts->AddFontDefault(&mono_config);

    // If we wanted to load a specific TTF font file, we would do:
    // monospace_font_ = io.Fonts->AddFontFromFileTTF("path/to/monospace.ttf", scaled_mono_size, &mono_config);

    if (!monospace_font_) {
        std::cerr << "Failed to load monospace font, falling back to default" << std::endl;
        // Fallback: use the main font if monospace failed
        monospace_font_ = main_font_;
    }

    // Configure header font
    ImFontConfig header_config;
    header_config.SizePixels = scaled_header_size;  // Scaled size for high-DPI
    header_config.OversampleH = 3;
    header_config.OversampleV = 3;
    header_config.PixelSnapH = true;
    strcpy(header_config.Name, "Header##Custom");

    header_font_ = io.Fonts->AddFontDefault(&header_config);
    if (!header_font_) {
        std::cerr << "Failed to load header font, using main font as fallback" << std::endl;
        header_font_ = main_font_;
    }

    // Build the font atlas
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    is_initialized_ = true;
    return true;
}

void FontManager::updateFontScaling(float dpi_scale) {
    ImGuiIO& io = ImGui::GetIO();
    
    // If no DPI scale provided, calculate from framebuffer scale
    if (dpi_scale <= 0.0f) {
        dpi_scale = io.DisplayFramebufferScale.x;
        if (dpi_scale < 1.0f) dpi_scale = 1.0f;  // Minimum scale of 1.0
    }
    
    // Apply DPI scaling to font sizes
    float base_main_size = 16.0f;
    float base_mono_size = 14.0f;
    float base_header_size = 20.0f;
    
    float scaled_main_size = base_main_size * dpi_scale;
    float scaled_mono_size = base_mono_size * dpi_scale;
    float scaled_header_size = base_header_size * dpi_scale;

    // Clear existing fonts
    io.Fonts->Clear();

    // Reconfigure main font with new scale
    ImFontConfig main_config;
    main_config.SizePixels = scaled_main_size;  // Scaled size for high-DPI
    main_config.OversampleH = 3;
    main_config.OversampleV = 3;
    main_config.PixelSnapH = true;

    main_font_ = io.Fonts->AddFontDefault(&main_config);

    // Reconfigure monospace font with new scale
    ImFontConfig mono_config;
    mono_config.SizePixels = scaled_mono_size;  // Scaled size for high-DPI
    mono_config.OversampleH = 3;
    mono_config.OversampleV = 3;
    mono_config.PixelSnapH = true;
    strcpy(mono_config.Name, "Monospace##Custom");

    monospace_font_ = io.Fonts->AddFontDefault(&mono_config);

    // Reconfigure header font with new scale
    ImFontConfig header_config;
    header_config.SizePixels = scaled_header_size;  // Scaled size for high-DPI
    header_config.OversampleH = 3;
    header_config.OversampleV = 3;
    header_config.PixelSnapH = true;
    strcpy(header_config.Name, "Header##Custom");

    header_font_ = io.Fonts->AddFontDefault(&header_config);

    // Rebuild the font atlas and upload to GPU
    io.Fonts->Build();
    
    // For Vulkan backend, we need to manually rebuild and upload the font texture
    // This is typically done in the render loop, but we force it here
    ImGui_ImplVulkan_DestroyFontUploadObjects();
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