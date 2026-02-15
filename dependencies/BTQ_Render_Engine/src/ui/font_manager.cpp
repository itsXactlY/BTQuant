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
      icons_font_(nullptr),
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
    main_config.OversampleH = 4;  // Increase oversampling to eliminate sub-pixel aliasing
    main_config.OversampleV = 4;  // Increase oversampling to eliminate sub-pixel aliasing
    main_config.PixelSnapH = true;

    // Load main font (using default for now, could be customized)
    main_font_ = io.Fonts->AddFontDefault(&main_config);
    if (!main_font_) {
        std::cerr << "Failed to load main font" << std::endl;
        return false;
    }

    // Configure monospace font for numerical displays with high oversampling for crisp text on high-DPI displays
    ImFontConfig mono_config;
    mono_config.SizePixels = scaled_mono_size;  // Scaled size for high-DPI
    mono_config.OversampleH = 4;  // Critical for eliminating sub-pixel aliasing on high-DPI screens
    mono_config.OversampleV = 4;  // Critical for eliminating sub-pixel aliasing on high-DPI screens
    mono_config.PixelSnapH = true;
    strcpy(mono_config.Name, "JetBrainsMono##Custom");

    // Attempt to load JetBrains Mono font with specific weights and styles (try multiple possible locations)
    // Prioritize system-wide installations first
    monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf", scaled_mono_size, &mono_config);
    
    // Try common variations of JetBrains Mono filename
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf", scaled_mono_size, &mono_config);
    }
    
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf", scaled_mono_size, &mono_config);
    }

    // If not in system fonts, try common Linux installation paths
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf", scaled_mono_size, &mono_config);
    }
    
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/TTF/JetBrainsMono-Bold.ttf", scaled_mono_size, &mono_config);
    }

    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/local/share/fonts/TTF/JetBrainsMono-Regular.ttf", scaled_mono_size, &mono_config);
    }
    
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/local/share/fonts/TTF/JetBrainsMono-Bold.ttf", scaled_mono_size, &mono_config);
    }

    // If not in system locations, try project resources
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("./resources/fonts/JetBrainsMono-Regular.ttf", scaled_mono_size, &mono_config);
    }
    
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("./resources/fonts/JetBrainsMono-Bold.ttf", scaled_mono_size, &mono_config);
    }

    // If not in project resources, try user-specific locations
    if (!monospace_font_) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string user_font_path = std::string(home_dir) + "/.local/share/fonts/JetBrainsMono-Regular.ttf";
            monospace_font_ = io.Fonts->AddFontFromFileTTF(user_font_path.c_str(), scaled_mono_size, &mono_config);
        }
    }
    
    if (!monospace_font_) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string user_font_path = std::string(home_dir) + "/.local/share/fonts/JetBrainsMono-Bold.ttf";
            monospace_font_ = io.Fonts->AddFontFromFileTTF(user_font_path.c_str(), scaled_mono_size, &mono_config);
        }
    }

    // If not in user-specific location, try alternative user locations
    if (!monospace_font_) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string user_font_path = std::string(home_dir) + "/.fonts/JetBrainsMono-Regular.ttf";
            monospace_font_ = io.Fonts->AddFontFromFileTTF(user_font_path.c_str(), scaled_mono_size, &mono_config);
        }
    }
    
    if (!monospace_font_) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string user_font_path = std::string(home_dir) + "/.fonts/JetBrainsMono-Bold.ttf";
            monospace_font_ = io.Fonts->AddFontFromFileTTF(user_font_path.c_str(), scaled_mono_size, &mono_config);
        }
    }

    // If JetBrains Mono isn't available, try other popular monospace fonts with high oversampling
    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "DejaVuSansMono##Custom");
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", scaled_mono_size, &fallback_config);
    }

    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "DejaVuSansMono-Bold##Custom");
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", scaled_mono_size, &fallback_config);
    }

    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "UbuntuMono##Custom");
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", scaled_mono_size, &fallback_config);
    }

    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "LiberationMono##Custom");
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", scaled_mono_size, &fallback_config);
    }

    // If JetBrains Mono isn't available, fall back to default monospace font with same high oversampling
    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "DefaultMonospace##Custom");
        monospace_font_ = io.Fonts->AddFontDefault(&fallback_config);
        std::cout << "Warning: JetBrains Mono font not found, using default monospace font with oversampling H=4, V=4" << std::endl;
    }

    if (!monospace_font_) {
        std::cerr << "Failed to load monospace font, falling back to default" << std::endl;
        // Fallback: use the main font if monospace failed
        monospace_font_ = main_font_;
    }

    // Configure header font
    ImFontConfig header_config;
    header_config.SizePixels = scaled_header_size;  // Scaled size for high-DPI
    header_config.OversampleH = 4;  // Increase oversampling to eliminate sub-pixel aliasing
    header_config.OversampleV = 4;  // Increase oversampling to eliminate sub-pixel aliasing
    header_config.PixelSnapH = true;
    strcpy(header_config.Name, "Header##Custom");

    header_font_ = io.Fonts->AddFontDefault(&header_config);
    if (!header_font_) {
        std::cerr << "Failed to load header font, using main font as fallback" << std::endl;
        header_font_ = main_font_;
    }

    // Configure FontAwesome 6 font configuration for UI iconography
    ImFontConfig icons_config;
    icons_config.SizePixels = scaled_main_size;  // Same size as main font for consistency
    icons_config.OversampleH = 4;  // High oversampling for crisp icons on high-DPI displays
    icons_config.OversampleV = 4;  // High oversampling for crisp icons on high-DPI displays
    icons_config.PixelSnapH = true;
    strcpy(icons_config.Name, "FontAwesome6##Icons");

    // Define the range of icons to include
    static const ImWchar icons_ranges[] = { 
        0xE000, 0xF8FF, // FontAwesome icons range
        0
    };

    // Attempt to load FontAwesome 6 font (try multiple possible locations)
    ImFont* icons_font = nullptr;
    
    // Try common system locations for FontAwesome 6
    icons_font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/fontawesome/Font Awesome 6 Free-Regular-400.otf", scaled_main_size, &icons_config, icons_ranges);
    
    if (!icons_font) {
        icons_font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/fontawesome/Font-Awesome-6-Free-Solid-900.otf", scaled_main_size, &icons_config, icons_ranges);
    }
    
    if (!icons_font) {
        icons_font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/TTF/Font Awesome 6 Brands-Regular-400.otf", scaled_main_size, &icons_config, icons_ranges);
    }
    
    // Try common alternative locations
    if (!icons_font) {
        icons_font = io.Fonts->AddFontFromFileTTF("./resources/fonts/Font Awesome 6 Free-Regular-400.otf", scaled_main_size, &icons_config, icons_ranges);
    }
    
    if (!icons_font) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string icons_font_path = std::string(home_dir) + "/.local/share/fonts/Font Awesome 6 Free-Regular-400.otf";
            icons_font = io.Fonts->AddFontFromFileTTF(icons_font_path.c_str(), scaled_main_size, &icons_config, icons_ranges);
        }
    }
    
    if (!icons_font) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string icons_font_path = std::string(home_dir) + "/.fonts/Font Awesome 6 Free-Regular-400.otf";
            icons_font = io.Fonts->AddFontFromFileTTF(icons_font_path.c_str(), scaled_main_size, &icons_config, icons_ranges);
        }
    }
    
    // If FontAwesome 6 is not available, log a warning but continue
    if (!icons_font) {
        std::cout << "Warning: FontAwesome 6 font not found, UI iconography may not be available" << std::endl;
    } else {
        // Assign the loaded icons font to our member variable
        icons_font_ = icons_font;
    }

    // Build the font atlas
    io.Fonts->Build();
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
    main_config.OversampleH = 4;  // Increase oversampling to eliminate sub-pixel aliasing
    main_config.OversampleV = 4;  // Increase oversampling to eliminate sub-pixel aliasing
    main_config.PixelSnapH = true;

    main_font_ = io.Fonts->AddFontDefault(&main_config);

    // Reconfigure monospace font with new scale - ensure high oversampling for crisp text on high-DPI displays
    ImFontConfig mono_config;
    mono_config.SizePixels = scaled_mono_size;  // Scaled size for high-DPI
    mono_config.OversampleH = 4;  // Critical for eliminating sub-pixel aliasing on high-DPI screens
    mono_config.OversampleV = 4;  // Critical for eliminating sub-pixel aliasing on high-DPI screens
    mono_config.PixelSnapH = true;
    strcpy(mono_config.Name, "JetBrainsMono##Custom");

    // Attempt to load JetBrains Mono font with specific weights and styles (try multiple possible locations)
    // Prioritize system-wide installations first
    monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf", scaled_mono_size, &mono_config);
    
    // Try common variations of JetBrains Mono filename
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf", scaled_mono_size, &mono_config);
    }
    
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf", scaled_mono_size, &mono_config);
    }

    // If not in system fonts, try common Linux installation paths
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf", scaled_mono_size, &mono_config);
    }
    
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/TTF/JetBrainsMono-Bold.ttf", scaled_mono_size, &mono_config);
    }

    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/local/share/fonts/TTF/JetBrainsMono-Regular.ttf", scaled_mono_size, &mono_config);
    }
    
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/local/share/fonts/TTF/JetBrainsMono-Bold.ttf", scaled_mono_size, &mono_config);
    }

    // If not in system locations, try project resources
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("./resources/fonts/JetBrainsMono-Regular.ttf", scaled_mono_size, &mono_config);
    }
    
    if (!monospace_font_) {
        monospace_font_ = io.Fonts->AddFontFromFileTTF("./resources/fonts/JetBrainsMono-Bold.ttf", scaled_mono_size, &mono_config);
    }

    // If not in project resources, try user-specific locations
    if (!monospace_font_) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string user_font_path = std::string(home_dir) + "/.local/share/fonts/JetBrainsMono-Regular.ttf";
            monospace_font_ = io.Fonts->AddFontFromFileTTF(user_font_path.c_str(), scaled_mono_size, &mono_config);
        }
    }
    
    if (!monospace_font_) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string user_font_path = std::string(home_dir) + "/.local/share/fonts/JetBrainsMono-Bold.ttf";
            monospace_font_ = io.Fonts->AddFontFromFileTTF(user_font_path.c_str(), scaled_mono_size, &mono_config);
        }
    }

    // If not in user-specific location, try alternative user locations
    if (!monospace_font_) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string user_font_path = std::string(home_dir) + "/.fonts/JetBrainsMono-Regular.ttf";
            monospace_font_ = io.Fonts->AddFontFromFileTTF(user_font_path.c_str(), scaled_mono_size, &mono_config);
        }
    }
    
    if (!monospace_font_) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string user_font_path = std::string(home_dir) + "/.fonts/JetBrainsMono-Bold.ttf";
            monospace_font_ = io.Fonts->AddFontFromFileTTF(user_font_path.c_str(), scaled_mono_size, &mono_config);
        }
    }

    // If JetBrains Mono isn't available, try other popular monospace fonts with high oversampling
    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "DejaVuSansMono##Custom");
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", scaled_mono_size, &fallback_config);
    }

    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "DejaVuSansMono-Bold##Custom");
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", scaled_mono_size, &fallback_config);
    }

    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "UbuntuMono##Custom");
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", scaled_mono_size, &fallback_config);
    }

    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "LiberationMono##Custom");
        monospace_font_ = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", scaled_mono_size, &fallback_config);
    }

    // If JetBrains Mono isn't available, fall back to default monospace font with same high oversampling
    if (!monospace_font_) {
        ImFontConfig fallback_config = mono_config;
        strcpy(fallback_config.Name, "DefaultMonospace##Custom");
        monospace_font_ = io.Fonts->AddFontDefault(&fallback_config);
        std::cout << "Warning: JetBrains Mono font not found (during font scaling update), using default monospace font with oversampling H=4, V=4" << std::endl;
    }

    // Reconfigure header font with new scale
    ImFontConfig header_config;
    header_config.SizePixels = scaled_header_size;  // Scaled size for high-DPI
    header_config.OversampleH = 4;  // Increase oversampling to eliminate sub-pixel aliasing
    header_config.OversampleV = 4;  // Increase oversampling to eliminate sub-pixel aliasing
    header_config.PixelSnapH = true;
    strcpy(header_config.Name, "Header##Custom");

    header_font_ = io.Fonts->AddFontDefault(&header_config);

    // Configure FontAwesome 6 font configuration for UI iconography
    ImFontConfig icons_config;
    icons_config.SizePixels = scaled_main_size;  // Same size as main font for consistency
    icons_config.OversampleH = 4;  // High oversampling for crisp icons on high-DPI displays
    icons_config.OversampleV = 4;  // High oversampling for crisp icons on high-DPI displays
    icons_config.PixelSnapH = true;
    strcpy(icons_config.Name, "FontAwesome6##Icons");

    // Define the range of icons to include
    static const ImWchar icons_ranges[] = { 
        0xE000, 0xF8FF, // FontAwesome icons range
        0
    };

    // Attempt to load FontAwesome 6 font (try multiple possible locations)
    ImFont* icons_font = nullptr;
    
    // Try common system locations for FontAwesome 6
    icons_font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/fontawesome/Font Awesome 6 Free-Regular-400.otf", scaled_main_size, &icons_config, icons_ranges);
    
    if (!icons_font) {
        icons_font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/fontawesome/Font-Awesome-6-Free-Solid-900.otf", scaled_main_size, &icons_config, icons_ranges);
    }
    
    if (!icons_font) {
        icons_font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/TTF/Font Awesome 6 Brands-Regular-400.otf", scaled_main_size, &icons_config, icons_ranges);
    }
    
    // Try common alternative locations
    if (!icons_font) {
        icons_font = io.Fonts->AddFontFromFileTTF("./resources/fonts/Font Awesome 6 Free-Regular-400.otf", scaled_main_size, &icons_config, icons_ranges);
    }
    
    if (!icons_font) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string icons_font_path = std::string(home_dir) + "/.local/share/fonts/Font Awesome 6 Free-Regular-400.otf";
            icons_font = io.Fonts->AddFontFromFileTTF(icons_font_path.c_str(), scaled_main_size, &icons_config, icons_ranges);
        }
    }
    
    if (!icons_font) {
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            std::string icons_font_path = std::string(home_dir) + "/.fonts/Font Awesome 6 Free-Regular-400.otf";
            icons_font = io.Fonts->AddFontFromFileTTF(icons_font_path.c_str(), scaled_main_size, &icons_config, icons_ranges);
        }
    }
    
    // If FontAwesome 6 is not available, log a warning but continue
    if (!icons_font) {
        std::cout << "Warning: FontAwesome 6 font not found (during font scaling update), UI iconography may not be available" << std::endl;
    } else {
        // Assign the loaded icons font to our member variable
        icons_font_ = icons_font;
    }

    // Rebuild the font atlas
    io.Fonts->Build();
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

ImFont* FontManager::getIconsFont() const {
    return icons_font_;
}

void FontManager::renderIcon(const char* icon_code) const {
    if (icons_font_) {
        ImGui::PushFont(icons_font_);
        ImGui::Text("%s", icon_code);
        ImGui::PopFont();
    } else {
        // If icons font is not available, just render the text
        ImGui::Text("%s", icon_code);
    }
}

void FontManager::pushIconsFont() const {
    if (icons_font_) {
        ImGui::PushFont(icons_font_);
    }
}

}  // namespace UI
}  // namespace BTQuant