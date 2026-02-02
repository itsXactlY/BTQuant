#pragma once

#include <imgui.h>
#include <functional>
#include <string>

namespace BTQuant {
namespace UI {

/**
 * @brief Manages font loading and provides consistent font access across the application
 * 
 * The FontManager ensures consistent use of monospace fonts for numerical displays
 * to improve readability and alignment in trading data presentations.
 */
class FontManager {
public:
    /**
     * @brief Get the singleton instance of the FontManager
     */
    static FontManager& getInstance();

    /**
     * @brief Initialize the font manager and load all required fonts
     * @return true if initialization was successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Get the main application font
     */
    ImFont* getMainFont() const;

    /**
     * @brief Get the monospace font for numerical displays
     */
    ImFont* getMonospaceFont() const;

    /**
     * @brief Get the header font
     */
    ImFont* getHeaderFont() const;

    /**
     * @brief Push the monospace font onto the ImGui font stack
     */
    void pushMonospaceFont() const;

    /**
     * @brief Pop the current font from the ImGui font stack
     */
    void popFont() const;

    /**
     * @brief Execute a render function with the monospace font applied
     * @param render_fn The function to execute with monospace font applied
     */
    void renderWithMonospaceFont(const std::function<void()>& render_fn) const;

    /**
     * @brief Execute a render function with the monospace font specifically for numerical displays
     * @param render_fn The function to execute with numerical display font applied
     */
    void renderWithNumericalFont(const std::function<void()>& render_fn) const;

    /**
     * @brief Render a numerical value with consistent monospace formatting
     * @param value The numerical value to render
     * @param format The printf-style format string for the value
     */
    void renderNumericalValue(float value, const char* format = "%.2f") const;

    /**
     * @brief Render a numerical value with consistent monospace formatting
     * @param value The integer value to render
     */
    void renderNumericalValue(int value) const;

    /**
     * @brief Check if the font manager has been initialized
     */
    bool isInitialized() const;

private:
    FontManager();  // Private constructor for singleton
    
    ImFont* main_font_;
    ImFont* monospace_font_;
    ImFont* header_font_;
    bool is_initialized_;
};

}  // namespace UI
}  // namespace BTQuant