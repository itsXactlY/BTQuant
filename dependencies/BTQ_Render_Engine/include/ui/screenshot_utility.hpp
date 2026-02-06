#pragma once

#include "../components/panel_base.hpp"
#include <string>

// Forward declaration of OpenGL functions if needed
#ifndef GL_RGB
#define GL_RGB 0x1907
#endif

#ifndef GL_UNSIGNED_BYTE
#define GL_UNSIGNED_BYTE 0x1401
#endif

// Forward declarations for OpenGL functions
extern "C" {
    void glGetIntegerv(unsigned int pname, int* params);
    void glReadPixels(int x, int y, int width, int height, unsigned int format, 
                     unsigned int type, void* pixels);
}

namespace BTQuant {

class ScreenshotUtility {
public:
    /**
     * @brief Captures a screenshot of the current OpenGL framebuffer
     * @param filename Path to save the captured screenshot
     * @return True if capture was successful, false otherwise
     */
    static bool capture_screenshot(const std::string& filename);
    
    /**
     * @brief Captures a screenshot of a specific panel
     * @param panel The panel to capture
     * @param base_filename Base filename for the screenshot
     * @return True if capture was successful, false otherwise
     */
    static bool capture_panel_screenshot(PanelBase* panel, const std::string& base_filename = "panel_screenshot");
    
private:
    /**
     * @brief Generates a timestamp string for unique filenames
     * @return Timestamp string in YYYYMMDD_HHMMSS format
     */
    static std::string generate_timestamp();
    
    /**
     * @brief Flips an image vertically (because OpenGL has origin at bottom-left)
     * @param image Pointer to image data
     * @param width Image width
     * @param height Image height
     * @param channels Number of color channels
     */
    static void flip_image_vertically(unsigned char* image, int width, int height, int channels);
    
    /**
     * @brief Creates the screenshots directory if it doesn't exist
     */
    static void create_screenshots_directory();
};

} // namespace BTQuant