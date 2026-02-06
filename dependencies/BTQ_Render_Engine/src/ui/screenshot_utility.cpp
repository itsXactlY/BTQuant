#include "ui/screenshot_utility.hpp"
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>

#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace BTQuant {

bool ScreenshotUtility::capture_screenshot(const std::string& filename) {
    // Get the current viewport dimensions
    int width, height;
    int channels = 3; // RGB
    
    // Get the viewport size from OpenGL
    glGetIntegerv(GL_VIEWPORT, &width);
    glGetIntegerv(GL_VIEWPORT, &height);
    
    // Allocate memory for the pixel data
    int num_pixels = width * height * channels;
    unsigned char* pixels = new unsigned char[num_pixels];
    
    // Read the pixels from the framebuffer
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    
    // Flip the image vertically since OpenGL has origin at bottom-left
    // but image formats expect top-left origin
    flip_image_vertically(pixels, width, height, channels);
    
    // Save the image to file using stb_image_write
    bool success = stbi_write_png(filename.c_str(), width, height, channels, pixels, width * channels);
    
    // Clean up allocated memory
    delete[] pixels;
    
    if (success) {
        std::cout << "Screenshot saved to: " << filename << std::endl;
        return true;
    } else {
        std::cerr << "Failed to save screenshot: " << filename << std::endl;
        return false;
    }
}

bool ScreenshotUtility::capture_panel_screenshot(PanelBase* panel, const std::string& base_filename) {
    // Generate a unique filename with timestamp
    std::string timestamp = generate_timestamp();
    std::string filename = base_filename + "_" + timestamp + ".png";
    
    // Create screenshots directory if it doesn't exist
    create_screenshots_directory();
    
    // Capture the screenshot
    return capture_screenshot("screenshots/" + filename);
}

std::string ScreenshotUtility::generate_timestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

void ScreenshotUtility::flip_image_vertically(unsigned char* image, int width, int height, int channels) {
    int row_size = width * channels;
    unsigned char* temp_row = new unsigned char[row_size];
    
    for (int y = 0; y < height / 2; ++y) {
        int top_offset = y * row_size;
        int bottom_offset = (height - 1 - y) * row_size;
        
        // Swap the rows
        memcpy(temp_row, image + top_offset, row_size);
        memcpy(image + top_offset, image + bottom_offset, row_size);
        memcpy(image + bottom_offset, temp_row, row_size);
    }
    
    delete[] temp_row;
}

void ScreenshotUtility::create_screenshots_directory() {
#ifdef __linux__
    struct stat info;
    if (stat("screenshots", &info) != 0) {
        // Directory doesn't exist, create it
        if (mkdir("screenshots", 0755) == -1) {
            std::cerr << "Failed to create screenshots directory: " << strerror(errno) << std::endl;
        }
    }
#elif _WIN32
    if (_access("screenshots", 0) == -1) {
        // Directory doesn't exist, create it
        if (_mkdir("screenshots") == -1) {
            std::cerr << "Failed to create screenshots directory" << std::endl;
        }
    }
#else
    // For other platforms, try the POSIX approach
    struct stat info;
    if (stat("screenshots", &info) != 0) {
        if (mkdir("screenshots", 0755) == -1) {
            std::cerr << "Failed to create screenshots directory" << std::endl;
        }
    }
#endif
}

} // namespace BTQuant