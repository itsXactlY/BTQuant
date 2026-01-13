#define VK_USE_PLATFORM_XLIB_KHR
#include "hotspine_vulkan_demo.hpp"

int main() {
    try {
        VulkanHotSpineDemo demo(1280, 720);
        demo.main_loop();
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
