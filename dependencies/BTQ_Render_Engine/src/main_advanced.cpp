#include "vulkan_dashboard_advanced.hpp"
#include <cstdlib>
#include <cstdio>

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    fprintf(stderr, "\n╔════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║   BTQuant Advanced Vulkan Dashboard (Production Ready)    ║\n");
    fprintf(stderr, "║   6 Crypto Symbols | Lock-Free HotSpine | 60 FPS         ║\n");
    fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n\n");
    
    fprintf(stderr, "[Main] Initializing dashboard (1280x720)...\n");
    fprintf(stderr, "[Main] Symbols: DOGEUSDT, ADAUSDT, BTCUSDT, BNBUSDT, ETHUSDT, SOLUSDT\n");
    fprintf(stderr, "[Main] Layout: 3x2 grid (color = price momentum)\n");
    fprintf(stderr, "[Main] Close window to exit\n\n");

    VulkanDashboard* dashboard = nullptr;
    
    if (!dashboard) {
        dashboard = new VulkanDashboard(1280, 720);
    }

    if (dashboard) {
        dashboard->main_loop();
        delete dashboard;
        fprintf(stderr, "\n[Main] Dashboard shutdown complete\n");
        return 0;
    } else {
        fprintf(stderr, "\n[FATAL] Failed to create dashboard\n");
        return 1;
    }
}