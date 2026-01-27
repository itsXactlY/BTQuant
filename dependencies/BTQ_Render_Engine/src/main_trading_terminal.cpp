/**
 * BTQuant Trading Terminal - Main Entry Point
 * 
 * Professional DOM trading terminal with real-time market data visualization
 * Built on BTQ_Render_Engine with Vulkan backend
 */

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <format>
#include <algorithm>
#include <execution>

// Vulkan headers
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

// GLM for math
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Project headers
#include "vulkan_base_types.hpp"
#include "vulkan_dashboard_advanced.hpp"
#include "hotspine_data_bridge.hpp"
#include "performance_monitor.hpp"
#include "symbol_manager.hpp"
#include "market_data_processor.hpp"

// Component headers
#include "components/panel_base.hpp"
#include "components/panel_manager.hpp"
#include "components/chart_panel.hpp"
#include "components/metrics_panel.hpp"
#include "components/orderbook_panel.hpp"
#include "components/status_bar_panel.hpp"
#include "components/tape_panel.hpp"
#include "components/volume_profile_panel.hpp"
#include "components/depth_chart_panel.hpp"
#include "components/footprint_panel.hpp"
#include "components/tpo_panel.hpp"
#include "components/dom_surface_panel.hpp"
#include "components/MarketMicrostructureRenderer.h"
#include "components/realtime_dashboard_component.hpp"
#include "components/correlation_heatmap_component.hpp"
#include "components/symbol_selector.hpp"
#include "components/theme_manager.hpp"
#include "components/interaction_manager.hpp"

// Trading headers
#include "trading/trading_interface.hpp"

// Analytics headers
#include "analytics/trading_analytics.hpp"

// Layout headers
#include "layout/dashboard_layout_manager.hpp"

// Config headers
#include "dashboard_config.hpp"

// Data headers
#include "data/ui_data_manager.hpp"

// System headers
#include "system/system_optimizer.hpp"

using namespace BTQuant;
using namespace BTQuant::Trading;
using namespace BTQuant::Analytics;
using namespace BTQuant::Layout;
using namespace BTQuant::System;

// ============================================================================
// Application State
// ============================================================================

struct ApplicationState {
    std::atomic<bool> running{true};
    std::atomic<bool> paused{false};
    std::atomic<uint64_t> frame_count{0};
    
    // Performance metrics
    double current_fps{0.0};
    double frame_time_ms{0.0};
    double cpu_usage{0.0};
    double gpu_usage{0.0};
    
    // Market data
    std::string current_symbol{"BTCUSDT"};
    std::string current_timeframe{"15m"};
    
    // UI state
    bool show_demo_window{false};
    bool show_performance_overlay{true};
    bool show_debug_info{false};
    
    // Theme
    std::string current_theme{"dark"};
};

// ============================================================================
// Global State
// ============================================================================

static ApplicationState g_app_state;
static std::unique_ptr<BTQuant::VulkanDashboardAdvanced> g_dashboard;
static std::unique_ptr<BTQuant::PanelManager> g_panel_manager;
static std::unique_ptr<BTQuant::HotSpineDataBridge> g_data_bridge;
static std::unique_ptr<BTQuant::RenderEngine::MarketDataProcessor> g_market_processor;
static std::unique_ptr<BTQuant::RenderEngine::SymbolManager> g_symbol_manager;
static std::unique_ptr<BTQuant::PerformanceMonitor> g_perf_monitor;
static std::unique_ptr<BTQuant::ThemeManager> g_theme_manager;
static std::unique_ptr<BTQuant::InteractionManager> g_interaction_manager;
static std::unique_ptr<BTQuant::TradingInterface> g_trading_interface;
static std::unique_ptr<BTQuant::TradingAnalytics> g_analytics;
static std::unique_ptr<BTQuant::Layout::DashboardLayoutManager> g_layout_manager;
static std::unique_ptr<BTQuant::Data::UIDataManager> g_ui_data_manager;
static std::unique_ptr<BTQuant::System::SystemOptimizer> g_system_optimizer;

// ============================================================================
// Performance Monitoring
// ============================================================================

void update_performance_metrics() {
    static auto last_time = std::chrono::high_resolution_clock::now();
    static uint64_t frame_counter = 0;
    
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(current_time - last_time).count();
    
    frame_counter++;
    
    if (elapsed >= 1.0) {
        g_app_state.current_fps = frame_counter / elapsed;
        g_app_state.frame_time_ms = (elapsed / frame_counter) * 1000.0;
        
        frame_counter = 0;
        last_time = current_time;
        
        // Update performance monitor
        if (g_perf_monitor) {
            g_perf_monitor->update_fps(g_app_state.current_fps);
            g_perf_monitor->update_frame_time(g_app_state.frame_time_ms);
        }
    }
}

// ============================================================================
// Market Data Processing
// ============================================================================

void process_market_data() {
    if (!g_data_bridge || !g_market_processor) {
        return;
    }
    
    // Read latest data from HotSpine
    auto trades = g_data_bridge->read_trades(1000);
    auto orderbook = g_data_bridge->read_orderbook();
    
    // Process through market data processor
    g_market_processor->process_trades(trades);
    g_market_processor->process_orderbook(orderbook);
    
    // Update analytics
    if (g_analytics) {
        g_analytics->update_trades(trades);
        g_analytics->update_orderbook(orderbook);
    }
}

// ============================================================================
// UI Rendering
// ============================================================================

void render_performance_overlay() {
    if (!g_app_state.show_performance_overlay) {
        return;
    }
    
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
    
    if (ImGui::Begin("Performance", &g_app_state.show_performance_overlay,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize)) {
        
        ImGui::Text("FPS: %.1f", g_app_state.current_fps);
        ImGui::Text("Frame Time: %.2f ms", g_app_state.frame_time_ms);
        ImGui::Text("CPU Usage: %.1f%%", g_app_state.cpu_usage);
        ImGui::Text("GPU Usage: %.1f%%", g_app_state.gpu_usage);
        ImGui::Text("Frame Count: %lu", g_app_state.frame_count.load());
        
        ImGui::Separator();
        
        if (g_perf_monitor) {
            auto metrics = g_perf_monitor->get_metrics();
            ImGui::Text("Render Time: %.2f ms", metrics.render_time_ms);
            ImGui::Text("Update Time: %.2f ms", metrics.update_time_ms);
            ImGui::Text("Data Processing: %.2f ms", metrics.data_processing_time_ms);
        }
        
        ImGui::Separator();
        
        ImGui::Text("Symbol: %s", g_app_state.current_symbol.c_str());
        ImGui::Text("Timeframe: %s", g_app_state.current_timeframe.c_str());
        
        if (g_data_bridge) {
            auto stats = g_data_bridge->get_statistics();
            ImGui::Text("Trades/sec: %.0f", stats.trades_per_second);
            ImGui::Text("Orderbook Updates/sec: %.0f", stats.orderbook_updates_per_second);
        }
    }
    
    ImGui::End();
}

void render_main_menu() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Layout")) {
                if (g_layout_manager) {
                    g_layout_manager->create_new_layout();
                }
            }
            if (ImGui::MenuItem("Load Layout")) {
                if (g_layout_manager) {
                    g_layout_manager->load_layout("default");
                }
            }
            if (ImGui::MenuItem("Save Layout")) {
                if (g_layout_manager) {
                    g_layout_manager->save_layout("default");
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                g_app_state.running.store(false);
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Performance Overlay", nullptr, g_app_state.show_performance_overlay)) {
                g_app_state.show_performance_overlay = !g_app_state.show_performance_overlay;
            }
            if (ImGui::MenuItem("Debug Info", nullptr, g_app_state.show_debug_info)) {
                g_app_state.show_debug_info = !g_app_state.show_debug_info;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Reset Layout")) {
                if (g_layout_manager) {
                    g_layout_manager->reset_layout();
                }
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Theme")) {
            if (ImGui::MenuItem("Dark", nullptr, g_app_state.current_theme == "dark")) {
                g_app_state.current_theme = "dark";
                if (g_theme_manager) {
                    g_theme_manager->set_theme("dark");
                }
            }
            if (ImGui::MenuItem("Light", nullptr, g_app_state.current_theme == "light")) {
                g_app_state.current_theme = "light";
                if (g_theme_manager) {
                    g_theme_manager->set_theme("light");
                }
            }
            if (ImGui::MenuItem("Midnight", nullptr, g_app_state.current_theme == "midnight")) {
                g_app_state.current_theme = "midnight";
                if (g_theme_manager) {
                    g_theme_manager->set_theme("midnight");
                }
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Tools")) {
            if (ImGui::MenuItem("System Optimizer")) {
                if (g_system_optimizer) {
                    g_system_optimizer->optimize();
                }
            }
            if (ImGui::MenuItem("Performance Benchmark")) {
                // Run benchmark
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Documentation")) {
                // Open documentation
            }
            if (ImGui::MenuItem("About")) {
                ImGui::OpenPopup("About");
            }
            ImGui::EndMenu();
        }
        
        ImGui::EndMainMenuBar();
    }
    
    // About popup
    if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("BTQuant Trading Terminal");
        ImGui::Separator();
        ImGui::Text("Version: 1.0.0");
        ImGui::Text("Built on BTQ_Render_Engine");
        ImGui::Text("Vulkan Backend");
        ImGui::Separator();
        ImGui::Text("© 2026 BTQuant");
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void render_dockspace() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    
    ImGui::Begin("DockSpace", nullptr, window_flags);
    ImGui::PopStyleVar(3);
    
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
    
    render_main_menu();
    
    ImGui::End();
}

// ============================================================================
// Main Render Loop
// ============================================================================

void render_frame() {
    // Update performance metrics
    update_performance_metrics();
    
    // Process market data
    process_market_data();
    
    // Start new ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Render dockspace
    render_dockspace();
    
    // Render performance overlay
    render_performance_overlay();
    
    // Update and render all panels
    if (g_panel_manager) {
        g_panel_manager->update_panels(1.0 / g_app_state.current_fps);
        g_panel_manager->render_panels();
    }
    
    // Render ImGui
    ImGui::Render();
    
    // Submit to Vulkan
    if (g_dashboard) {
        g_dashboard->render(ImGui::GetDrawData());
    }
    
    // Increment frame counter
    g_app_state.frame_count.fetch_add(1, std::memory_order_relaxed);
}

// ============================================================================
// Initialization
// ============================================================================

bool initialize_application() {
    std::cout << "Initializing BTQuant Trading Terminal..." << std::endl;
    
    // Initialize performance monitor
    g_perf_monitor = std::make_unique<BTQuant::PerformanceMonitor>();
    std::cout << "✓ Performance monitor initialized" << std::endl;
    
    // Initialize theme manager
    g_theme_manager = std::make_unique<BTQuant::ThemeManager>();
    g_theme_manager->set_theme("dark");
    std::cout << "✓ Theme manager initialized" << std::endl;
    
    // Initialize symbol manager
    g_symbol_manager = std::make_unique<BTQuant::RenderEngine::SymbolManager>();
    g_symbol_manager->load_symbols("config/symbols.json");
    std::cout << "✓ Symbol manager initialized" << std::endl;
    
    // Initialize data bridge
    g_data_bridge = std::make_unique<BTQuant::HotSpineDataBridge>("/dev/shm/hotspine_trades");
    if (!g_data_bridge->initialize()) {
        std::cerr << "✗ Failed to initialize HotSpine data bridge" << std::endl;
        return false;
    }
    std::cout << "✓ HotSpine data bridge initialized" << std::endl;
    
    // Initialize market data processor
    g_market_processor = std::make_unique<BTQuant::RenderEngine::MarketDataProcessor>();
    std::cout << "✓ Market data processor initialized" << std::endl;
    
    // Initialize UI data manager
    g_ui_data_manager = std::make_unique<BTQuant::Data::UIDataManager>();
    std::cout << "✓ UI data manager initialized" << std::endl;
    
    // Initialize analytics
    g_analytics = std::make_unique<BTQuant::TradingAnalytics>();
    std::cout << "✓ Trading analytics initialized" << std::endl;
    
    // Initialize trading interface
    g_trading_interface = std::make_unique<BTQuant::TradingInterface>();
    std::cout << "✓ Trading interface initialized" << std::endl;
    
    // Initialize interaction manager
    g_interaction_manager = std::make_unique<BTQuant::InteractionManager>();
    std::cout << "✓ Interaction manager initialized" << std::endl;
    
    // Initialize system optimizer
    g_system_optimizer = std::make_unique<BTQuant::System::SystemOptimizer>();
    g_system_optimizer->optimize();
    std::cout << "✓ System optimizer initialized" << std::endl;
    
    // Initialize layout manager
    g_layout_manager = std::make_unique<BTQuant::Layout::DashboardLayoutManager>();
    std::cout << "✓ Layout manager initialized" << std::endl;
    
    // Initialize dashboard
    g_dashboard = std::make_unique<BTQuant::VulkanDashboardAdvanced>();
    if (!g_dashboard->initialize()) {
        std::cerr << "✗ Failed to initialize Vulkan dashboard" << std::endl;
        return false;
    }
    std::cout << "✓ Vulkan dashboard initialized" << std::endl;
    
    // Initialize panel manager
    g_panel_manager = std::make_unique<BTQuant::PanelManager>();
    
    // Add panels
    BTQuant::PanelConfig chart_config{"Chart", ImVec2(800, 600), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::ChartPanel>(chart_config, g_app_state.current_symbol));
    
    BTQuant::PanelConfig dom_config{"DOM Surface", ImVec2(400, 600), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::DOMSurfacePanel>(dom_config, g_app_state.current_symbol));
    
    BTQuant::PanelConfig footprint_config{"Footprint", ImVec2(400, 600), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::FootprintPanel>(footprint_config, g_app_state.current_symbol));
    
    BTQuant::PanelConfig volume_config{"Volume Profile", ImVec2(400, 600), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::VolumeProfilePanel>(volume_config, g_app_state.current_symbol));
    
    BTQuant::PanelConfig depth_config{"Depth Chart", ImVec2(400, 600), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::DepthChartPanel>(depth_config, g_app_state.current_symbol));
    
    BTQuant::PanelConfig tpo_config{"TPO Profile", ImVec2(400, 600), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::TPOPanel>(tpo_config, g_app_state.current_symbol));
    
    BTQuant::PanelConfig orderbook_config{"Orderbook", ImVec2(400, 600), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::OrderbookPanel>(orderbook_config, g_app_state.current_symbol));
    
    BTQuant::PanelConfig tape_config{"Tape", ImVec2(400, 600), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::TapePanel>(tape_config, g_app_state.current_symbol));
    
    BTQuant::PanelConfig status_config{"Status Bar", ImVec2(1200, 30), true};
    g_panel_manager->add_panel(std::make_unique<BTQuant::StatusBarPanel>(status_config,
        std::shared_ptr<BTQuant::HotSpineDataBridge>(g_data_bridge.get(), [](auto*){}),
        std::shared_ptr<BTQuant::RenderEngine::MarketDataProcessor>(g_market_processor.get(), [](auto*){})));
    
    std::cout << "✓ Panel manager initialized with " << g_panel_manager->get_panel_count() << " panels" << std::endl;
    
    // Load default layout
    g_layout_manager->load_layout("default");
    std::cout << "✓ Default layout loaded" << std::endl;
    
    std::cout << "\n✓ Application initialized successfully!" << std::endl;
    return true;
}

// ============================================================================
// Cleanup
// ============================================================================

void cleanup_application() {
    std::cout << "\nCleaning up..." << std::endl;
    
    // Save current layout
    if (g_layout_manager) {
        g_layout_manager->save_layout("default");
    }
    
    // Cleanup in reverse order
    g_layout_manager.reset();
    g_system_optimizer.reset();
    g_interaction_manager.reset();
    g_trading_interface.reset();
    g_analytics.reset();
    g_ui_data_manager.reset();
    g_market_processor.reset();
    g_data_bridge.reset();
    g_symbol_manager.reset();
    g_theme_manager.reset();
    g_perf_monitor.reset();
    g_panel_manager.reset();
    g_dashboard.reset();
    
    std::cout << "✓ Cleanup complete" << std::endl;
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  BTQuant Trading Terminal v1.0.0" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--symbol" && i + 1 < argc) {
            g_app_state.current_symbol = argv[++i];
        } else if (arg == "--timeframe" && i + 1 < argc) {
            g_app_state.current_timeframe = argv[++i];
        } else if (arg == "--theme" && i + 1 < argc) {
            g_app_state.current_theme = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --symbol SYMBOL      Set initial symbol (default: BTCUSDT)" << std::endl;
            std::cout << "  --timeframe TF       Set initial timeframe (default: 15m)" << std::endl;
            std::cout << "  --theme THEME        Set theme (dark/light/midnight)" << std::endl;
            std::cout << "  --help               Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Initialize application
    if (!initialize_application()) {
        std::cerr << "\n✗ Failed to initialize application" << std::endl;
        return 1;
    }
    
    std::cout << "\nStarting main loop..." << std::endl;
    
    // Main render loop
    while (g_app_state.running.load()) {
        // Poll events
        glfwPollEvents();
        
        // Check if window should close
        if (glfwWindowShouldClose(g_dashboard->get_window())) {
            g_app_state.running.store(false);
            break;
        }
        
        // Render frame
        render_frame();
    }
    
    // Cleanup
    cleanup_application();
    
    std::cout << "\nShutdown complete. Goodbye!" << std::endl;
    return 0;
}
