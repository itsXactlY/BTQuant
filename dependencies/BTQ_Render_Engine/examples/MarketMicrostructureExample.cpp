/**
 * @file MarketMicrostructureExample.cpp
 * @brief Example integration of the Market Microstructure Renderer
 * 
 * Demonstrates how to set up and use the renderer with a Hotspine data source.
 * This example shows the complete initialization, rendering loop, and data
 * update pipeline.
 * 
 * @author Market Microstructure Renderer Team
 * @version 1.0.0
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include "../include/components/MarketMicrostructureRenderer.h"
#include "../include/components/VulkanSynchronization.h"
#include "../include/trading/HotspineData.h"

using namespace components;
using namespace trading;

// ============================================
// Global Variables
// ============================================

std::atomic<bool> shouldExit = false;
std::atomic<uint32_t> frameCounter = 0;

// ============================================
// Vulkan Context (Simplified for Example)
// ============================================

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    uint32_t graphicsFamilyIndex = 0;
};

VulkanContext vulkanContext;

// ============================================
// Mock Hotspine Data Generator
// ============================================

class MockHotspineDataSource {
public:
    MockHotspineDataSource() {
        std::random_device rd;
        rng.seed(rd());
    }
    
    HotspineOrderBookSnapshot generateOrderBookSnapshot() {
        static uint32_t timeIndex = 0;
        
        HotspineOrderBookSnapshot snapshot;
        snapshot.currentTimeIndex = timeIndex++;
        snapshot.priceLevelsCount = 256;
        snapshot.basePrice = 100.0f - 5.0f;
        snapshot.priceRange = 10.0f;
        
        // Generate synthetic order book levels
        for (uint32_t i = 0; i < snapshot.priceLevelsCount; ++i) {
            const float price = snapshot.basePrice + (i / 255.0f) * snapshot.priceRange;
            const uint32_t bidQty = static_cast<uint32_t>(1000 * std::exp(-0.5f * std::pow((price - 100.0f), 2) / 0.1f));
            const uint32_t askQty = static_cast<uint32_t>(1000 * std::exp(-0.5f * std::pow((price - 100.0f), 2) / 0.1f));
            
            snapshot.levels[i] = OrderBookLevel{
                price,
                askQty,
                bidQty,
                static_cast<uint32_t>(bidQty / 10 + askQty / 10)
            };
        }
        
        return snapshot;
    }
    
    HotspineTradeTicks generateTradeTicks() {
        static uint64_t currentTime = 0;
        
        HotspineTradeTicks trades;
        trades.tickCount = 64;
        trades.startTime = currentTime;
        trades.endTime = currentTime + 1000000; // 1 millisecond
        trades.minPrice = 95.0f;
        trades.maxPrice = 105.0f;
        
        for (uint32_t i = 0; i < trades.tickCount; ++i) {
            const float price = 100.0f + (i % 10) - 5.0f;
            const uint32_t size = 10 + (i % 50);
            const bool isBuy = i % 2 == 0;
            
            trades.ticks[i] = TradeTick{
                price,
                size,
                currentTime + i * 1000,
                isBuy
            };
        }
        
        currentTime += 1000000;
        return trades;
    }
    
    std::vector<CandleCluster> generateFootprintClusters() {
        std::vector<CandleCluster> clusters;
        clusters.reserve(100);
        
        for (uint32_t i = 0; i < 100; ++i) {
            const float centerX = static_cast<float>(i) / 100.0f;
            const float centerY = 0.5f + 0.3f * std::sin(i * 0.1f);
            const float width = 0.01f;
            const float height = 0.005f;
            
            clusters.push_back(CandleCluster{
                centerX,
                centerY,
                width,
                height,
                static_cast<uint32_t>(100 + i * 10),
                static_cast<uint32_t>(100 + (99 - i) * 10),
                static_cast<uint32_t>(5 + i % 20),
                100.0f + 0.1f * (i - 50),
                i % 3 == 0
            });
        }
        
        return clusters;
    }
    
private:
    std::mt19937 rng;
};

// ============================================
// Vulkan Initialization (Simplified)
// ============================================

bool initializeVulkanContext() {
    // This is a simplified initialization - real code would use proper Vulkan setup
    // including instance, device, queue, and surface creation
    
    std::cout << "Initializing Vulkan context..." << std::endl;
    
    // Create instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Market Microstructure Example";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "BTQ Render Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    const char* extensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME // For X11
    };
    
    createInfo.ppEnabledExtensionNames = extensions;
    createInfo.enabledExtensionCount = 2;
    
    if (vkCreateInstance(&createInfo, nullptr, &vulkanContext.instance) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }
    
    std::cout << "Vulkan context initialized successfully" << std::endl;
    return true;
}

void cleanupVulkanContext() {
    if (vulkanContext.commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vulkanContext.device, vulkanContext.commandPool, nullptr);
    }
    
    if (vulkanContext.surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(vulkanContext.instance, vulkanContext.surface, nullptr);
    }
    
    if (vulkanContext.device != VK_NULL_HANDLE) {
        vkDestroyDevice(vulkanContext.device, nullptr);
    }
    
    if (vulkanContext.physicalDevice != VK_NULL_HANDLE) {
        // Physical device doesn't need explicit cleanup
    }
    
    if (vulkanContext.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(vulkanContext.instance, nullptr);
    }
}

// ============================================
// Main Renderer Loop
// ============================================

void rendererLoop(std::shared_ptr<MarketMicrostructureRenderer> renderer) {
    std::cout << "Starting renderer loop..." << std::endl;
    
    try {
        vk::VulkanSyncContext syncContext(vulkanContext.device, vulkanContext.commandPool);
        vk::TimelineSemaphore timelineSemaphore(vulkanContext.device);
        
        while (!shouldExit) {
            const uint32_t currentFrame = frameCounter % vk::MAX_FRAMES_IN_FLIGHT;
            
            // Wait for previous frame to complete and prepare for next
            if (!syncContext.prepareFrame(currentFrame, timelineSemaphore)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            // Acquire command buffer
            auto cmdBuffer = syncContext.acquireCommandBuffer();
            
            // Begin command buffer
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            
            if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
                std::cerr << "Failed to begin command buffer" << std::endl;
                continue;
            }
            
            // Render market microstructure
            if (!renderer->render(cmdBuffer, currentFrame, syncContext, timelineSemaphore)) {
                std::cerr << "Renderer render failed" << std::endl;
                vkEndCommandBuffer(cmdBuffer);
                continue;
            }
            
            // End command buffer
            if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
                std::cerr << "Failed to end command buffer" << std::endl;
                continue;
            }
            
            // Submit command buffer
            if (!syncContext.submitCommandBuffer(vulkanContext.graphicsQueue, cmdBuffer,
                                               timelineSemaphore, currentFrame + 1, currentFrame)) {
                std::cerr << "Failed to submit command buffer" << std::endl;
                continue;
            }
            
            // Present to screen (placeholder - real implementation needed)
            // This would normally use vkQueuePresentKHR
            
            frameCounter++;
            
            // Cap frame rate
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
    } catch (const std::exception& e) {
        std::cerr << "Renderer loop exception: " << e.what() << std::endl;
    }
    
    std::cout << "Renderer loop exited" << std::endl;
}

// ============================================
// Data Update Thread
// ============================================

void dataUpdateThread(std::shared_ptr<MarketMicrostructureRenderer> renderer) {
    std::cout << "Starting data update thread..." << std::endl;
    
    MockHotspineDataSource dataSource;
    auto lastUpdateTime = std::chrono::steady_clock::now();
    
    while (!shouldExit) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastUpdateTime
        ).count();
        
        // Update every 100ms
        if (elapsed >= 100) {
            // Update LOB data
            const auto orderBookSnapshot = dataSource.generateOrderBookSnapshot();
            renderer->updateLOBData(orderBookSnapshot);
            
            // Update trade data
            const auto tradeTicks = dataSource.generateTradeTicks();
            renderer->updateTradeData(tradeTicks);
            
            // Update footprint chart data
            const auto clusters = dataSource.generateFootprintClusters();
            renderer->updateFootprintClusters(clusters);
            
            lastUpdateTime = now;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "Data update thread exited" << std::endl;
}

// ============================================
// Main Function
// ============================================

int main() {
    std::cout << "=== Market Microstructure Renderer Example ===" << std::endl;
    
    // Initialize Vulkan context
    if (!initializeVulkanContext()) {
        std::cerr << "Failed to initialize Vulkan" << std::endl;
        return EXIT_FAILURE;
    }
    
    try {
        // Create renderer configuration
        RendererConfig config;
        config.lobHeatmap.width = 1024;
        config.lobHeatmap.height = 512;
        config.lobHeatmap.maxLiquidity = 100000.0f;
        config.lobHeatmap.invertYAxis = true;
        
        config.footprintChart.maxClusters = 4096;
        config.footprintChart.cellMinSize = 2.0f;
        config.footprintChart.cellMaxSize = 20.0f;
        config.footprintChart.showLabels = true;
        
        config.tpoProfile.bucketCount = 256;
        config.tpoProfile.priceResolution = 0.1f;
        config.tpoProfile.timeWindowMs = 30000;
        config.tpoProfile.resetOnUpdate = true;
        
        // Create and initialize renderer
        std::cout << "Creating renderer..." << std::endl;
        auto renderer = MarketMicrostructureRenderer::create(vulkanContext.device, config);
        if (!renderer) {
            throw std::runtime_error("Failed to create renderer");
        }
        
        std::cout << "Renderer created successfully" << std::endl;
        
        // Start data update thread
        std::thread updateThread(dataUpdateThread, renderer);
        
        // Start renderer loop
        std::thread renderThread(rendererLoop, renderer);
        
        // Wait for user input
        std::cout << "Press enter to exit..." << std::endl;
        std::cin.get();
        
        // Signal exit
        shouldExit = true;
        
        // Wait for threads to complete
        if (updateThread.joinable()) {
            updateThread.join();
        }
        
        if (renderThread.joinable()) {
            renderThread.join();
        }
        
        // Print statistics
        const auto stats = renderer->getStats();
        std::cout << "\n=== Renderer Statistics ===" << std::endl;
        std::cout << "Frames Rendered: " << stats.framesRendered << std::endl;
        std::cout << "LOB Updates: " << stats.lobUpdates << std::endl;
        std::cout << "Trade Updates: " << stats.tradeUpdates << std::endl;
        std::cout << "Average Frame Time: " << stats.averageFrameTimeMs << " ms" << std::endl;
        std::cout << "Footprint Cells Rendered: " << stats.footprintCellsRendered << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        cleanupVulkanContext();
        return EXIT_FAILURE;
    }
    
    // Cleanup Vulkan context
    cleanupVulkanContext();
    
    std::cout << "=== Example completed successfully ===" << std::endl;
    return EXIT_SUCCESS;
}