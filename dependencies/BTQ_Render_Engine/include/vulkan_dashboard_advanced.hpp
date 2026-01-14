#pragma once

/**
 * BTQuant Advanced Vulkan Dashboard
 * 
 * A cutting-edge financial trading dashboard built on modern Vulkan rendering
 * technology, designed for professional-grade performance and real-time market
 * data visualization.
 * 
 * Features:
 * - Modern Vulkan rendering pipeline with MSAA and HDR support
 * - GPU-driven rendering with compute shader integration
 * - Real-time HotSpine market data integration
 * - Professional UI component system
 * - Multi-threaded command buffer management
 * - Advanced memory management with VMA integration
 * - 60 FPS performance with 1000+ UI elements
 * - <1ms data-to-display latency
 */

// Platform-specific Vulkan includes
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>

// X11 includes with macro protection
#ifdef KeyPress
#undef KeyPress
#endif
#ifdef KeyRelease
#undef KeyRelease
#endif
#include <X11/Xlib.h>
#include <X11/Xutil.h>
// Redefine X11 constants with different names to avoid conflicts
#define X11_KeyPress 2
#define X11_KeyRelease 3

// Math library
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// HotSpine data integration
#include "../../ccapi/example/src/hotspine/hotspine_layout.hpp"

// Standard library includes
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <queue>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <exception>
#include <iostream>
#include <fstream>

// Concurrent data structures
#include "concurrentqueue.h"

namespace BTQuant {

// Forward declarations
class VulkanCore;
class GPUMemoryManager;
class UIComponent;
class MarketDataProcessor;
class DataStreamManager;

// ============================================================================
// Core Configuration and Constants
// ============================================================================

struct DashboardConfig {
    // Vulkan configuration
    bool enable_validation_layers = true;
    bool enable_msaa = true;
    VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_4_BIT;
    bool enable_hdr = true;
    VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    
    // Performance targets
    uint32_t target_fps = 60;
    uint32_t max_ui_elements = 1000;
    float max_data_latency_ms = 1.0f;
    
    // Memory configuration
    size_t vertex_pool_size = 64 * 1024 * 1024;    // 64MB
    size_t uniform_pool_size = 16 * 1024 * 1024;   // 16MB
    size_t storage_pool_size = 32 * 1024 * 1024;   // 32MB
    
    // HotSpine integration
    std::string shm_name = "/btquant_hotspine";
    std::string symbol_registry_path = "/dev/shm/btquant_symbols.json";
};

// Professional trading dashboard theme
struct DashboardTheme {
    // Background colors
    glm::vec4 background_primary{0.1f, 0.1f, 0.1f, 1.0f};
    glm::vec4 background_secondary{0.15f, 0.15f, 0.15f, 1.0f};
    glm::vec4 background_panel{0.12f, 0.12f, 0.12f, 1.0f};
    
    // Text colors
    glm::vec4 text_primary{0.9f, 0.9f, 0.9f, 1.0f};
    glm::vec4 text_secondary{0.7f, 0.7f, 0.7f, 1.0f};
    glm::vec4 text_muted{0.5f, 0.5f, 0.5f, 1.0f};
    
    // Market data colors
    glm::vec4 price_up{0.0f, 0.8f, 0.0f, 1.0f};
    glm::vec4 price_down{0.8f, 0.0f, 0.0f, 1.0f};
    glm::vec4 price_neutral{0.6f, 0.6f, 0.6f, 1.0f};
    
    // UI accent colors
    glm::vec4 accent_primary{0.2f, 0.6f, 1.0f, 1.0f};
    glm::vec4 accent_secondary{0.8f, 0.4f, 0.0f, 1.0f};
    glm::vec4 border_color{0.3f, 0.3f, 0.3f, 1.0f};
    
    // Status colors
    glm::vec4 status_connected{0.0f, 0.8f, 0.0f, 1.0f};
    glm::vec4 status_disconnected{0.8f, 0.0f, 0.0f, 1.0f};
    glm::vec4 status_warning{0.8f, 0.8f, 0.0f, 1.0f};
};

// ============================================================================
// Error Handling and Validation
// ============================================================================

class VulkanException : public std::exception {
public:
    VulkanException(VkResult result, const std::string& operation)
        : result_(result), operation_(operation) {
        message_ = "Vulkan error in " + operation + ": " + std::to_string(static_cast<int>(result));
    }
    
    const char* what() const noexcept override { return message_.c_str(); }
    VkResult result() const { return result_; }
    const std::string& operation() const { return operation_; }

private:
    VkResult result_;
    std::string operation_;
    std::string message_;
};

class VulkanErrorHandler {
public:
    static void check_result(VkResult result, const std::string& operation) {
        if (result != VK_SUCCESS) {
            throw VulkanException(result, operation);
        }
    }
    
    static void setup_debug_messenger(VkInstance instance);
    static void cleanup_debug_messenger(VkInstance instance);

private:
    static VkDebugUtilsMessengerEXT debug_messenger_;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        void* user_data);
};

// ============================================================================
// Memory Management System
// ============================================================================

struct BufferAllocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped_ptr = nullptr;
    VkDeviceSize size = 0;
    VkDeviceSize offset = 0;
    bool is_mapped = false;
};

class MemoryPool {
public:
    MemoryPool(VkDevice device, VkPhysicalDevice physical_device, 
               VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
               VkDeviceSize pool_size);
    ~MemoryPool();
    
    BufferAllocation allocate(VkDeviceSize size, VkDeviceSize alignment = 1);
    void deallocate(const BufferAllocation& allocation);
    
    VkDeviceSize get_total_size() const { return pool_size_; }
    VkDeviceSize get_used_size() const { return used_size_; }
    float get_usage_percentage() const { return static_cast<float>(used_size_) / pool_size_; }

private:
    VkDevice device_;
    VkBuffer pool_buffer_;
    VkDeviceMemory pool_memory_;
    void* mapped_ptr_;
    VkDeviceSize pool_size_;
    std::atomic<VkDeviceSize> used_size_{0};
    
    struct FreeBlock {
        VkDeviceSize offset;
        VkDeviceSize size;
    };
    std::vector<FreeBlock> free_blocks_;
    std::mutex allocation_mutex_;
    
    uint32_t find_memory_type(VkPhysicalDevice physical_device, 
                             uint32_t type_filter, 
                             VkMemoryPropertyFlags properties);
};

class GPUMemoryManager {
public:
    GPUMemoryManager(VkDevice device, VkPhysicalDevice physical_device, const DashboardConfig& config);
    ~GPUMemoryManager();
    
    // Specialized allocators for different buffer types
    BufferAllocation allocate_vertex_buffer(VkDeviceSize size);
    BufferAllocation allocate_index_buffer(VkDeviceSize size);
    BufferAllocation allocate_uniform_buffer(VkDeviceSize size);
    BufferAllocation allocate_storage_buffer(VkDeviceSize size);
    
    void deallocate_buffer(const BufferAllocation& allocation);
    
    // Memory usage statistics
    struct MemoryStats {
        VkDeviceSize vertex_pool_used;
        VkDeviceSize uniform_pool_used;
        VkDeviceSize storage_pool_used;
        float vertex_pool_usage;
        float uniform_pool_usage;
        float storage_pool_usage;
    };
    MemoryStats get_memory_stats() const;

private:
    VkDevice device_;
    std::unique_ptr<MemoryPool> vertex_pool_;
    std::unique_ptr<MemoryPool> uniform_pool_;
    std::unique_ptr<MemoryPool> storage_pool_;
};

// ============================================================================
// Vulkan Core Rendering System
// ============================================================================

class VulkanCore {
public:
    VulkanCore(const DashboardConfig& config);
    ~VulkanCore();
    
    // Initialization and cleanup
    void initialize(Display* display, Window window, uint32_t width, uint32_t height);
    void cleanup();
    
    // Frame rendering
    void begin_frame();
    void end_frame();
    VkCommandBuffer get_current_command_buffer() const { return current_command_buffer_; }
    
    // Resource access
    VkDevice get_device() const { return device_; }
    VkPhysicalDevice get_physical_device() const { return physical_device_; }
    VkQueue get_graphics_queue() const { return graphics_queue_; }
    VkQueue get_present_queue() const { return present_queue_; }
    VkQueue get_compute_queue() const { return compute_queue_; }
    VkRenderPass get_render_pass() const { return render_pass_; }
    VkExtent2D get_swapchain_extent() const { return swapchain_extent_; }
    
    // Memory management
    GPUMemoryManager& get_memory_manager() { return *memory_manager_; }
    
    // Synchronization
    void wait_idle() { vkDeviceWaitIdle(device_); }
    
    // Swapchain management
    void recreate_swapchain(uint32_t width, uint32_t height);

private:
    DashboardConfig config_;
    
    // Core Vulkan objects
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    
    // Queues
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_ = UINT32_MAX;
    uint32_t present_queue_family_ = UINT32_MAX;
    uint32_t compute_queue_family_ = UINT32_MAX;
    
    // Swapchain
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_image_views_;
    VkFormat swapchain_image_format_;
    VkExtent2D swapchain_extent_;
    
    // Render pass and framebuffers
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;
    
    // MSAA resources
    VkImage msaa_color_image_ = VK_NULL_HANDLE;
    VkImageView msaa_color_image_view_ = VK_NULL_HANDLE;
    VkDeviceMemory msaa_color_memory_ = VK_NULL_HANDLE;
    
    // Depth buffer
    VkImage depth_image_ = VK_NULL_HANDLE;
    VkImageView depth_image_view_ = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory_ = VK_NULL_HANDLE;
    
    // Command buffers
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;
    VkCommandBuffer current_command_buffer_ = VK_NULL_HANDLE;
    
    // Synchronization
    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence> in_flight_fences_;
    uint32_t current_frame_ = 0;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    
    // Memory management
    std::unique_ptr<GPUMemoryManager> memory_manager_;
    
    // Frame timing
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    float delta_time_ = 0.0f;
    
    // Private initialization methods
    void create_instance();
    void select_physical_device();
    void create_logical_device();
    void create_surface(Display* display, Window window);
    void create_swapchain(uint32_t width, uint32_t height);
    void create_image_views();
    void create_render_pass();
    void create_msaa_resources();
    void create_depth_resources();
    void create_framebuffers();
    void create_command_pool();
    void create_command_buffers();
    void create_sync_objects();
    
    // Helper methods
    std::vector<const char*> get_required_extensions();
    bool check_validation_layer_support();
    bool is_device_suitable(VkPhysicalDevice device);
    VkSampleCountFlagBits get_max_usable_sample_count();
    VkFormat find_supported_format(const std::vector<VkFormat>& candidates,
                                  VkImageTiling tiling,
                                  VkFormatFeatureFlags features);
    VkFormat find_depth_format();
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    
    // Cleanup helpers
    void cleanup_swapchain();
};

// ============================================================================
// UI Component System
// ============================================================================

struct InputEvent {
    enum EventType { MouseMove, MouseButton, KeyDown, KeyUp, Scroll };
    EventType type;
    glm::vec2 position;
    int button = 0;
    int key = 0;
    bool pressed = false;
    glm::vec2 scroll_delta{0.0f};
};

class UIComponent {
public:
    UIComponent(const glm::vec2& position, const glm::vec2& size)
        : position_(position), size_(size) {}
    
    virtual ~UIComponent() = default;
    
    // Core interface
    virtual void update(float delta_time) = 0;
    virtual void render(VkCommandBuffer cmd) = 0;
    virtual void handle_input(const InputEvent& event) = 0;
    
    // Layout management
    void set_position(const glm::vec2& position) { position_ = position; dirty_ = true; }
    void set_size(const glm::vec2& size) { size_ = size; dirty_ = true; }
    glm::vec2 get_position() const { return position_; }
    glm::vec2 get_size() const { return size_; }
    
    // Visibility and state
    void set_visible(bool visible) { visible_ = visible; }
    bool is_visible() const { return visible_; }
    bool is_dirty() const { return dirty_; }
    void mark_clean() { dirty_ = false; }

protected:
    glm::vec2 position_;
    glm::vec2 size_;
    bool visible_ = true;
    bool dirty_ = true;
    DashboardTheme theme_;
};

// Data grid for displaying tabular market data
class DataGridComponent : public UIComponent {
public:
    struct CellData {
        std::string text;
        glm::vec4 color;
        float numeric_value = 0.0f;
        bool highlight = false;
        bool is_numeric = false;
    };
    
    DataGridComponent(const glm::vec2& position, const glm::vec2& size, 
                     size_t rows, size_t columns);
    ~DataGridComponent();
    
    // Data management
    void set_cell_data(size_t row, size_t col, const CellData& data);
    void set_row_data(size_t row, const std::vector<CellData>& row_data);
    void set_column_header(size_t col, const std::string& header);
    void set_column_width(size_t col, float width);
    
    // Sorting and filtering
    void enable_sorting(size_t column, bool ascending = true);
    void set_filter(const std::string& filter_text);
    
    // UIComponent interface
    void update(float delta_time) override;
    void render(VkCommandBuffer cmd) override;
    void handle_input(const InputEvent& event) override;

private:
    size_t rows_, columns_;
    std::vector<std::vector<CellData>> grid_data_;
    std::vector<std::string> column_headers_;
    std::vector<float> column_widths_;
    
    // Rendering resources
    BufferAllocation vertex_buffer_;
    BufferAllocation index_buffer_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    
    // Sorting state
    int sort_column_ = -1;
    bool sort_ascending_ = true;
    
    void rebuild_geometry();
    void sort_data();
};

// Real-time price chart component
class RealtimeChartComponent : public UIComponent {
public:
    struct DataPoint {
        float timestamp;
        float value;
        float volume = 0.0f;
    };
    
    RealtimeChartComponent(const glm::vec2& position, const glm::vec2& size);
    ~RealtimeChartComponent();
    
    // Data management
    void add_data_point(float timestamp, float value, float volume = 0.0f);
    void set_time_window(float seconds);
    void set_y_range(float min_y, float max_y);
    void enable_auto_scale(bool enable) { auto_scale_ = enable; }
    
    // Display options
    void enable_candlestick_mode(bool enable);
    void set_line_color(const glm::vec4& color) { line_color_ = color; }
    
    // UIComponent interface
    void update(float delta_time) override;
    void render(VkCommandBuffer cmd) override;
    void handle_input(const InputEvent& event) override;

private:
    std::deque<DataPoint> data_points_;
    float time_window_ = 60.0f;
    float min_y_ = 0.0f, max_y_ = 100.0f;
    bool auto_scale_ = true;
    bool candlestick_mode_ = false;
    glm::vec4 line_color_{1.0f, 1.0f, 1.0f, 1.0f};
    
    // Rendering resources
    BufferAllocation line_vertex_buffer_;
    BufferAllocation candlestick_vertex_buffer_;
    VkPipeline line_pipeline_ = VK_NULL_HANDLE;
    VkPipeline candlestick_pipeline_ = VK_NULL_HANDLE;
    
    void rebuild_line_geometry();
    void rebuild_candlestick_geometry();
    void update_y_range();
};

// Momentum heatmap visualization
class HeatmapComponent : public UIComponent {
public:
    struct HeatmapData {
        float value;
        glm::vec4 color;
        std::string label;
        uint32_t symbol_id;
    };
    
    HeatmapComponent(const glm::vec2& position, const glm::vec2& size,
                    size_t grid_width, size_t grid_height);
    ~HeatmapComponent();
    
    // Data management
    void set_data(const std::vector<std::vector<HeatmapData>>& data);
    void update_cell(size_t x, size_t y, const HeatmapData& data);
    void set_color_scheme(const std::vector<glm::vec4>& colors);
    
    // Display options
    void enable_interpolation(bool enable) { interpolation_enabled_ = enable; }
    void set_value_range(float min_val, float max_val);
    
    // UIComponent interface
    void update(float delta_time) override;
    void render(VkCommandBuffer cmd) override;
    void handle_input(const InputEvent& event) override;

private:
    size_t grid_width_, grid_height_;
    std::vector<std::vector<HeatmapData>> heatmap_data_;
    std::vector<glm::vec4> color_scheme_;
    bool interpolation_enabled_ = true;
    float min_value_ = -1.0f, max_value_ = 1.0f;
    
    // Compute shader resources for interpolation
    VkBuffer compute_input_buffer_;
    VkBuffer compute_output_buffer_;
    VkDescriptorSet compute_descriptor_set_ = VK_NULL_HANDLE;
    VkPipeline compute_pipeline_ = VK_NULL_HANDLE;
    
    // Rendering resources
    BufferAllocation vertex_buffer_;
    BufferAllocation index_buffer_;
    VkPipeline render_pipeline_ = VK_NULL_HANDLE;
    
    void rebuild_geometry();
    void dispatch_compute_interpolation();
    glm::vec4 interpolate_color(float value);
};

// Order book visualization component
class OrderBookComponent : public UIComponent {
public:
    struct OrderBookLevel {
        double price;
        double size;
        double total_size;
    };
    
    struct OrderBookData {
        std::vector<OrderBookLevel> bids;
        std::vector<OrderBookLevel> asks;
        double spread;
        double mid_price;
        uint64_t timestamp;
    };
    
    OrderBookComponent(const glm::vec2& position, const glm::vec2& size);
    ~OrderBookComponent();
    
    // Data management
    void update_orderbook(const OrderBookData& data);
    void set_symbol(const std::string& symbol) { symbol_ = symbol; }
    void set_precision(int price_precision, int size_precision);
    
    // Display options
    void set_max_levels(size_t levels) { max_levels_ = levels; }
    void enable_size_bars(bool enable) { show_size_bars_ = enable; }
    
    // UIComponent interface
    void update(float delta_time) override;
    void render(VkCommandBuffer cmd) override;
    void handle_input(const InputEvent& event) override;

private:
    OrderBookData current_data_;
    std::string symbol_;
    size_t max_levels_ = 10;
    int price_precision_ = 2;
    int size_precision_ = 4;
    bool show_size_bars_ = true;
    
    // Rendering resources
    BufferAllocation text_vertex_buffer_;
    BufferAllocation bar_vertex_buffer_;
    VkPipeline text_pipeline_ = VK_NULL_HANDLE;
    VkPipeline bar_pipeline_ = VK_NULL_HANDLE;
    
    void rebuild_geometry();
    std::string format_price(double price);
    std::string format_size(double size);
};

// System log display component
class LogDisplayComponent : public UIComponent {
public:
    enum LogLevel { Debug, Info, Warning, Error };
    
    struct LogEntry {
        std::chrono::system_clock::time_point timestamp;
        LogLevel level;
        std::string message;
        glm::vec4 color;
    };
    
    LogDisplayComponent(const glm::vec2& position, const glm::vec2& size);
    ~LogDisplayComponent();
    
    // Log management
    void add_log_entry(LogLevel level, const std::string& message);
    void set_max_entries(size_t max_entries) { max_entries_ = max_entries; }
    void set_auto_scroll(bool auto_scroll) { auto_scroll_ = auto_scroll; }
    void clear_logs();
    
    // Filtering
    void set_log_level_filter(LogLevel min_level) { min_log_level_ = min_level; }
    void set_text_filter(const std::string& filter) { text_filter_ = filter; }
    
    // UIComponent interface
    void update(float delta_time) override;
    void render(VkCommandBuffer cmd) override;
    void handle_input(const InputEvent& event) override;

private:
    std::deque<LogEntry> log_entries_;
    size_t max_entries_ = 1000;
    bool auto_scroll_ = true;
    LogLevel min_log_level_ = Debug;
    std::string text_filter_;
    float scroll_offset_ = 0.0f;
    
    // Rendering resources
    BufferAllocation text_vertex_buffer_;
    VkPipeline text_pipeline_ = VK_NULL_HANDLE;
    
    void rebuild_text_geometry();
    glm::vec4 get_log_level_color(LogLevel level);
    std::string format_timestamp(const std::chrono::system_clock::time_point& time);
};

// ============================================================================
// Market Data Integration
// ============================================================================

struct ProcessedTrade {
    uint32_t symbol_id;
    double price;
    double size;
    uint64_t timestamp;
    bool is_buy;
    double price_change;
    double volume_weighted_price;
};

struct ProcessedOrderbook {
    uint32_t symbol_id;
    std::array<HotSpine::HotOrderbookLevel, 20> bids;
    std::array<HotSpine::HotOrderbookLevel, 20> asks;
    uint8_t bids_count;
    uint8_t asks_count;
    double spread;
    double mid_price;
    uint64_t timestamp;
};

class MarketDataProcessor {
public:
    MarketDataProcessor(const std::string& shm_name);
    ~MarketDataProcessor();
    
    // Lifecycle management
    void start_processing();
    void stop_processing();
    bool is_running() const { return running_; }
    
    // Data access (thread-safe)
    std::vector<ProcessedTrade> get_recent_trades(uint32_t symbol_id, size_t count = 100);
    std::optional<ProcessedOrderbook> get_latest_orderbook(uint32_t symbol_id);
    
    // Statistics
    uint64_t get_trades_processed() const { return trades_processed_; }
    uint64_t get_orderbooks_processed() const { return orderbooks_processed_; }
    double get_processing_rate() const;

private:
    std::string shm_name_;
    std::atomic<bool> running_{false};
    std::thread processing_thread_;
    
    // Data queues
    moodycamel::ConcurrentQueue<ProcessedTrade> trade_queue_;
    std::unordered_map<uint32_t, ProcessedOrderbook> latest_orderbooks_;
    std::mutex orderbook_mutex_;
    
    // Statistics
    std::atomic<uint64_t> trades_processed_{0};
    std::atomic<uint64_t> orderbooks_processed_{0};
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // Processing methods
    void processing_loop();
    void process_trades();
    void process_orderbooks();
    ProcessedTrade convert_trade(const HotSpine::HotTrade& hot_trade);
    ProcessedOrderbook convert_orderbook(const HotSpine::HotOrderbookSnapshot& hot_orderbook);
};

class DataStreamManager {
public:
    template<typename T>
    using Callback = std::function<void(const T&)>;
    
    // Subscription management
    template<typename T>
    void subscribe(const std::string& channel, Callback<T> callback);
    
    // Data publishing
    void publish_trade_update(const ProcessedTrade& trade);
    void publish_orderbook_update(const ProcessedOrderbook& orderbook);
    void publish_price_change(uint32_t symbol_id, double change_percent);
    
private:
    std::unordered_map<std::string, std::vector<std::function<void(const void*)>>> subscribers_;
    std::mutex subscribers_mutex_;
};

// ============================================================================
// Main Dashboard Class
// ============================================================================

class VulkanDashboard {
public:
    VulkanDashboard(uint32_t width, uint32_t height, const DashboardConfig& config = {});
    ~VulkanDashboard();
    
    // Lifecycle management
    void initialize();
    void main_loop();
    void shutdown();
    
    // Component management
    void add_component(std::unique_ptr<UIComponent> component);
    void remove_component(UIComponent* component);
    
    // Data integration
    void start_market_data_processing();
    void stop_market_data_processing();
    
    // Performance monitoring
    struct PerformanceStats {
        float fps;
        float frame_time_ms;
        float cpu_usage_percent;
        GPUMemoryManager::MemoryStats memory_stats;
        uint64_t ui_elements_rendered;
        float data_latency_ms;
    };
    PerformanceStats get_performance_stats() const;

private:
    // Configuration
    DashboardConfig config_;
    DashboardTheme theme_;
    uint32_t width_, height_;
    
    // X11 window management
    Display* display_ = nullptr;
    Window window_;
    Atom wm_delete_window_;
    
    // Vulkan rendering
    std::unique_ptr<VulkanCore> vulkan_core_;
    
    // UI components
    std::vector<std::unique_ptr<UIComponent>> components_;
    
    // Market data integration
    std::unique_ptr<MarketDataProcessor> market_data_processor_;
    std::unique_ptr<DataStreamManager> data_stream_manager_;
    
    // Performance monitoring
    mutable std::mutex stats_mutex_;
    PerformanceStats current_stats_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
    
    // Frame timing
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    std::deque<float> frame_times_;
    
    // Private methods
    void init_x11();
    void init_vulkan();
    void init_components();
    void setup_data_subscriptions();
    void handle_x11_events();
    void update_components(float delta_time);
    void render_components();
    void update_performance_stats();
    void cleanup_x11();
    
    // Event handlers
    void on_trade_received(const ProcessedTrade& trade);
    void on_orderbook_updated(const ProcessedOrderbook& orderbook);
    void on_window_resize(uint32_t new_width, uint32_t new_height);
};

} // namespace BTQuant