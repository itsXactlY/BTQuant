#pragma once

#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

// Dear ImGui
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_glfw.h>  // For input handling compatibility

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <array>
#include <algorithm>
#include <map>
#include <deque>
#include <unordered_map>
#include <fstream>
#include <sstream>

#include "shader_spirv.hpp"

// Debug callback function
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    (void)pUserData;
    (void)messageType;

    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(stderr, "[Vulkan Debug] %s\n", pCallbackData->pMessage);
    }

    return VK_FALSE;
}

// ==== Configuration from market_data_collector ====

// Dynamic symbol loading
struct Symbol {
    std::string name;
    uint32_t symbol_id;
};

// ==== HotSpine structures (from project's hotspine_layout.hpp) ====

namespace HotSpine {

// Magic number for HotSpine shared memory (0x42545155 = "BTQU" in little endian)
constexpr uint32_t HOTSPINE_MAGIC = 0x42545155;

// Shared memory layout constants
constexpr uint64_t HOTSPINE_VERSION = 2;
constexpr uint64_t DEFAULT_CAPACITY = 1000000;          // 1 million trades
constexpr uint64_t DEFAULT_ORDERBOOK_CAPACITY = 100000; // 100k orderbooks
constexpr size_t HEADER_SIZE = 4096;                    // 4KB for header

// Shared memory header structure
struct SharedMemoryHeader {
  uint32_t magic;       // Magic number for validation (0x42545155)
  uint32_t version;     // Version number
  uint64_t capacity;    // Number of trade entries in buffer
  uint64_t write_index; // Write position (next slot to write)
  uint64_t read_index;  // Read position (next slot to read)
  uint64_t lost_count;  // Number of lost trades due to buffer overflow

  // Orderbook fields (replaces previous padding to match Python side)
  uint64_t orderbook_write_index;
  uint64_t orderbook_read_index;
  uint64_t orderbook_lost_count;
  uint64_t orderbook_capacity; // Number of orderbook entries in buffer

  uint8_t padding[8]; // remaining padding for alignment
};

// Trade data structure (must match between writer and reader)
struct HotTrade {
  uint64_t ts_exchange; // exchange timestamp in microseconds
  uint64_t ts_local;    // local receive timestamp in microseconds
  double price;
  double size;
  uint32_t symbol_id; // symbol ID (hash or mapping)
  uint8_t side;       // 0=buy, 1=sell
  uint8_t padding[3]; // Explicit padding for 8-byte alignment
};

} // namespace HotSpine

using HotSpineHeader = HotSpine::SharedMemoryHeader;
using HotTrade = HotSpine::HotTrade;

// ==== Fast aggregation structures (cache-line aligned) ====

struct alignas(64) SymbolStats {
    double last_price = 0.0;
    double vwap = 0.0;
    double bid_ask_ratio = 0.0;  // buy_count / sell_count
    double momentum = 0.0;        // price - previous_price
    double volatility = 0.0;      // rolling stdev
    uint64_t total_volume = 0;
    uint32_t buy_count = 0;
    uint32_t sell_count = 0;
    float color_r = 0.5f;         // normalized color for rendering
    float color_g = 0.5f;
    float color_b = 0.1f;
};

struct alignas(64) CandleBar {
    double open, high, low, close;
    uint64_t volume;
    uint64_t ts;
};

// ==== Advanced HotSpine Reader with multi-symbol aggregation ====

class HotSpineReaderAdvanced {
public:
    explicit HotSpineReaderAdvanced(const std::string& shm_name = "/btquant_hotspine")
    {
        fd_ = shm_open(shm_name.c_str(), O_RDONLY, 0644);
        if (fd_ < 0) {
            perror("shm_open");
            fprintf(stderr, "Shared memory not available for %s, continuing with defaults\n", shm_name.c_str());
            available_ = false;
            return;
        }

        struct stat sb{};
        if (fstat(fd_, &sb) < 0) {
            perror("fstat");
            close(fd_);
            throw std::runtime_error("fstat failed on HotSpine shm");
        }
        total_size_ = static_cast<size_t>(sb.st_size);

        void* ptr = mmap(nullptr, total_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (ptr == MAP_FAILED) {
            perror("mmap");
            close(fd_);
            throw std::runtime_error("mmap failed on HotSpine shm");
        }

        header_ = static_cast<HotSpineHeader*>(ptr);
        // Validate magic
        if (header_->magic != HotSpine::HOTSPINE_MAGIC) {
            fprintf(stderr, "Invalid HotSpine magic: 0x%x, expected 0x%x\n", header_->magic, HotSpine::HOTSPINE_MAGIC);
            munmap(ptr, total_size_);
            close(fd_);
            available_ = false;
            return;
        }

        const size_t header_size = HotSpine::HEADER_SIZE;
        capacity_ = header_->capacity;
        buffer_ = reinterpret_cast<HotTrade*>(static_cast<char*>(ptr) + header_size);

        last_read_pos_ = header_->read_index;

        // Initialize symbol stats - we'll resize dynamically
        stats_.clear();

        fprintf(stderr, "[HotSpine] Attached: %zu trades capacity, version %u\n", capacity_, header_->version);
    }

    ~HotSpineReaderAdvanced() {
        if (header_) {
            munmap(header_, total_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    // Non-blocking poll: updates stats_ for all symbols
    void poll_and_aggregate() {
        if (!available_) return;
        uint64_t write_pos = __atomic_load_n(&header_->write_index, __ATOMIC_ACQUIRE);
        uint64_t read_pos  = last_read_pos_;

        while (read_pos != write_pos) {
            const HotTrade& t = buffer_[read_pos % capacity_];

            // Dynamic symbol handling
            SymbolStats& stat = stats_[t.symbol_id]; // map will create if not exists

            // Update running stats
            double prev = stat.last_price;
            stat.last_price = t.price;
            stat.momentum = (prev > 0.0) ? (t.price - prev) : 0.0;
            stat.total_volume += static_cast<uint64_t>(t.size);

            if (t.side == 0) stat.buy_count++;
            else             stat.sell_count++;

            // Update bid/ask ratio
            if (stat.sell_count > 0) {
                stat.bid_ask_ratio = static_cast<double>(stat.buy_count) / stat.sell_count;
            }

            // Compute color: green for up, red for down
            if (stat.momentum > 0.0) {
                stat.color_r = 0.0f;
                stat.color_g = 1.0f;
                stat.color_b = 0.0f;
            } else if (stat.momentum < 0.0) {
                stat.color_r = 1.0f;
                stat.color_g = 0.0f;
                stat.color_b = 0.0f;
            } else {
                stat.color_r = 0.5f;
                stat.color_g = 0.5f;
                stat.color_b = 0.1f;
            }

            read_pos = (read_pos + 1) % capacity_;
        }

        last_read_pos_ = read_pos;
    }

    const SymbolStats& get_symbol_stats(uint32_t symbol_id) const {
        auto it = stats_.find(symbol_id);
        if (it == stats_.end()) {
            static SymbolStats default_stats{};
            return default_stats;
        }
        return it->second;
    }

    // Get all symbol IDs that have data
    std::vector<uint32_t> get_active_symbols() const {
        std::vector<uint32_t> symbols;
        for (const auto& pair : stats_) {
            symbols.push_back(pair.first);
        }
        return symbols;
    }

    bool is_healthy() const {
        return header_ && header_->magic == HotSpine::HOTSPINE_MAGIC;
    }

private:
    bool             available_{true};
    int              fd_{-1};
    size_t           total_size_{0};
    HotSpineHeader*  header_{nullptr};
    HotTrade*        buffer_{nullptr};
    size_t           capacity_{0};
    uint64_t         last_read_pos_{0};
    std::unordered_map<uint32_t, SymbolStats> stats_;
};

// ==== Vulkan Dashboard with multi-panel layout ====

class VulkanDashboard {
public:
    VulkanDashboard(uint32_t width, uint32_t height)
        : width_(width), height_(height), hotspine_("/btquant_hotspine")
    {
        load_symbols();
        init_x11();
        init_vulkan();
        fprintf(stderr, "[Vulkan Dashboard] Initialization complete. Rendering %zu symbols.\n", symbols_.size());
    }

    ~VulkanDashboard() {
        vkDeviceWaitIdle(device_);

        // Cleanup ImGui
        if (imgui_initialized_) {
            ImGui_ImplVulkan_Shutdown();
            ImGui::DestroyContext();
        }
        if (imgui_descriptor_pool_) vkDestroyDescriptorPool(device_, imgui_descriptor_pool_, nullptr);

        // Cleanup shaders
        if (vert_shader_) vkDestroyShaderModule(device_, vert_shader_, nullptr);
        if (frag_shader_) vkDestroyShaderModule(device_, frag_shader_, nullptr);

        // Cleanup Vulkan
        for (auto fb : swapchain_framebuffers_) {
            if (fb) vkDestroyFramebuffer(device_, fb, nullptr);
        }
        for (auto sem : image_available_semaphores_) {
            if (sem) vkDestroySemaphore(device_, sem, nullptr);
        }
        for (auto sem : render_finished_semaphores_) {
            if (sem) vkDestroySemaphore(device_, sem, nullptr);
        }
        for (auto fence : in_flight_fences_) {
            if (fence) vkDestroyFence(device_, fence, nullptr);
        }
        if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (render_pass_) vkDestroyRenderPass(device_, render_pass_, nullptr);
        for (auto view : swapchain_image_views_) {
            if (view) vkDestroyImageView(device_, view, nullptr);
        }
        if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
        if (device_) vkDestroyDevice(device_, nullptr);
        if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (instance_) vkDestroyInstance(instance_, nullptr);

        // Cleanup X11
        if (window_) {
            XDestroyWindow(display_, window_);
        }
        if (display_) {
            XCloseDisplay(display_);
        }
    }

    void main_loop() {
        bool running = true;
        uint32_t frame_count = 0;
        auto last_fps_time = std::chrono::high_resolution_clock::now();

        fprintf(stderr, "[Dashboard] Main loop start\n");

        while (running) {
            // X11 event pump
            while (XPending(display_)) {
                XEvent e;
                XNextEvent(display_, &e);
                if (e.type == ClientMessage) {
                    if ((Atom)e.xclient.data.l[0] == wm_delete_window_) {
                        running = false;
                    }
                }
            }

            // Poll HotSpine (fast path)
            hotspine_.poll_and_aggregate();

            // Render frame
            draw_frame();
            frame_count++;

            // FPS logging
            if (frame_count % 60 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();
                double fps = (frame_count * 1000.0) / elapsed;
                fprintf(stderr, "[Dashboard] FPS: %.1f", fps);
                // Log first few symbols
                for (size_t i = 0; i < std::min(size_t(3), symbols_.size()); ++i) {
                    const auto& sym = symbols_[i];
                    double price = hotspine_.get_symbol_stats(sym.symbol_id).last_price;
                    fprintf(stderr, " | %s: %.2f", sym.name.c_str(), price);
                }
                fprintf(stderr, "\n");
                last_fps_time = now;
            }
        }

        fprintf(stderr, "[Dashboard] Main loop exit\n");
    }

private:
    // X11
    Display*  display_{nullptr};
    Window    window_{0};
    Atom      wm_delete_window_{0};
    uint32_t  width_{1280};
    uint32_t  height_{720};

    // Vulkan
    VkInstance               instance_{VK_NULL_HANDLE};
    VkPhysicalDevice         physical_device_{VK_NULL_HANDLE};
    VkDevice                 device_{VK_NULL_HANDLE};
    VkQueue                  graphics_queue_{VK_NULL_HANDLE};
    VkQueue                  present_queue_{VK_NULL_HANDLE};
    VkSurfaceKHR             surface_{VK_NULL_HANDLE};
    VkSwapchainKHR           swapchain_{VK_NULL_HANDLE};
    VkFormat                 swapchain_image_format_{};
    VkExtent2D               swapchain_extent_{};
    std::vector<VkImage>     swapchain_images_;
    std::vector<VkImageView> swapchain_image_views_;
    std::vector<VkFramebuffer> swapchain_framebuffers_;
    VkRenderPass             render_pass_{VK_NULL_HANDLE};
    VkPipelineLayout         pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline               pipeline_{VK_NULL_HANDLE};
    VkCommandPool            command_pool_{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence>     in_flight_fences_;
    VkShaderModule           vert_shader_{VK_NULL_HANDLE};
    VkShaderModule           frag_shader_{VK_NULL_HANDLE};
    size_t                   current_frame_{0};

    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    // ImGui
    VkDescriptorPool         imgui_descriptor_pool_{VK_NULL_HANDLE};
    bool                     imgui_initialized_{false};

    // HotSpine
    HotSpineReaderAdvanced hotspine_;
    std::vector<Symbol> symbols_;

private:
    void load_symbols() {
        std::ifstream file("/dev/shm/btquant_symbols.json");
        if (!file.is_open()) {
            fprintf(stderr, "[Dashboard] Could not open /dev/shm/btquant_symbols.json, using defaults\n");
            // Fallback to some defaults
            symbols_ = {
                {"BTCUSDT", 10008},
                {"ETHUSDT", 10006},
                {"BNBUSDT", 10017},
                {"ADAUSDT", 10016},
                {"SOLUSDT", 10007},
                {"DOGEUSDT", 10009}
            };
            return;
        }

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        // Simple JSON parsing for {"symbols": [{"id": 10008, "exchange": "binance", "symbol": "BTCUSDT"}, ...]}
        size_t pos = content.find("\"symbols\":");
        if (pos == std::string::npos) return;

        pos = content.find('[', pos);
        if (pos == std::string::npos) return;

        size_t end = content.find(']', pos);
        if (end == std::string::npos) return;

        std::string symbols_str = content.substr(pos + 1, end - pos - 1);
        std::stringstream ss(symbols_str);
        std::string item;
        while (std::getline(ss, item, '}')) {
            if (item.empty()) continue;
            // Find "id": and "symbol":
            size_t id_pos = item.find("\"id\":");
            size_t sym_pos = item.find("\"symbol\":");
            if (id_pos != std::string::npos && sym_pos != std::string::npos) {
                size_t id_start = item.find(':', id_pos) + 1;
                size_t id_end = item.find(',', id_start);
                uint32_t id = std::stoul(item.substr(id_start, id_end - id_start));

                size_t sym_start = item.find('\"', sym_pos + 10) + 1;
                size_t sym_end = item.find('\"', sym_start);
                std::string symbol = item.substr(sym_start, sym_end - sym_start);

                symbols_.push_back({symbol, id});
            }
        }

        fprintf(stderr, "[Dashboard] Loaded %zu symbols from shared memory\n", symbols_.size());
    }

    void init_x11() {
        display_ = XOpenDisplay(nullptr);
        if (!display_) {
            throw std::runtime_error("Failed to open X display");
        }

        int screen = DefaultScreen(display_);
        Window root = RootWindow(display_, screen);

        XSetWindowAttributes swa{};
        swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;
        window_ = XCreateWindow(
            display_, root,
            0, 0, width_, height_, 0,
            CopyFromParent, InputOutput, CopyFromParent,
            CWEventMask, &swa
        );

        XStoreName(display_, window_, "BTQuant Advanced Dashboard");
        XMapWindow(display_, window_);

        wm_delete_window_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display_, window_, &wm_delete_window_, 1);

        fprintf(stderr, "[X11] Window created: %ux%u\n", width_, height_);
    }

    void init_vulkan() {
        create_instance();
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_swapchain();
        create_image_views();
        create_render_pass();
        create_shaders();
        create_pipeline();
        create_framebuffers();
        create_command_pool();
        create_command_buffers();
        create_sync_objects();
        create_imgui_descriptor_pool();
        init_imgui();
    }

    void create_instance() {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "BTQuant Advanced Dashboard";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "No Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_1;

        const char* extensions[] = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME
        };

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledExtensionCount = 3;
        create_info.ppEnabledExtensionNames = extensions;

        VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
        debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_create_info.pfnUserCallback = debug_callback;
        debug_create_info.pUserData = nullptr;

        create_info.pNext = &debug_create_info;

        if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }
        fprintf(stderr, "[Vulkan] Instance created\n");
    }

    void create_surface() {
        VkXlibSurfaceCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        create_info.dpy = display_;
        create_info.window = window_;
        if (vkCreateXlibSurfaceKHR(instance_, &create_info, nullptr, &surface_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Xlib surface");
        }
        fprintf(stderr, "[Vulkan] Surface created\n");
    }

    void pick_physical_device() {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        if (device_count == 0) throw std::runtime_error("No Vulkan devices");

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
        physical_device_ = devices[0];

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_device_, &props);
        fprintf(stderr, "[Vulkan] Selected device: %s\n", props.deviceName);
    }

    void create_logical_device() {
        uint32_t queue_family_index = find_graphics_queue_family();

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_index;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = 1;
        create_info.pQueueCreateInfos = &queue_info;
        create_info.pEnabledFeatures = &features;

        const char* extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        create_info.enabledExtensionCount = 1;
        create_info.ppEnabledExtensionNames = extensions;

        if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device");
        }

        vkGetDeviceQueue(device_, queue_family_index, 0, &graphics_queue_);
        present_queue_ = graphics_queue_;
        fprintf(stderr, "[Vulkan] Logical device created\n");
    }

    uint32_t find_graphics_queue_family() {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, nullptr);
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, props.data());

        for (uint32_t i = 0; i < count; ++i) {
            VkBool32 present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_, &present_support);
            if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present_support) {
                return i;
            }
        }
        throw std::runtime_error("No graphics queue family with presentation support");
    }

    void create_swapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &caps);

        uint32_t format_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, formats.data());

        VkSurfaceFormatKHR chosen = formats[0];
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = f;
                break;
            }
        }

        swapchain_image_format_ = chosen.format;
        swapchain_extent_.width = width_;
        swapchain_extent_.height = height_;

        uint32_t image_count = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
            image_count = caps.maxImageCount;
        }

        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface_;
        create_info.minImageCount = image_count;
        create_info.imageFormat = swapchain_image_format_;
        create_info.imageColorSpace = chosen.colorSpace;
        create_info.imageExtent = swapchain_extent_;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.preTransform = caps.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        create_info.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swapchain");
        }

        vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
        swapchain_images_.resize(image_count);
        vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());

        fprintf(stderr, "[Vulkan] Swapchain created (%u images)\n", image_count);
    }

    void create_image_views() {
        swapchain_image_views_.resize(swapchain_images_.size());

        for (size_t i = 0; i < swapchain_images_.size(); ++i) {
            VkImageViewCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            create_info.image = swapchain_images_[i];
            create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            create_info.format = swapchain_image_format_;
            create_info.components = {
                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
            };
            create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            create_info.subresourceRange.baseMipLevel = 0;
            create_info.subresourceRange.levelCount = 1;
            create_info.subresourceRange.baseArrayLayer = 0;
            create_info.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device_, &create_info, nullptr, &swapchain_image_views_[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views");
            }
        }
        fprintf(stderr, "[Vulkan] Image views created\n");
    }

    void create_render_pass() {
        VkAttachmentDescription color_attachment{};
        color_attachment.format = swapchain_image_format_;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_ref{};
        color_ref.attachment = 0;
        color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_ref;

        VkRenderPassCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        create_info.attachmentCount = 1;
        create_info.pAttachments = &color_attachment;
        create_info.subpassCount = 1;
        create_info.pSubpasses = &subpass;

        if (vkCreateRenderPass(device_, &create_info, nullptr, &render_pass_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass");
        }
        fprintf(stderr, "[Vulkan] Render pass created\n");
    }

    void create_shaders() {
        bool vert_ok = create_shader_module_safe(device_, VERTEX_SHADER_SPIRV, VERTEX_SHADER_SIZE, vert_shader_);
        bool frag_ok = create_shader_module_safe(device_, FRAGMENT_SHADER_SPIRV, FRAGMENT_SHADER_SIZE, frag_shader_);
        if (!vert_ok || !frag_ok) {
            fprintf(stderr, "[Vulkan] Shader module creation failed: vert=%d frag=%d\n", vert_ok, frag_ok);
            throw std::runtime_error("Failed to create shader modules");
        }
        fprintf(stderr, "[Vulkan] Shaders created (vert size=%zu, frag size=%zu)\n", VERTEX_SHADER_SIZE, FRAGMENT_SHADER_SIZE);
    }

    void create_pipeline() {
        VkPipelineShaderStageCreateInfo vert_stage{};
        vert_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vert_stage.module = vert_shader_;
        vert_stage.pName = "main";

        VkPipelineShaderStageCreateInfo frag_stage{};
        frag_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_stage.module = frag_shader_;
        frag_stage.pName = "main";

        VkPipelineShaderStageCreateInfo stages[] = {vert_stage, frag_stage};

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

        VkViewport viewport{};
        viewport.x = 0;
        viewport.y = 0;
        viewport.width = static_cast<float>(swapchain_extent_.width);
        viewport.height = static_cast<float>(swapchain_extent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchain_extent_;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorblend_attachment{};
        colorblend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorblend_attachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorblend{};
        colorblend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorblend.logicOpEnable = VK_FALSE;
        colorblend.attachmentCount = 1;
        colorblend.pAttachments = &colorblend_attachment;

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = stages;
        pipeline_info.pVertexInputState = &vertex_input;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &colorblend;
        pipeline_info.layout = pipeline_layout_;
        pipeline_info.renderPass = render_pass_;
        pipeline_info.subpass = 0;

        VkResult result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
        if (result != VK_SUCCESS) {
            fprintf(stderr, "[Vulkan] vkCreateGraphicsPipelines failed with VkResult: %d\n", result);
            throw std::runtime_error("Failed to create graphics pipeline");
        }

        fprintf(stderr, "[Vulkan] Graphics pipeline created\n");
    }

    void create_framebuffers() {
        swapchain_framebuffers_.resize(swapchain_image_views_.size());

        for (size_t i = 0; i < swapchain_image_views_.size(); ++i) {
            VkImageView attachments[] = { swapchain_image_views_[i] };

            VkFramebufferCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            create_info.renderPass = render_pass_;
            create_info.attachmentCount = 1;
            create_info.pAttachments = attachments;
            create_info.width = swapchain_extent_.width;
            create_info.height = swapchain_extent_.height;
            create_info.layers = 1;

            if (vkCreateFramebuffer(device_, &create_info, nullptr, &swapchain_framebuffers_[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer");
            }
        }
        fprintf(stderr, "[Vulkan] Framebuffers created\n");
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        create_info.queueFamilyIndex = find_graphics_queue_family();

        if (vkCreateCommandPool(device_, &create_info, nullptr, &command_pool_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }
        fprintf(stderr, "[Vulkan] Command pool created\n");
    }

    void create_command_buffers() {
        command_buffers_.resize(swapchain_framebuffers_.size());

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool_;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());

        if (vkAllocateCommandBuffers(device_, &alloc_info, command_buffers_.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers");
        }
        fprintf(stderr, "[Vulkan] Command buffers allocated\n");
    }

    void create_sync_objects() {
        image_available_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
        render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
        in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(device_, &sem_info, nullptr, &image_available_semaphores_[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device_, &sem_info, nullptr, &render_finished_semaphores_[i]) != VK_SUCCESS ||
                vkCreateFence(device_, &fence_info, nullptr, &in_flight_fences_[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create sync objects");
            }
        }
        fprintf(stderr, "[Vulkan] Sync objects created\n");
    }

    void create_imgui_descriptor_pool() {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000;
        pool_info.poolSizeCount = std::size(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;

        if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &imgui_descriptor_pool_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create ImGui descriptor pool");
        }
        fprintf(stderr, "[ImGui] Descriptor pool created\n");
    }

    void init_imgui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();

        ImGui_ImplVulkan_InitInfo init_info{};
        init_info.Instance = instance_;
        init_info.PhysicalDevice = physical_device_;
        init_info.Device = device_;
        init_info.QueueFamily = find_graphics_queue_family();
        init_info.Queue = graphics_queue_;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = imgui_descriptor_pool_;
        init_info.Subpass = 0;
        init_info.MinImageCount = 2;
        init_info.ImageCount = static_cast<uint32_t>(swapchain_images_.size());
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = nullptr;

        fprintf(stderr, "[ImGui] Initializing Vulkan backend with render pass\n");
        ImGui_ImplVulkan_Init(&init_info, render_pass_);

        // Upload fonts
        VkCommandBuffer command_buffer = begin_single_time_commands();
        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
        end_single_time_commands(command_buffer);

        imgui_initialized_ = true;
        fprintf(stderr, "[ImGui] Initialized\n");
    }

    VkCommandBuffer begin_single_time_commands() {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = command_pool_;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer command_buffer;
        vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(command_buffer, &begin_info);
        return command_buffer;
    }

    void end_single_time_commands(VkCommandBuffer command_buffer) {
        vkEndCommandBuffer(command_buffer);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphics_queue_);

        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
    }

    void record_command_buffer(VkCommandBuffer cmd, uint32_t image_index) {
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(cmd, &begin_info);

        // Clear the entire framebuffer
        VkClearValue clear_color{};
        clear_color.color = { { 0.1f, 0.1f, 0.1f, 1.0f } };

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass = render_pass_;
        rp_info.framebuffer = swapchain_framebuffers_[image_index];
        rp_info.renderArea.offset = { 0, 0 };
        rp_info.renderArea.extent = swapchain_extent_;
        rp_info.clearValueCount = 1;
        rp_info.pClearValues = &clear_color;

        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

        // Render ImGui
        if (imgui_initialized_) {
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

    void draw_frame() {
        vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);

        uint32_t image_index;
        VkResult result = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX,
            image_available_semaphores_[current_frame_],
            VK_NULL_HANDLE, &image_index
        );
        if (result != VK_SUCCESS) {
            fprintf(stderr, "vkAcquireNextImageKHR failed: %d\n", result);
            return;
        }

        // Build ImGui UI
        if (imgui_initialized_) {
            RenderDashboardUI();
        }

        vkResetCommandBuffer(command_buffers_[image_index], 0);
        record_command_buffer(command_buffers_[image_index], image_index);

        VkSemaphore wait_semaphores[] = { image_available_semaphores_[current_frame_] };
        VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore signal_semaphores[] = { render_finished_semaphores_[current_frame_] };

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = wait_semaphores;
        submit_info.pWaitDstStageMask = wait_stages;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffers_[image_index];
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = signal_semaphores;

        if (vkQueueSubmit(graphics_queue_, 1, &submit_info, in_flight_fences_[current_frame_]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw");
        }

        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = signal_semaphores;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain_;
        present_info.pImageIndices = &image_index;

        vkQueuePresentKHR(present_queue_, &present_info);

        current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void RenderDashboardUI() {
        // Start ImGui frame
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(swapchain_extent_.width, swapchain_extent_.height);
        io.DeltaTime = 1.0f / 60.0f; // Placeholder
    
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();

        // Set up full-screen window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(width_, height_));
        ImGui::Begin("BTQuant Dashboard", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

        // Top status bar
        ImGui::BeginChild("TopBar", ImVec2(0, 50), true);
        ImGui::Columns(3, "StatusColumns", false);

        // Latency
        ImGui::Text("Latency: %.1f ms", 1.2f); // Placeholder
        ImGui::NextColumn();

        // FPS
        ImGui::Text("FPS: %.1f", 60.0f); // Placeholder
        ImGui::NextColumn();

        // Connection status
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "● Connected");
        ImGui::Columns(1);
        ImGui::EndChild();

        // Main content area
        ImGui::BeginChild("MainContent", ImVec2(0, 0), true);

        // Left: Market data table
        ImGui::BeginChild("MarketData", ImVec2(ImGui::GetWindowWidth() * 0.7f, 0), true);
        if (ImGui::BeginTable("MarketTable", 5, ImGuiTableFlags_Sortable | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn("Symbol", ImGuiTableColumnFlags_DefaultSort);
            ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_DefaultSort);
            ImGui::TableSetupColumn("Chg%", ImGuiTableColumnFlags_DefaultSort);
            ImGui::TableSetupColumn("Vol", ImGuiTableColumnFlags_DefaultSort);
            ImGui::TableSetupColumn("Spread", ImGuiTableColumnFlags_DefaultSort);
            ImGui::TableHeadersRow();

            // Populate table with symbol data
            for (const auto& sym : symbols_) {
                const SymbolStats& stat = hotspine_.get_symbol_stats(sym.symbol_id);
                if (stat.last_price > 0.0) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%s", sym.name.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.2f", stat.last_price);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.2f%%", stat.momentum * 100.0);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.0lu", stat.total_volume);
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%.4f", 0.0001f); // Placeholder spread
                }
            }
            ImGui::EndTable();
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Right: Heatmap and logs
        ImGui::BeginChild("RightPanel", ImVec2(0, 0), false);

        // Heatmap widget
        ImGui::BeginChild("Heatmap", ImVec2(0, ImGui::GetWindowHeight() * 0.6f), true);
        ImGui::Text("Momentum Heatmap");
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size = ImVec2(ImGui::GetContentRegionAvail().x, 200);

        // Draw heatmap tiles
        int tiles_x = 10;
        int tiles_y = 6;
        float tile_w = canvas_size.x / tiles_x;
        float tile_h = canvas_size.y / tiles_y;

        int tile_idx = 0;
        for (int y = 0; y < tiles_y; ++y) {
            for (int x = 0; x < tiles_x; ++x) {
                if (static_cast<size_t>(tile_idx) < symbols_.size()) {
                    const SymbolStats& stat = hotspine_.get_symbol_stats(symbols_[tile_idx].symbol_id);
                    ImU32 color = IM_COL32(
                        (int)(stat.color_r * 255),
                        (int)(stat.color_g * 255),
                        (int)(stat.color_b * 255),
                        255
                    );
                    draw_list->AddRectFilled(
                        ImVec2(canvas_pos.x + x * tile_w, canvas_pos.y + y * tile_h),
                        ImVec2(canvas_pos.x + (x + 1) * tile_w, canvas_pos.y + (y + 1) * tile_h),
                        color
                    );
                }
                tile_idx++;
            }
        }
        ImGui::Dummy(canvas_size);
        ImGui::EndChild();

        // Log console
        ImGui::BeginChild("Logs", ImVec2(0, 0), true);
        ImGui::Text("HotSpine Logs");
        ImGui::BeginChild("LogScroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
        // Placeholder logs
        ImGui::Text("[INFO] Connected to HotSpine");
        ImGui::Text("[INFO] Processing BTCUSDT trades");
        ImGui::Text("[WARN] High latency detected");
        ImGui::EndChild();
        ImGui::EndChild();

        ImGui::EndChild();

        ImGui::EndChild();

        ImGui::End();

        // Render ImGui
        ImGui::Render();
    }
};