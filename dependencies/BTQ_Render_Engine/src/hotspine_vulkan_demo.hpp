#pragma once

#include <vulkan/vulkan.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
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

// ==== HotSpine structures (must match your C++ code) ====

struct HotSpineHeader {
    uint64_t version;
    uint64_t write_pos;
    uint64_t read_pos;
    uint64_t buffer_size;
    uint32_t lost_count;
    uint8_t  status;
};

struct HotTrade {
    uint64_t ts_exchange;
    uint64_t ts_local;
    double   price;
    double   size;
    uint32_t symbol_id;
    uint8_t  side;      // 0=BUY, 1=SELL
};

// Simple wrapper to read from shared memory ring
class HotSpineReaderSimple {
public:
    explicit HotSpineReaderSimple(const std::string& shm_name = "/btquant_hotspine")
    {
        fd_ = shm_open(shm_name.c_str(), O_RDONLY, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("shm_open failed for " + shm_name);
        }

        struct stat sb{};
        if (fstat(fd_, &sb) < 0) {
            close(fd_);
            throw std::runtime_error("fstat failed on HotSpine shm");
        }
        total_size_ = static_cast<size_t>(sb.st_size);

        void* ptr = mmap(nullptr, total_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (ptr == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed on HotSpine shm");
        }

        header_ = static_cast<HotSpineHeader*>(ptr);
        const size_t header_size = sizeof(HotSpineHeader);
        const size_t trades_bytes = total_size_ - header_size;
        capacity_ = trades_bytes / sizeof(HotTrade);
        buffer_ = reinterpret_cast<HotTrade*>(static_cast<char*>(ptr) + header_size);

        last_read_pos_ = header_->read_pos;
    }

    ~HotSpineReaderSimple() {
        if (header_) {
            munmap(header_, total_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    // Non-blocking poll: returns last seen price if any trades were read
    bool poll_latest_price(double& out_price) {
        uint64_t write_pos = __atomic_load_n(&header_->write_pos, __ATOMIC_ACQUIRE);
        uint64_t read_pos  = last_read_pos_;
        bool got = false;
        while (read_pos != write_pos) {
            const HotTrade& t = buffer_[read_pos % capacity_];
            out_price = t.price;
            read_pos = (read_pos + 1) % capacity_;
            got = true;
        }
        last_read_pos_ = read_pos;
        return got;
    }

private:
    int              fd_{-1};
    size_t           total_size_{0};
    HotSpineHeader*  header_{nullptr};
    HotTrade*        buffer_{nullptr};
    size_t           capacity_{0};
    uint64_t         last_read_pos_{0};
};

// ==== Minimal Vulkan + X11 renderer ====

class VulkanHotSpineDemo {
public:
    VulkanHotSpineDemo(uint32_t width, uint32_t height)
        : width_(width), height_(height), hotspine_("/btquant_hotspine")
    {
        init_x11();
        init_vulkan();
    }

    ~VulkanHotSpineDemo() {
        vkDeviceWaitIdle(device_);

        // Cleanup Vulkan
        for (auto fb : swapchain_framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        vkDestroyRenderPass(device_, render_pass_, nullptr);
        for (auto view : swapchain_image_views_) vkDestroyImageView(device_, view, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);

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
        double last_price = 0.0;
        double base_price = 0.0;

        while (running) {
            // X11 event pump
            while (XPending(display_)) {
                XEvent e;
                XNextEvent(display_, &e);
                if (e.type == ClientMessage) {
                    running = false;
                }
            }

            // Poll HotSpine
            double price;
            if (hotspine_.poll_latest_price(price)) {
                last_price = price;
                if (base_price == 0.0) base_price = price;
            }

            draw_frame(last_price, base_price);
        }
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
    size_t                   current_frame_{0};

    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    // HotSpine
    HotSpineReaderSimple hotspine_;

private:
    // === X11 setup ===
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

        XStoreName(display_, window_, "BTQuant HotSpine Vulkan Demo");
        XMapWindow(display_, window_);

        wm_delete_window_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display_, window_, &wm_delete_window_, 1);
    }

    // === Vulkan helpers ===
    void init_vulkan() {
        create_instance();
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_swapchain();
        create_image_views();
        create_render_pass();
        create_pipeline();
        create_framebuffers();
        create_command_pool();
        create_command_buffers();
        create_sync_objects();
    }

    void create_instance() {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "BTQuant Vulkan Demo";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "No Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_1;

        const char* extensions[] = {
            "VK_KHR_surface",
            "VK_KHR_xlib_surface"
        };

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledExtensionCount = 2;
        create_info.ppEnabledExtensionNames = extensions;

        if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }
    }

    void create_surface() {
        VkXlibSurfaceCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        create_info.dpy = display_;
        create_info.window = window_;
        if (vkCreateXlibSurfaceKHR(instance_, &create_info, nullptr, &surface_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Xlib surface");
        }
    }

    void pick_physical_device() {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        if (device_count == 0) throw std::runtime_error("No Vulkan devices");

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
        physical_device_ = devices[0]; // pick first for now
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
    }

    uint32_t find_graphics_queue_family() {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, nullptr);
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, props.data());

        for (uint32_t i = 0; i < count; ++i) {
            if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                return i;
            }
        }
        throw std::runtime_error("No graphics queue family");
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
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM) {
                chosen = f; break;
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

        uint32_t queue_family_index = find_graphics_queue_family();
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.preTransform = caps.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR; // vsync
        create_info.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swapchain");
        }

        vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
        swapchain_images_.resize(image_count);
        vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());
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
    }

    // For brevity, we use a fixed-function pipeline + no shaders, so this is
    // a stub. In practice, you would create simple vertex+fragment shaders.
    void create_pipeline() {
        // Minimal pipeline with no vertex buffer (fullscreen quad via vertex shader)
        // You can stub this to a fixed color first.
        // For the prototype, we skip full shader code; just assume pipeline_ is valid.
        // In real code, you’d create VkShaderModule from SPIR-V and configure the pipeline.
        // Here we leave it as an exercise to plug your existing shader utils.
        // To keep this answer compact, I’m not dumping full SPIR-V here.
        pipeline_layout_ = VK_NULL_HANDLE;
        pipeline_ = VK_NULL_HANDLE;
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
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        create_info.queueFamilyIndex = find_graphics_queue_family();

        if (vkCreateCommandPool(device_, &create_info, nullptr, &command_pool_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }
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
    }

    void record_command_buffer(VkCommandBuffer cmd, uint32_t image_index, float price_norm) {
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(cmd, &begin_info);

        VkClearValue clear_color{};
        clear_color.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass = render_pass_;
        rp_info.framebuffer = swapchain_framebuffers_[image_index];
        rp_info.renderArea.offset = { 0, 0 };
        rp_info.renderArea.extent = swapchain_extent_;
        rp_info.clearValueCount = 1;
        rp_info.pClearValues = &clear_color;

        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

        // Here you’d bind pipeline and draw a rectangle whose height is price_norm.
        // For now, this is left as a stub; you can plug in your own shader/vertex data.

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

    void draw_frame(double last_price, double base_price) {
        vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);

        uint32_t image_index;
        vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX,
            image_available_semaphores_[current_frame_],
            VK_NULL_HANDLE, &image_index
        );

        vkResetCommandBuffer(command_buffers_[image_index], 0);

        float price_norm = 0.5f;
        if (base_price > 0.0) {
            double rel = (last_price - base_price) / base_price;
            // Clamp to [-5%, +5%] mapped to [0,1]
            if (rel < -0.05) rel = -0.05;
            if (rel >  0.05) rel =  0.05;
            price_norm = static_cast<float>((rel + 0.05) / 0.10);
        }

        record_command_buffer(command_buffers_[image_index], image_index, price_norm);

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
};
