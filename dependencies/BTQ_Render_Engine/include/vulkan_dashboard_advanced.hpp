#pragma once

#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

// GLM for math operations
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Standard library includes
#include <string>
#include <vector>
#include <cstdio>
#include <chrono>
#include <thread>

class VulkanDashboard {
public:
    VulkanDashboard(uint32_t width, uint32_t height) : width_(width), height_(height) {
        init_x11();
    }

    ~VulkanDashboard() {
        if (window_) {
            XDestroyWindow(display_, window_);
        }
        if (display_) {
            XCloseDisplay(display_);
        }
    }

    void main_loop() {
        bool running = true;
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

            // Placeholder for rendering
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }

        fprintf(stderr, "[Dashboard] Main loop exit\n");
    }

private:
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

    uint32_t width_;
    uint32_t height_;
    Display* display_;
    Window window_;
    Atom wm_delete_window_;
};