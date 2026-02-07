/**
 * BTQuant UI Application - Main Entry Point
 * 
 * This application demonstrates the UI components including Tutorial, Tooltips, and Haptic Feedback
 * that were previously implemented but not integrated into the main application.
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

// Include the UI components that need to be integrated
#include "config/config_loader.hpp"
#include "detectors/liquidity_imbalance_detector.hpp"
#include "detectors/spoofing_detector.hpp"
#include "detectors/spread_arbitrage_detector.hpp"
#include "detectors/stop_hunt_detector.hpp"
#include "detectors/whale_frontrun_detector.hpp"
#include "hotspine_extended_reader.hpp"
#include "utils/dynamic_logger.hpp"

// Include UI components from BTQ_Render_Engine
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include "ui/haptic_feedback.hpp"
#include "ui/tooltips.hpp"
#include "ui/tutorial.hpp"

using namespace BTQuant;
using namespace BTQuant::UI;

class BTQuantUIApp {
private:
    GLFWwindow* window_;
    bool show_demo_window_ = false;
    bool show_another_window_ = false;
    ImVec4 clear_color_ = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    
    // UI managers
    TutorialManager& tutorial_manager_;
    TooltipManager& tooltip_manager_;
    HapticFeedback& haptic_feedback_;

public:
    BTQuantUIApp() 
        : tutorial_manager_(get_global_tutorial_manager())
        , tooltip_manager_(get_global_tooltip_manager())
        , haptic_feedback_(HapticFeedback::getInstance()) {
        
        // Initialize haptic feedback system
        haptic_feedback_.initialize();
        
        std::cout << "BTQuant UI Application initialized" << std::endl;
    }

    bool initialize() {
        // Setup window
        glfwSetErrorCallback(glfw_error_callback);
        if (glfwInit() == 0) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }

        // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
        // GL ES 2.0 + GLSL 100
        const char* glsl_version = "#version 100";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#elif defined(__APPLE__)
        // GL 3.2 + GLSL 150
        const char* glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
        // GL 3.0 + GLSL 130
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

        // Create window with graphics context
        window_ = glfwCreateWindow(1280, 720, "BTQuant UI Demo", NULL, NULL);
        if (window_ == NULL) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            return false;
        }
        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1); // Enable vsync

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsLight();

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        return true;
    }

    void run() {
        std::cout << "Starting BTQuant UI Application..." << std::endl;
        
        // Show tutorial on first run
        show_tutorial_if_first_run();

        // Main loop
        while (!glfwWindowShouldClose(window_)) {
            // Poll and handle events (inputs, window resize, etc.)
            glfwPollEvents();

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()!)
            if (show_demo_window_)
                ImGui::ShowDemoWindow(&show_demo_window_);

            // 2. Show a simple window that we create ourselves.
            {
                static float f = 0.0f;
                static int counter = 0;

                ImGui::Begin("BTQuant Dashboard");                          // Create a window called "BTQuant Dashboard" and append into it.

                ImGui::Text("Welcome to BTQuant UI!");               // Display some text (you can use a format strings too)
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

                // Button with tooltip
                if (ImGui::Button("Test Haptic Feedback")) {
                    haptic_feedback_.trigger(HapticFeedback::FeedbackType::MediumClick);
                    counter++;
                }
                
                // Register and show tooltip for the button
                tooltip_manager_.register_tooltip("test_haptic_button", "Click to test haptic feedback functionality");
                tooltip_manager_.show_tooltip_for_last_item("test_haptic_button");

                ImGui::SliderFloat("float", &f, 0.0f, 1.0f, "%.3f");            // Edit 1 float using a slider from 0.0f to 1.0f
                ImGui::InputFloat("input_float", &f, 0.1f, 1.0f, "%.3f");       // Edit 1 float using an input box
                
                // Register and show tooltip for the slider
                tooltip_manager_.register_tooltip("float_slider", "Adjust the float value with this slider");
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
                    tooltip_manager_.show_tooltip("float_slider");
                }

                if (ImGui::Button("Button")) {                            // Buttons return true when clicked (most widgets return true when edited/activated)
                    counter++;
                    haptic_feedback_.trigger(HapticFeedback::FeedbackType::LightClick);
                }
                
                // Register and show tooltip for the button
                tooltip_manager_.register_tooltip("simple_button", "Simple button with haptic feedback on click");
                tooltip_manager_.show_tooltip_for_last_item("simple_button");

                ImGui::SameLine();
                ImGui::Text("counter = %d", counter);

                ImGui::Checkbox("Demo Window", &show_demo_window_);      // Edit bools storing our window open/close state
                ImGui::Checkbox("Another Window", &show_another_window_);

                // Button to start tutorial
                if (ImGui::Button("Start Tutorial")) {
                    start_tutorial();
                }
                
                // Register and show tooltip for the tutorial button
                tooltip_manager_.register_tooltip("start_tutorial_button", "Start the first-time user tutorial");
                tooltip_manager_.show_tooltip_for_last_item("start_tutorial_button");

                // Button to trigger haptic feedback for important interaction
                if (ImGui::Button("Important Action")) {
                    haptic_feedback_.triggerForImportantInteraction();
                }
                
                // Register and show tooltip for the important action button
                tooltip_manager_.register_tooltip("important_action_button", "Trigger haptic feedback for important interactions");
                tooltip_manager_.show_tooltip_for_last_item("important_action_button");

                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
                ImGui::End();
            }

            // 3. Show another simple window with more UI components
            if (show_another_window_) {
                ImGui::Begin("Another Window", &show_another_window_);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
                ImGui::Text("Hello from another window!");
                
                // More UI elements with tooltips
                if (ImGui::Button("Settings")) {
                    haptic_feedback_.trigger(HapticFeedback::FeedbackType::Selection);
                }
                
                tooltip_manager_.register_tooltip("settings_button", "Open application settings");
                tooltip_manager_.show_tooltip_for_last_item("settings_button");

                ImGui::SliderInt("Volume", &counter, 0, 100, "%d%%");
                
                tooltip_manager_.register_tooltip("volume_slider", "Adjust application volume");
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
                    tooltip_manager_.show_tooltip("volume_slider");
                }

                ImGui::End();
            }

            // Render tutorial if active
            render_tutorial();

            // Rendering
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window_, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(clear_color_.x * clear_color_.w, clear_color_.y * clear_color_.w, clear_color_.z * clear_color_.w, clear_color_.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window_);
        }

        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window_);
        glfwTerminate();
    }

    static void glfw_error_callback(int error, const char* description) {
        fprintf(stderr, "Glfw Error %d: %s\n", error, description);
    }
};

int main(int, char**) {
    std::cout << "========================================" << std::endl;
    std::cout << "  BTQuant UI Application" << std::endl;
    std::cout << "========================================" << std::endl;

    BTQuantUIApp app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize BTQuant UI Application" << std::endl;
        return 1;
    }

    app.run();

    std::cout << "BTQuant UI Application closed." << std::endl;
    return 0;
}