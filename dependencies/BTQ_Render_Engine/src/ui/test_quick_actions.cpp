#include "ui/quick_actions.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <iostream>

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a simple window to test the quick actions toolbar
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Quick Actions Toolbar Test", nullptr, nullptr);

    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create and use the quick actions toolbar
        static BTQuant::QuickActionsToolbar toolbar;
        
        // Add a custom action dynamically
        static bool added_action = false;
        if (!added_action) {
            toolbar.add_action(BTQuant::QuickAction("Custom", "⭐", []() {
                std::cout << "Custom action triggered!" << std::endl;
            }, "A custom quick action"));
            added_action = true;
        }

        // Render the toolbar
        toolbar.render();

        // Simple demo window to show the toolbar in context
        ImGui::Begin("Demo Window");
        ImGui::Text("Quick Actions Toolbar Demo");
        ImGui::Text("Check the floating toolbar at the top-left corner.");
        ImGui::Text("You can drag it around and click the action buttons.");
        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        
        // Clear screen here in a real implementation
        
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}