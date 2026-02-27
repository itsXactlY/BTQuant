#include "setup_wizard.hpp"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <iostream>

// Example integration of the setup wizard into the main application
class Application {
private:
    GLFWwindow* window;
    SetupWizard::FirstRunSetup setupWizard;
    bool showMainWindow = false;

public:
    Application() {
        // Initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return;
        }

        // Create window
        window = glfwCreateWindow(1280, 720, "PubBTQuant Trading Terminal", NULL, NULL);
        if (!window) {
            std::cerr << "Failed to create window" << std::endl;
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync
    }

    void run() {
        // Check if first run setup is needed
        bool needsSetup = !setupWizard.isSetupComplete();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Show setup wizard if needed
            if (needsSetup) {
                setupWizard.render();
                
                // If setup is complete, show main window
                if (setupWizard.isSetupComplete()) {
                    needsSetup = false;
                    showMainWindow = true;
                }
            } else {
                // Show main application window
                if (showMainWindow) {
                    showMainWindowContent();
                }
            }

            // Rendering
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }
    }

private:
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    void showMainWindowContent() {
        ImGui::Begin("PubBTQuant Trading Terminal");

        // Main application content would go here
        ImGui::Text("Welcome to PubBTQuant!");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        // Example trading interface elements
        if (ImGui::CollapsingHeader("Market Data")) {
            ImGui::Text("Symbol: BTC/USD");
            ImGui::Text("Price: $45,230.50");
            ImGui::Text("24h Change: +2.3%");
        }

        if (ImGui::CollapsingHeader("Trading Panel")) {
            ImGui::Text("Place your trades here");
            // Trading controls would go here
        }

        if (ImGui::CollapsingHeader("Charts")) {
            ImGui::Text("Interactive charts would appear here");
            // Chart rendering would go here
        }

        ImGui::End();
    }
};