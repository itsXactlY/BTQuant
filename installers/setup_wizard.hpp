#pragma once
#include <imgui.h>
#include <imgui_internal.h>
#include <string>
#include <vector>

namespace SetupWizard {

enum class SetupStep {
    Welcome,
    License,
    DataDirectory,
    ExchangeSetup,
    ThemeSelection,
    Completion
};

class FirstRunSetup {
private:
    SetupStep currentStep = SetupStep::Welcome;
    bool isCompleted = false;
    
    // Setup data
    std::string dataDirectory;
    std::string selectedExchange;
    std::string selectedTheme = "Dark";
    bool acceptedLicense = false;
    bool autoStart = false;
    
    // Available options
    std::vector<std::string> exchanges = {"Binance", "Coinbase", "Kraken", "Bybit", "OKX"};
    std::vector<std::string> themes = {"Dark", "Light", "Blue", "Classic"};

public:
    FirstRunSetup();
    ~FirstRunSetup() = default;
    
    void render();
    bool isSetupComplete() const { return isCompleted; }
    
private:
    void renderWelcomeStep();
    void renderLicenseStep();
    void renderDataDirectoryStep();
    void renderExchangeSetupStep();
    void renderThemeSelectionStep();
    void renderCompletionStep();
    
    void drawNavigationButtons();
    void nextStep();
    void previousStep();
    void finishSetup();
    
    // Helper methods
    void saveConfiguration() const;
    void loadConfiguration();
};

// Implementation
FirstRunSetup::FirstRunSetup() {
    // Initialize with default values
    char* homeDir = getenv("HOME");
    if (!homeDir) homeDir = getenv("USERPROFILE");  // Windows fallback
    if (homeDir) {
        dataDirectory = std::string(homeDir) + "/.pubbtquant/data";
    } else {
        dataDirectory = "./pubbtquant_data";
    }
    
    // Check if setup was already completed
    loadConfiguration();
    if (acceptedLicense) {  // If license was previously accepted, setup is complete
        isCompleted = true;
    }
}

void FirstRunSetup::render() {
    if (isCompleted) return;
    
    // Create a modal window for the setup wizard
    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    
    ImGui::Begin("PubBTQuant - First Run Setup", nullptr, 
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);
    
    // Draw header
    ImGui::TextColored(ImVec4(0.2f, 0.7f, 1.0f, 1.0f), "Welcome to PubBTQuant Trading Terminal");
    ImGui::Separator();
    
    // Draw step indicator
    ImGui::Spacing();
    const char* stepNames[] = {"Welcome", "License", "Data Directory", "Exchange Setup", "Theme Selection", "Completion"};
    int currentStepInt = static_cast<int>(currentStep);
    
    for (int i = 0; i < 6; ++i) {
        if (i > 0) ImGui::SameLine();
        
        ImVec4 color = (i <= currentStepInt) ? 
                      ImVec4(0.2f, 0.7f, 1.0f, 1.0f) : 
                      ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
        
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::Text("%s", stepNames[i]);
        ImGui::PopStyleColor();
        
        if (i < 5) {
            ImGui::SameLine();
            ImGui::Text(" > ");
        }
    }
    
    ImGui::Separator();
    
    // Render current step
    switch (currentStep) {
        case SetupStep::Welcome:
            renderWelcomeStep();
            break;
        case SetupStep::License:
            renderLicenseStep();
            break;
        case SetupStep::DataDirectory:
            renderDataDirectoryStep();
            break;
        case SetupStep::ExchangeSetup:
            renderExchangeSetupStep();
            break;
        case SetupStep::ThemeSelection:
            renderThemeSelectionStep();
            break;
        case SetupStep::Completion:
            renderCompletionStep();
            break;
    }
    
    ImGui::End();
}

void FirstRunSetup::renderWelcomeStep() {
    ImGui::TextWrapped("Thank you for choosing PubBTQuant, an advanced trading terminal with real-time analytics.");
    ImGui::TextWrapped("This setup wizard will help you configure your trading environment.");
    ImGui::Spacing();
    
    ImGui::BulletText("Connect to multiple exchanges");
    ImGui::BulletText("Real-time market data visualization");
    ImGui::BulletText("Advanced charting and analysis tools");
    ImGui::BulletText("Customizable layouts and themes");
    ImGui::BulletText("Risk management and alert systems");
    
    ImGui::Spacing();
    ImGui::Text("Click 'Continue' to proceed with the setup.");
    
    drawNavigationButtons();
}

void FirstRunSetup::renderLicenseStep() {
    ImGui::Text("License Agreement");
    ImGui::Separator();
    
    ImGui::BeginChild("LicenseText", ImVec2(0, 300), true);
    ImGui::TextWrapped(R"(MIT License

Copyright (c) 2023-2026 PubBTQuant Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.)");
    ImGui::EndChild();
    
    ImGui::Spacing();
    ImGui::Checkbox("I accept the terms of the license agreement", &acceptedLicense);
    
    drawNavigationButtons();
}

void FirstRunSetup::renderDataDirectoryStep() {
    ImGui::Text("Data Storage Location");
    ImGui::Separator();
    
    ImGui::Text("Choose where to store your trading data, layouts, and settings:");
    ImGui::InputText("Data Directory", &dataDirectory);
    
    if (ImGui::Button("Browse...")) {
        // In a real implementation, this would open a file dialog
        // For now, we'll just show a message
        ImGui::OpenPopup("BrowseDataDir");
    }
    
    if (ImGui::BeginPopupModal("BrowseDataDir", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("In a real implementation, this would open a file browser.");
        ImGui::Text("For now, please enter the path manually.");
        
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    ImGui::Spacing();
    ImGui::Text("Note: This directory will store your market data, layouts, and personal settings.");
    
    drawNavigationButtons();
}

void FirstRunSetup::renderExchangeSetupStep() {
    ImGui::Text("Exchange Configuration");
    ImGui::Separator();
    
    ImGui::Text("Select your preferred exchange(s) to connect:");
    
    static int selectedExchangeIdx = 0;
    if (ImGui::BeginCombo("##ExchangeCombo", exchanges[selectedExchangeIdx].c_str())) {
        for (int n = 0; n < exchanges.size(); n++) {
            const bool isSelected = (selectedExchangeIdx == n);
            if (ImGui::Selectable(exchanges[n].c_str(), isSelected)) {
                selectedExchangeIdx = n;
                selectedExchange = exchanges[n];
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    
    ImGui::Spacing();
    ImGui::Text("Enter your API credentials (optional for demo mode):");
    
    static std::string apiKey = "";
    static std::string apiSecret = "";
    
    ImGui::InputText("API Key", &apiKey);
    ImGui::InputText("API Secret", &apiSecret, ImGuiInputTextFlags_Password);
    
    ImGui::Spacing();
    ImGui::Text("Note: API credentials are stored securely on your local machine.");
    
    drawNavigationButtons();
}

void FirstRunSetup::renderThemeSelectionStep() {
    ImGui::Text("Appearance Settings");
    ImGui::Separator();
    
    ImGui::Text("Choose your preferred theme:");
    
    static int selectedThemeIdx = 0;
    if (ImGui::BeginCombo("##ThemeCombo", themes[selectedThemeIdx].c_str())) {
        for (int n = 0; n < themes.size(); n++) {
            const bool isSelected = (selectedThemeIdx == n);
            if (ImGui::Selectable(themes[n].c_str(), isSelected)) {
                selectedThemeIdx = n;
                selectedTheme = themes[n];
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    
    ImGui::Spacing();
    ImGui::Checkbox("Start PubBTQuant automatically at login", &autoStart);
    
    // Preview of theme (simplified)
    ImGui::Spacing();
    ImGui::Text("Theme Preview:");
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 1.0f, 1.0f));
    ImGui::Button("Sample Button");
    ImGui::PopStyleColor(2);
    
    drawNavigationButtons();
}

void FirstRunSetup::renderCompletionStep() {
    ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "Setup Complete!");
    ImGui::Separator();
    
    ImGui::TextWrapped("Your PubBTQuant trading terminal is now configured and ready to use.");
    ImGui::Spacing();
    
    ImGui::BulletText("Data directory: %s", dataDirectory.c_str());
    ImGui::BulletText("Selected exchange: %s", selectedExchange.empty() ? "Demo Mode" : selectedExchange.c_str());
    ImGui::BulletText("Selected theme: %s", selectedTheme.c_str());
    
    ImGui::Spacing();
    ImGui::TextWrapped("Click 'Finish' to start using PubBTQuant. You can change these settings later in the application preferences.");
    
    // Navigation buttons for completion step
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    if (ImGui::Button("Finish", ImVec2(120, 0))) {
        finishSetup();
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Back", ImVec2(120, 0))) {
        previousStep();
    }
}

void FirstRunSetup::drawNavigationButtons() {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    bool canGoBack = currentStep != SetupStep::Welcome;
    bool canGoNext = true;
    
    // Special validation for license step
    if (currentStep == SetupStep::License && !acceptedLicense) {
        canGoNext = false;
    }
    
    if (canGoBack && ImGui::Button("Back", ImVec2(120, 0))) {
        previousStep();
    }
    
    ImGui::SameLine();
    
    if (canGoNext && ImGui::Button((currentStep == SetupStep::Completion) ? "Finish" : "Continue", ImVec2(120, 0))) {
        if (currentStep == SetupStep::Completion) {
            finishSetup();
        } else {
            nextStep();
        }
    }
    
    if (!canGoNext && currentStep == SetupStep::License) {
        ImGui::SameLine();
        ImGui::TextDisabled("(Accept the license to continue)");
    }
}

void FirstRunSetup::nextStep() {
    int stepValue = static_cast<int>(currentStep);
    if (stepValue < 5) {  // 5 is the max step index
        currentStep = static_cast<SetupStep>(stepValue + 1);
    }
}

void FirstRunSetup::previousStep() {
    int stepValue = static_cast<int>(currentStep);
    if (stepValue > 0) {
        currentStep = static_cast<SetupStep>(stepValue - 1);
    }
}

void FirstRunSetup::finishSetup() {
    saveConfiguration();
    isCompleted = true;
}

void FirstRunSetup::saveConfiguration() const {
    // In a real implementation, this would save to a configuration file
    // For now, we'll just print to console
    printf("Saving configuration:\n");
    printf("  Data Directory: %s\n", dataDirectory.c_str());
    printf("  Selected Exchange: %s\n", selectedExchange.c_str());
    printf("  Selected Theme: %s\n", selectedTheme.c_str());
    printf("  Auto-start: %s\n", autoStart ? "Yes" : "No");
    printf("  License Accepted: %s\n", acceptedLicense ? "Yes" : "No");
    
    // Create the data directory if it doesn't exist
    // In a real implementation, we would use cross-platform file operations
}

void FirstRunSetup::loadConfiguration() {
    // In a real implementation, this would load from a configuration file
    // For now, we'll just set defaults
    acceptedLicense = false;  // If this remains false, setup hasn't been completed
}

} // namespace SetupWizard