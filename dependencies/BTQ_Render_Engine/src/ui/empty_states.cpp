#include "../../include/ui/empty_states.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <sstream>

namespace btq {
namespace ui {

// Constructor
EmptyStateManager::EmptyStateManager() {
    // Initialize with default values if needed
}

// Destructor
EmptyStateManager::~EmptyStateManager() {
    // Clean up resources if needed
}

void EmptyStateManager::renderEmptyState(const EmptyStateConfig& config) {
    // Center the content in the current window
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8 * config.scale, 8 * config.scale));
    
    // Calculate the available space and center the content
    ImVec2 windowSize = ImGui::GetContentRegionAvail();
    ImGui::SetCursorPosX(windowSize.x * 0.5f - ImGui::CalcTextSize(config.title.c_str()).x * 0.5f);
    renderTitle(config.title, config.color);
    
    if (!config.icon.empty()) {
        ImGui::SetCursorPosX(windowSize.x * 0.5f - ImGui::CalcTextSize(config.icon.c_str()).x * 0.5f);
        renderIcon(config.icon, config.color, config.scale);
    }
    
    if (!config.message.empty()) {
        ImGui::SetCursorPosX(windowSize.x * 0.5f - ImGui::CalcTextSize(config.message.c_str()).x * 0.5f);
        renderMessage(config.message, config.color);
    }
    
    if (!config.suggestions.empty()) {
        ImGui::Spacing();
        renderSuggestions(config.suggestions, config.color);
    }
    
    ImGui::PopStyleVar();
}

void EmptyStateManager::renderSimpleEmptyState(const std::string& message, const std::string& title) {
    EmptyStateConfig config;
    config.title = title;
    config.message = message;
    config.icon = "🔍";
    config.color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
    config.scale = 1.0f;
    
    renderEmptyState(config);
}

void EmptyStateManager::renderTradingEmptyState(const std::string& context) {
    EmptyStateConfig config;
    config.icon = "💼";
    config.color = ImVec4(0.6f, 0.7f, 1.0f, 1.0f);
    config.scale = 1.2f;
    
    if (context == "chart") {
        config.title = "No Chart Data";
        config.message = "There is no chart data to display.";
        config.suggestions = {
            "Load market data for a symbol",
            "Check your data feed connection",
            "Select a different timeframe"
        };
    } else if (context == "orders") {
        config.title = "No Orders";
        config.message = "You don't have any active orders.";
        config.suggestions = {
            "Place a new order",
            "Check your order history",
            "Review your trading strategy"
        };
    } else if (context == "positions") {
        config.title = "No Positions";
        config.message = "You don't have any open positions.";
        config.suggestions = {
            "Open a new position",
            "Review your portfolio allocation",
            "Check your trading signals"
        };
    } else {
        config.title = "No " + context + " Data";
        config.message = "There is no " + context + " data to display.";
        config.suggestions = {
            "Connect to a data source",
            "Import historical data",
            "Configure your data settings"
        };
    }
    
    renderEmptyState(config);
}

void EmptyStateManager::renderAnalyticalEmptyState(const std::string& context) {
    EmptyStateConfig config;
    config.icon = "📊";
    config.color = ImVec4(0.4f, 0.8f, 0.4f, 1.0f);
    config.scale = 1.2f;
    
    if (context == "indicators") {
        config.title = "No Indicators";
        config.message = "No technical indicators are applied to this chart.";
        config.suggestions = {
            "Add a technical indicator",
            "Apply a moving average",
            "Configure RSI or MACD"
        };
    } else if (context == "analysis") {
        config.title = "No Analysis";
        config.message = "No analysis has been performed yet.";
        config.suggestions = {
            "Run a backtest",
            "Perform statistical analysis",
            "Generate a performance report"
        };
    } else if (context == "reports") {
        config.title = "No Reports";
        config.message = "No reports have been generated.";
        config.suggestions = {
            "Create a new report",
            "Export performance data",
            "Generate risk analysis"
        };
    } else {
        config.title = "No " + context + " Data";
        config.message = "There is no " + context + " data to analyze.";
        config.suggestions = {
            "Load analytical data",
            "Configure analysis parameters",
            "Import data for analysis"
        };
    }
    
    renderEmptyState(config);
}

void EmptyStateManager::renderDataFeedEmptyState(const std::string& context) {
    EmptyStateConfig config;
    config.icon = "📡";
    config.color = ImVec4(1.0f, 0.7f, 0.3f, 1.0f);
    config.scale = 1.2f;
    
    if (context == "market data") {
        config.title = "No Market Data";
        config.message = "Market data feed is not connected or unavailable.";
        config.suggestions = {
            "Check your internet connection",
            "Verify data feed settings",
            "Select a different data provider"
        };
    } else if (context == "news") {
        config.title = "No News";
        config.message = "No news articles are available for this symbol.";
        config.suggestions = {
            "Select a different symbol",
            "Check news feed settings",
            "Refresh the news feed"
        };
    } else if (context == "alerts") {
        config.title = "No Alerts";
        config.message = "You don't have any active alerts.";
        config.suggestions = {
            "Create a new alert",
            "Configure alert conditions",
            "Check alert settings"
        };
    } else {
        config.title = "No " + context + " Feed";
        config.message = "The " + context + " feed is not available.";
        config.suggestions = {
            "Check connection settings",
            "Verify subscription status",
            "Contact support for assistance"
        };
    }
    
    renderEmptyState(config);
}

void EmptyStateManager::renderActionableEmptyState(
    const EmptyStateConfig& config,
    const std::vector<std::string>& button_labels,
    const std::vector<std::function<void()>>& button_callbacks) {
    
    // Render the empty state content
    renderEmptyState(config);
    
    // Add action buttons below the content
    if (!button_labels.empty() && !button_callbacks.empty()) {
        ImGui::Spacing();
        ImGui::Spacing();
        
        float totalButtonWidth = 0.0f;
        for (const auto& label : button_labels) {
            totalButtonWidth += ImGui::CalcTextSize(label.c_str()).x + ImGui::GetStyle().FramePadding.x * 2.0f + ImGui::GetStyle().ItemSpacing.x;
        }
        
        // Adjust for the last item spacing
        totalButtonWidth -= ImGui::GetStyle().ItemSpacing.x;
        
        ImVec2 windowSize = ImGui::GetContentRegionAvail();
        float startX = windowSize.x * 0.5f - totalButtonWidth * 0.5f;
        ImGui::SetCursorPosX(startX);
        
        for (size_t i = 0; i < button_labels.size() && i < button_callbacks.size(); ++i) {
            if (i > 0) {
                ImGui::SameLine();
            }
            
            if (ImGui::Button(button_labels[i].c_str())) {
                if (button_callbacks[i]) {
                    button_callbacks[i]();
                }
            }
        }
    }
}

void EmptyStateManager::renderIcon(const std::string& icon, const ImVec4& color, float scale) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::PushFont(nullptr); // Use default font
    ImGui::Text("%s", icon.c_str());
    ImGui::PopStyleColor();
}

void EmptyStateManager::renderTitle(const std::string& title, const ImVec4& color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::TextColored(color, "%s", title.c_str());
    ImGui::PopStyleColor();
}

void EmptyStateManager::renderMessage(const std::string& message, const ImVec4& color) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(color.x, color.y, color.z, color.w * 0.8f)); // Slightly dimmer than title
    ImGui::TextWrapped("%s", message.c_str());
    ImGui::PopStyleColor();
}

void EmptyStateManager::renderSuggestions(const std::vector<std::string>& suggestions, const ImVec4& color) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(color.x, color.y, color.z, color.w * 0.7f)); // Even dimmer for suggestions
    
    for (const auto& suggestion : suggestions) {
        // Add bullet point to each suggestion
        ImGui::Bullet();
        ImGui::SameLine();
        ImGui::TextWrapped("%s", suggestion.c_str());
    }
    
    ImGui::PopStyleColor();
}

} // namespace ui
} // namespace btq