#include "../../../include/ui/tutorial.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <string>
#include <vector>

namespace BTQuant {
namespace UI {

// TutorialStep implementation
TutorialStep::TutorialStep(const std::string& title, 
                          const std::string& description, 
                          const std::string& target_control_id,
                          const glm::vec2& highlight_position,
                          const glm::vec2& highlight_size)
    : title(title)
    , description(description)
    , target_control_id(target_control_id)
    , highlight_position(highlight_position)
    , highlight_size(highlight_size) {}

// TutorialManager implementation
TutorialManager::TutorialManager() 
    : is_active_(false)
    , current_step_(0)
    , show_tutorial_on_startup_(true)
    , window_alpha_(0.95f) {
    initialize_tutorial_steps();
}

void TutorialManager::initialize_tutorial_steps() {
    // Welcome step
    steps_.emplace_back(
        "Welcome to QuantTower",
        "This guided tour will help you get started with the platform. "
        "Click 'Next' to continue learning about key features.",
        "",
        glm::vec2(0.0f, 0.0f),
        glm::vec2(0.0f, 0.0f)
    );

    // Dashboard Overview
    steps_.emplace_back(
        "Dashboard Overview",
        "This is your main trading dashboard. You can arrange multiple panels here "
        "to monitor markets, analyze charts, and execute trades. The top toolbar "
        "contains exchange selectors, symbol search, and panel creation options.",
        "dashboard_overview",
        glm::vec2(0.0f, 0.0f),
        glm::vec2(800.0f, 100.0f)
    );

    // Exchange Selection
    steps_.emplace_back(
        "Exchange Selection",
        "Select which cryptocurrency exchanges to include in your trading view. "
        "You can connect to multiple exchanges simultaneously for comprehensive market coverage.",
        "exchange_selector_button",
        glm::vec2(10.0f, 10.0f),
        glm::vec2(150.0f, 30.0f)
    );

    // Symbol Search
    steps_.emplace_back(
        "Symbol Search",
        "Search for trading symbols across all selected exchanges. "
        "You can quickly find and add symbols to your watchlist or charts.",
        "symbol_search_top",
        glm::vec2(170.0f, 10.0f),
        glm::vec2(200.0f, 30.0f)
    );

    // Panel Creation
    steps_.emplace_back(
        "Panel Creation",
        "Add different types of panels to your dashboard: Price charts, Footprint charts, "
        "Volume profiles, Order books, Time & Sales, and Watchlists. Each panel can be "
        "customized for your specific trading needs.",
        "add_chart_panel",
        glm::vec2(380.0f, 10.0f),
        glm::vec2(200.0f, 30.0f)
    );

    // Chart Features
    steps_.emplace_back(
        "Chart Features",
        "Charts include advanced technical analysis tools. You can change timeframes, "
        "add indicators (RSI, MACD, Bollinger Bands, etc.), adjust drawing tools, "
        "and use crosshairs for precise price/time readings.",
        "chart_timeframe_selector",
        glm::vec2(10.0f, 50.0f),
        glm::vec2(300.0f, 40.0f)
    );

    // Trading Controls
    steps_.emplace_back(
        "Trading Controls",
        "Execute trades directly from the interface. Place market or limit orders, "
        "set take-profit and stop-loss levels, and monitor your open positions. "
        "All trading controls are designed for speed and accuracy.",
        "place_buy_order",
        glm::vec2(10.0f, 100.0f),
        glm::vec2(200.0f, 60.0f)
    );

    // Indicators and Analysis
    steps_.emplace_back(
        "Indicators & Analysis",
        "Access a wide range of technical indicators and analytical tools. "
        "Customize indicator settings, create custom studies, and backtest strategies "
        "using historical data.",
        "indicator_selector",
        glm::vec2(320.0f, 100.0f),
        glm::vec2(250.0f, 60.0f)
    );

    // Alerts and Notifications
    steps_.emplace_back(
        "Alerts & Notifications",
        "Set up custom alerts for price movements, volume spikes, technical signals, "
        "and news events. Receive notifications via sound, visual cues, or email.",
        "alerts_menu",
        glm::vec2(10.0f, 170.0f),
        glm::vec2(150.0f, 40.0f)
    );

    // Settings and Customization
    steps_.emplace_back(
        "Settings & Customization",
        "Fully customize the interface to match your workflow. Adjust themes, "
        "fonts, colors, panel layouts, and performance settings to optimize "
        "your trading experience.",
        "settings_menu",
        glm::vec2(170.0f, 170.0f),
        glm::vec2(140.0f, 40.0f)
    );

    // Completion
    steps_.emplace_back(
        "Tutorial Complete!",
        "You've completed the guided tour. You can restart this tutorial anytime "
        "from the Help menu. Continue exploring the platform and customize it "
        "for your trading strategy. Happy trading!",
        "",
        glm::vec2(0.0f, 0.0f),
        glm::vec2(0.0f, 0.0f)
    );
}

void TutorialManager::start_tutorial() {
    is_active_ = true;
    current_step_ = 0;
    has_been_shown_ = true;
}

void TutorialManager::stop_tutorial() {
    is_active_ = false;
    current_step_ = 0;
}

void TutorialManager::next_step() {
    if (current_step_ < steps_.size() - 1) {
        current_step_++;
    } else {
        stop_tutorial();
    }
}

void TutorialManager::previous_step() {
    if (current_step_ > 0) {
        current_step_--;
    }
}

void TutorialManager::render_tutorial_window() {
    if (!is_active_) {
        return;
    }

    // Set up a modal-style window for the tutorial
    ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f - 250, 
                                   ImGui::GetIO().DisplaySize.y * 0.5f - 150), 
                           ImGuiCond_FirstUseEver);

    // Create a semi-transparent overlay effect
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, window_alpha_);

    // Create the tutorial window
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | 
                            ImGuiWindowFlags_NoResize |
                            ImGuiWindowFlags_NoMove |
                            ImGuiWindowFlags_NoScrollbar |
                            ImGuiWindowFlags_NoScrollWithMouse;

    if (ImGui::Begin("First-Time User Guide", nullptr, flags)) {
        if (!steps_.empty() && current_step_ < steps_.size()) {
            const auto& current = steps_[current_step_];

            // Title
            ImGui::TextColored(ImVec4(0.0f, 0.95f, 1.0f, 1.0f), "%s", current.title.c_str());
            ImGui::Separator();

            // Description
            ImGui::Spacing();
            ImGui::TextWrapped("%s", current.description.c_str());
            ImGui::Spacing();

            // Progress indicator
            ImGui::Text("Step %d of %zu", current_step_ + 1, steps_.size());
            ImGui::ProgressBar(static_cast<float>(current_step_ + 1) / static_cast<float>(steps_.size()), 
                              ImVec2(-1.0f, 0.0f), "");
            
            // Navigation buttons
            ImGui::Spacing();
            ImGui::BeginGroup();
            
            if (current_step_ > 0) {
                if (ImGui::Button("Previous")) {
                    previous_step();
                }
                ImGui::SameLine();
            }
            
            if (current_step_ < steps_.size() - 1) {
                if (ImGui::Button("Next")) {
                    next_step();
                }
            } else {
                if (ImGui::Button("Finish")) {
                    stop_tutorial();
                }
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Skip")) {
                stop_tutorial();
            }
            
            ImGui::EndGroup();
        }
    }
    ImGui::End();
    
    ImGui::PopStyleVar(); // Restore alpha
    
    // Draw highlight overlay if applicable
    draw_highlight_overlay();
}

void TutorialManager::draw_highlight_overlay() {
    if (!is_active_ || steps_.empty() || current_step_ >= steps_.size()) {
        return;
    }

    const auto& current = steps_[current_step_];
    
    // Only draw highlight if we have position and size info
    if (current.highlight_size.x > 0 && current.highlight_size.y > 0) {
        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        
        // Calculate screen position based on the highlight coordinates
        ImVec2 pos = ImVec2(current.highlight_position.x, current.highlight_position.y);
        ImVec2 size = ImVec2(current.highlight_size.x, current.highlight_size.y);
        ImVec2 min_pos = pos;
        ImVec2 max_pos = ImVec2(pos.x + size.x, pos.y + size.y);
        
        // Draw a semi-transparent overlay to dim the rest of the screen
        ImVec2 display_size = ImGui::GetIO().DisplaySize;
        draw_list->AddRectFilled(ImVec2(0, 0), display_size, 
                                IM_COL32(0, 0, 0, 180)); // Semi-transparent black
        
        // Clear the highlight area to make it stand out
        draw_list->AddRectFilled(min_pos, max_pos, 
                                IM_COL32(30, 30, 30, 220)); // Slightly transparent gray
        
        // Draw a bright border around the highlighted area
        draw_list->AddRect(min_pos, max_pos, 
                          IM_COL32(0, 150, 255, 255), 0.0f, 0, 3.0f); // Bright blue border
    }
}

bool TutorialManager::should_show_on_startup() const {
    return show_tutorial_on_startup_ && !has_been_shown_;
}

void TutorialManager::set_show_on_startup(bool show) {
    show_tutorial_on_startup_ = show;
}

void TutorialManager::update(float dt) {
    // Handle any tutorial-specific updates
    // For now, this is mainly for animation effects if needed in the future
}

void TutorialManager::set_has_been_shown(bool shown) {
    has_been_shown_ = shown;
}

bool TutorialManager::has_been_shown() const {
    return has_been_shown_;
}

// Global tutorial manager instance
static TutorialManager g_tutorial_manager;

TutorialManager& get_global_tutorial_manager() {
    return g_tutorial_manager;
}

void show_tutorial_if_first_run() {
    auto& tutorial_mgr = get_global_tutorial_manager();
    if (tutorial_mgr.should_show_on_startup()) {
        tutorial_mgr.start_tutorial();
        tutorial_mgr.set_has_been_shown(true);
    }
}

void start_tutorial() {
    get_global_tutorial_manager().start_tutorial();
}

void stop_tutorial() {
    get_global_tutorial_manager().stop_tutorial();
}

void render_tutorial() {
    get_global_tutorial_manager().render_tutorial_window();
}

} // namespace UI
} // namespace BTQuant