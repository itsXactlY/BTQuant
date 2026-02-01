#include "../../include/components/multi_vwap_panel.hpp"

#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <ctime>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

MultiVWAPPanel::MultiVWAPPanel(const PanelConfig& config,
                               std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), processor_(processor) {
    // Initialize with a daily VWAP by default
    add_daily_vwap();
}

void MultiVWAPPanel::update(float dt) {
    update_timer_ += dt;
    if (update_timer_ >= UPDATE_INTERVAL) {
        update_vwaps();
        update_timer_ = 0.0f;
    }
}

void MultiVWAPPanel::render() {
    begin_panel_window();

    if (!is_visible()) {
        end_panel_window();
        return;
    }

    render_vwap_controls();
    ImGui::Separator();
    render_add_vwap_section();
    ImGui::Separator();
    render_vwap_list();

    end_panel_window();
}

void MultiVWAPPanel::update_vwaps() {
    // Update VWAP calculations if market data is available
    if (processor_) {
        // For each VWAP instance, recalculate based on current market data
        for (auto& instance : vwap_instances_) {
            // Get the chart data to recalculate the VWAP
            if (processor_->hasData()) {
                auto chart_data = processor_->getChartData();

                // Convert chart data to OHLCVCandle format for VWAP calculation
                std::vector<BTQuant::RenderEngine::OHLCVCandle> bars;
                for (const auto& candle : chart_data) {
                    BTQuant::RenderEngine::OHLCVCandle bar;
                    bar.timestamp = candle.timestamp;
                    bar.open = candle.open;
                    bar.high = candle.high;
                    bar.low = candle.low;
                    bar.close = candle.close;
                    bar.volume = candle.volume;
                    bars.push_back(bar);
                }

                // Recalculate the VWAP from the anchor point
                instance.vwap.calculate(bars);
            }
        }
    }
}

void MultiVWAPPanel::render_vwap_controls() {
    ImGui::Text("VWAP Controls");
    ImGui::Checkbox("Show Daily VWAP", &show_daily_vwap_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Weekly VWAP", &show_weekly_vwap_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Monthly VWAP", &show_monthly_vwap_);
}

void MultiVWAPPanel::render_add_vwap_section() {
    ImGui::Text("Add New VWAP Instance");

    ImGui::InputText("Name", &new_vwap_name_, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::ColorEdit4("Color", &new_vwap_color_.x);

    if (ImGui::Button("Add Daily VWAP")) {
        add_daily_vwap();
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Weekly VWAP")) {
        add_weekly_vwap();
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Monthly VWAP")) {
        add_monthly_vwap();
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Custom VWAP")) {
        add_custom_vwap();
    }
}

void MultiVWAPPanel::render_vwap_list() {
    ImGui::Text("Active VWAP Instances (%zu)", vwap_instances_.size());

    if (vwap_instances_.empty()) {
        ImGui::Text("No VWAP instances active. Add one using the controls above.");
        return;
    }

    ImGui::BeginTable("VWAPTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Visible", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    ImGui::TableSetupColumn("Color", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableHeadersRow();

    for (auto it = vwap_instances_.begin(); it != vwap_instances_.end();) {
        auto& instance = *it;

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", instance.name.c_str());

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%s", instance.type.c_str());

        ImGui::TableSetColumnIndex(2);
        bool visible = instance.visible;
        if (ImGui::Checkbox(("##visible" + instance.id).c_str(), &visible)) {
            instance.visible = visible;
        }

        ImGui::TableSetColumnIndex(3);
        // Show color button with right-click context menu for color picker
        ImGui::ColorButton(("##color" + instance.id).c_str(), instance.color,
                          ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoTooltip, ImVec2(20, 20));

        // Right-click context menu for color picker
        std::string picker_popup_id = "Color Picker##" + instance.id;
        if (ImGui::BeginPopupContextItem(picker_popup_id.c_str())) {
            ImGui::Text("Pick color for %s", instance.name.c_str());
            ImGui::Separator();
            if (ImGui::ColorPicker3(("##picker" + instance.id).c_str(), &instance.color.x,
                                   ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoSidePreview |
                                   ImGuiColorEditFlags_NoSmallPreview)) {
                // Color was changed
            }
            ImGui::EndPopup();
        }

        ImGui::TableSetColumnIndex(4);
        if (ImGui::Button(("Delete##" + instance.id).c_str())) {
            it = vwap_instances_.erase(it);
            continue; // Skip increment since erase returns next iterator
        } else {
            ++it;
        }
    }

    ImGui::EndTable();
}

void MultiVWAPPanel::add_daily_vwap() {
    // Calculate start of today in milliseconds (UTC)
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto local_time = std::gmtime(&time_t_now); // Use gmtime for UTC

    // Set hour, minute, second to 0 for start of day
    local_time->tm_hour = 0;
    local_time->tm_min = 0;
    local_time->tm_sec = 0;

    auto today_start_time_t = std::mktime(local_time);
    auto today_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::from_time_t(today_start_time_t)).count();

    std::string id = generate_unique_id();
    VWAPInstance instance(id, "Daily VWAP", today_start_ms,
                         ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "daily"); // Yellow

    vwap_instances_.push_back(instance);
}

void MultiVWAPPanel::add_weekly_vwap() {
    // Calculate start of week (Monday) in milliseconds (UTC)
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto local_time = std::gmtime(&time_t_now); // Use gmtime for UTC

    // Calculate days since Monday (0=Sun, 1=Mon, ..., 6=Sat)
    // tm_wday: 0=Sunday, 1=Monday, ..., 6=Saturday
    int days_since_monday = (local_time->tm_wday + 6) % 7; // +6 to make Monday=0

    // Subtract days to get to Monday
    auto monday_time = std::chrono::system_clock::from_time_t(time_t_now) -
                       std::chrono::hours(24 * days_since_monday);

    // Get time_t for Monday at 00:00:00
    auto monday_time_t = std::chrono::system_clock::to_time_t(monday_time);
    auto monday_local_time = std::gmtime(&monday_time_t);

    // Set hour, minute, second to 0
    monday_local_time->tm_hour = 0;
    monday_local_time->tm_min = 0;
    monday_local_time->tm_sec = 0;

    auto week_start_time_t = std::mktime(monday_local_time);
    auto week_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::from_time_t(week_start_time_t)).count();

    std::string id = generate_unique_id();
    VWAPInstance instance(id, "Weekly VWAP", week_start_ms,
                         ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "weekly"); // Cyan

    vwap_instances_.push_back(instance);
}

void MultiVWAPPanel::add_monthly_vwap() {
    // Calculate start of month in milliseconds (UTC)
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto local_time = std::gmtime(&time_t_now); // Use gmtime for UTC

    // Set day to 1, hour, minute, second to 0
    local_time->tm_mday = 1;
    local_time->tm_hour = 0;
    local_time->tm_min = 0;
    local_time->tm_sec = 0;

    auto month_start_time_t = std::mktime(local_time);
    auto month_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::from_time_t(month_start_time_t)).count();

    std::string id = generate_unique_id();
    VWAPInstance instance(id, "Monthly VWAP", month_start_ms,
                         ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "monthly"); // Orange

    vwap_instances_.push_back(instance);
}

void MultiVWAPPanel::add_custom_vwap() {
    std::string id = generate_unique_id();
    VWAPInstance instance(id, new_vwap_name_, new_vwap_anchor_time_,
                         new_vwap_color_, "custom");

    vwap_instances_.push_back(instance);
}

void MultiVWAPPanel::remove_vwap(const std::string& id) {
    vwap_instances_.erase(
        std::remove_if(vwap_instances_.begin(), vwap_instances_.end(),
            [&id](const VWAPInstance& instance) {
                return instance.id == id;
            }),
        vwap_instances_.end()
    );
}

std::string MultiVWAPPanel::generate_unique_id() {
    auto now = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    return "vwap_" + std::to_string(ms);
}

uint64_t MultiVWAPPanel::get_start_of_day(uint64_t timestamp) {
    // Convert timestamp from milliseconds to time_t
    auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(
        std::chrono::milliseconds(timestamp));

    auto time_t_val = std::chrono::system_clock::to_time_t(tp);
    auto local_time = std::gmtime(&time_t_val); // Use gmtime for UTC

    // Set hour, minute, second to 0
    local_time->tm_hour = 0;
    local_time->tm_min = 0;
    local_time->tm_sec = 0;

    auto start_of_day = std::mktime(local_time);
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::from_time_t(start_of_day)).count();
}

uint64_t MultiVWAPPanel::get_start_of_week(uint64_t timestamp) {
    // Convert timestamp from milliseconds to time_t
    auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(
        std::chrono::milliseconds(timestamp));

    auto time_t_val = std::chrono::system_clock::to_time_t(tp);
    auto local_time = std::gmtime(&time_t_val); // Use gmtime for UTC

    // Calculate days since Monday
    int days_since_monday = (local_time->tm_wday + 6) % 7; // Adjust so Monday=0

    // Subtract days to get to Monday
    auto monday_tp = tp - std::chrono::hours(24 * days_since_monday);

    // Get time_t for Monday at 00:00:00
    auto monday_time_t = std::chrono::system_clock::to_time_t(monday_tp);
    auto monday_local_time = std::gmtime(&monday_time_t);

    // Set hour, minute, second to 0
    monday_local_time->tm_hour = 0;
    monday_local_time->tm_min = 0;
    monday_local_time->tm_sec = 0;

    auto start_of_week = std::mktime(monday_local_time);
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::from_time_t(start_of_week)).count();
}

uint64_t MultiVWAPPanel::get_start_of_month(uint64_t timestamp) {
    // Convert timestamp from milliseconds to time_t
    auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(
        std::chrono::milliseconds(timestamp));

    auto time_t_val = std::chrono::system_clock::to_time_t(tp);
    auto local_time = std::gmtime(&time_t_val); // Use gmtime for UTC

    // Set day to 1, hour, minute, second to 0
    local_time->tm_mday = 1;
    local_time->tm_hour = 0;
    local_time->tm_min = 0;
    local_time->tm_sec = 0;

    auto start_of_month = std::mktime(local_time);
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::from_time_t(start_of_month)).count();
}

}  // namespace BTQuant