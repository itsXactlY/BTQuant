#include "../include/components/chart_panel_settings.hpp"

#include "imgui.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>

// Include the chart panel header for the implementation
#include "../include/components/chart_panel.hpp"

namespace BTQuant {

ChartPanelSettings::ChartPanelSettings(void* chart_panel_ptr)
    : BasePanelSettings("Chart Panel Settings"), chart_panel_ptr_(chart_panel_ptr) {
    reset_temp_config();

    // Load settings when the settings object is created
    load_settings();
}

ChartPanel* ChartPanelSettings::get_chart_panel() const {
    return static_cast<ChartPanel*>(chart_panel_ptr_);
}

void ChartPanelSettings::render() {
    if (!is_modal_open_) return;

    // Create modal window
    ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);
    if (ImGui::Begin((title_ + "###chart_settings_modal").c_str(), &is_modal_open_,
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {

        ImGui::Text("Chart Panel Settings");
        ImGui::Separator();

        // SMA Settings
        ImGui::Text("Simple Moving Averages");
        ImGui::Indent();
        ImGui::Checkbox("Show SMA 9", &temp_config_.show_sma_9);
        ImGui::SameLine();
        ImGui::Checkbox("Show SMA 20", &temp_config_.show_sma_20);
        ImGui::SameLine();
        ImGui::Checkbox("Show SMA 50", &temp_config_.show_sma_50);
        ImGui::SameLine();
        ImGui::Checkbox("Show SMA 200", &temp_config_.show_sma_200);
        ImGui::Unindent();

        // EMA Settings
        ImGui::Text("Exponential Moving Averages");
        ImGui::Indent();
        ImGui::Checkbox("Show EMA 9", &temp_config_.show_ema_9);
        ImGui::SameLine();
        ImGui::Checkbox("Show EMA 21", &temp_config_.show_ema_21);
        ImGui::SameLine();
        ImGui::Checkbox("Show EMA 50", &temp_config_.show_ema_50);
        ImGui::SameLine();
        ImGui::Checkbox("Show EMA 200", &temp_config_.show_ema_200);
        ImGui::Unindent();

        // Oscillator Settings
        ImGui::Text("Oscillators");
        ImGui::Indent();
        ImGui::Checkbox("Show RSI", &temp_config_.show_rsi);
        ImGui::SameLine();
        ImGui::Checkbox("Show MACD", &temp_config_.show_macd);
        ImGui::SameLine();
        ImGui::Checkbox("Show Stochastic", &temp_config_.show_stochastic);
        ImGui::SameLine();
        ImGui::Checkbox("Show ATR", &temp_config_.show_atr);
        ImGui::Unindent();

        // Overlay Settings
        ImGui::Text("Overlays");
        ImGui::Indent();
        ImGui::Checkbox("Show Bollinger Bands", &temp_config_.show_bollinger);
        ImGui::SameLine();
        ImGui::Checkbox("Show Fibonacci", &temp_config_.show_fibonacci);
        ImGui::SameLine();
        ImGui::Checkbox("Show Volume Profile", &temp_config_.show_volume_profile);
        ImGui::SameLine();
        ImGui::Checkbox("Show Crosshair Info", &temp_config_.show_crosshair_info);
        ImGui::Unindent();

        // RSI Configuration
        if (temp_config_.show_rsi) {
            ImGui::Separator();
            ImGui::Text("RSI Configuration");
            ImGui::SliderInt("Period", &temp_config_.rsi_period, 1, 50);
            ImGui::SliderDouble("Overbought", &temp_config_.rsi_overbought, 50.0, 100.0);
            ImGui::SliderDouble("Oversold", &temp_config_.rsi_oversold, 0.0, 50.0);
        }

        // Bollinger Bands Configuration
        if (temp_config_.show_bollinger) {
            ImGui::Separator();
            ImGui::Text("Bollinger Bands Configuration");
            ImGui::SliderInt("Period", &temp_config_.bollinger_period, 1, 50);
            ImGui::SliderDouble("Std Dev", &temp_config_.bollinger_std_dev, 0.5, 5.0);
        }

        // MACD Configuration
        if (temp_config_.show_macd) {
            ImGui::Separator();
            ImGui::Text("MACD Configuration");
            ImGui::SliderInt("Fast Period", &temp_config_.macd_fast_period, 1, 50);
            ImGui::SliderInt("Slow Period", &temp_config_.macd_slow_period, 1, 50);
            ImGui::SliderInt("Signal Period", &temp_config_.macd_signal_period, 1, 20);
        }

        // Stochastic Configuration
        if (temp_config_.show_stochastic) {
            ImGui::Separator();
            ImGui::Text("Stochastic Configuration");
            ImGui::SliderInt("K Period", &temp_config_.stochastic_k_period, 1, 50);
            ImGui::SliderInt("D Period", &temp_config_.stochastic_d_period, 1, 50);
            ImGui::SliderInt("Slow Period", &temp_config_.stochastic_slow_period, 1, 20);
        }

        // ATR Configuration
        if (temp_config_.show_atr) {
            ImGui::Separator();
            ImGui::Text("ATR Configuration");
            ImGui::SliderInt("Period", &temp_config_.atr_period, 1, 50);
        }

        // Buttons
        ImGui::Separator();
        if (ImGui::Button("Apply")) {
            apply_settings();
            save_settings();  // Save settings when applied
            is_modal_open_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            is_modal_open_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            reset_temp_config();
        }
    }
    ImGui::End();
}

void ChartPanelSettings::apply_settings() {
    if (chart_panel_ptr_) {
        ChartPanel* chart_panel = get_chart_panel();
        if (chart_panel) {
            // Update the chart panel's indicator config with temporary settings
            // We need to copy the fields from our temp config to the real indicator config
            chart_panel->indicator_config_.show_sma_9 = temp_config_.show_sma_9;
            chart_panel->indicator_config_.show_sma_10 = temp_config_.show_sma_10;
            chart_panel->indicator_config_.show_sma_20 = temp_config_.show_sma_20;
            chart_panel->indicator_config_.show_sma_50 = temp_config_.show_sma_50;
            chart_panel->indicator_config_.show_sma_200 = temp_config_.show_sma_200;
            chart_panel->indicator_config_.show_ema_9 = temp_config_.show_ema_9;
            chart_panel->indicator_config_.show_ema_10 = temp_config_.show_ema_10;
            chart_panel->indicator_config_.show_ema_20 = temp_config_.show_ema_20;
            chart_panel->indicator_config_.show_ema_21 = temp_config_.show_ema_21;
            chart_panel->indicator_config_.show_ema_50 = temp_config_.show_ema_50;
            chart_panel->indicator_config_.show_ema_200 = temp_config_.show_ema_200;
            chart_panel->indicator_config_.show_rsi = temp_config_.show_rsi;
            chart_panel->indicator_config_.show_macd = temp_config_.show_macd;
            chart_panel->indicator_config_.show_bollinger = temp_config_.show_bollinger;
            chart_panel->indicator_config_.show_stochastic = temp_config_.show_stochastic;
            chart_panel->indicator_config_.show_atr = temp_config_.show_atr;
            chart_panel->indicator_config_.show_volume_profile = temp_config_.show_volume_profile;
            chart_panel->indicator_config_.show_fibonacci = temp_config_.show_fibonacci;
            chart_panel->indicator_config_.show_crosshair_info = temp_config_.show_crosshair_info;
            chart_panel->indicator_config_.fib_start_price = temp_config_.fib_start_price;
            chart_panel->indicator_config_.fib_end_price = temp_config_.fib_end_price;
            chart_panel->indicator_config_.rsi_period = temp_config_.rsi_period;
            chart_panel->indicator_config_.rsi_overbought = temp_config_.rsi_overbought;
            chart_panel->indicator_config_.rsi_oversold = temp_config_.rsi_oversold;
            chart_panel->indicator_config_.bollinger_period = temp_config_.bollinger_period;
            chart_panel->indicator_config_.bollinger_std_dev = temp_config_.bollinger_std_dev;
            chart_panel->indicator_config_.macd_fast_period = temp_config_.macd_fast_period;
            chart_panel->indicator_config_.macd_slow_period = temp_config_.macd_slow_period;
            chart_panel->indicator_config_.macd_signal_period = temp_config_.macd_signal_period;
            chart_panel->indicator_config_.stochastic_k_period = temp_config_.stochastic_k_period;
            chart_panel->indicator_config_.stochastic_d_period = temp_config_.stochastic_d_period;
            chart_panel->indicator_config_.stochastic_slow_period = temp_config_.stochastic_slow_period;
            chart_panel->indicator_config_.atr_period = temp_config_.atr_period;

            // Reinitialize active indicators to reflect the new settings
            chart_panel->initialize_active_indicators();
        }
    }
}

void ChartPanelSettings::reset_temp_config() {
    if (chart_panel_ptr_) {
        ChartPanel* chart_panel = get_chart_panel();
        if (chart_panel) {
            // Copy the real indicator config to our temp config
            temp_config_.show_sma_9 = chart_panel->indicator_config_.show_sma_9;
            temp_config_.show_sma_10 = chart_panel->indicator_config_.show_sma_10;
            temp_config_.show_sma_20 = chart_panel->indicator_config_.show_sma_20;
            temp_config_.show_sma_50 = chart_panel->indicator_config_.show_sma_50;
            temp_config_.show_sma_200 = chart_panel->indicator_config_.show_sma_200;
            temp_config_.show_ema_9 = chart_panel->indicator_config_.show_ema_9;
            temp_config_.show_ema_10 = chart_panel->indicator_config_.show_ema_10;
            temp_config_.show_ema_20 = chart_panel->indicator_config_.show_ema_20;
            temp_config_.show_ema_21 = chart_panel->indicator_config_.show_ema_21;
            temp_config_.show_ema_50 = chart_panel->indicator_config_.show_ema_50;
            temp_config_.show_ema_200 = chart_panel->indicator_config_.show_ema_200;
            temp_config_.show_rsi = chart_panel->indicator_config_.show_rsi;
            temp_config_.show_macd = chart_panel->indicator_config_.show_macd;
            temp_config_.show_bollinger = chart_panel->indicator_config_.show_bollinger;
            temp_config_.show_stochastic = chart_panel->indicator_config_.show_stochastic;
            temp_config_.show_atr = chart_panel->indicator_config_.show_atr;
            temp_config_.show_volume_profile = chart_panel->indicator_config_.show_volume_profile;
            temp_config_.show_fibonacci = chart_panel->indicator_config_.show_fibonacci;
            temp_config_.show_crosshair_info = chart_panel->indicator_config_.show_crosshair_info;
            temp_config_.fib_start_price = chart_panel->indicator_config_.fib_start_price;
            temp_config_.fib_end_price = chart_panel->indicator_config_.fib_end_price;
            temp_config_.rsi_period = chart_panel->indicator_config_.rsi_period;
            temp_config_.rsi_overbought = chart_panel->indicator_config_.rsi_overbought;
            temp_config_.rsi_oversold = chart_panel->indicator_config_.rsi_oversold;
            temp_config_.bollinger_period = chart_panel->indicator_config_.bollinger_period;
            temp_config_.bollinger_std_dev = chart_panel->indicator_config_.bollinger_std_dev;
            temp_config_.macd_fast_period = chart_panel->indicator_config_.macd_fast_period;
            temp_config_.macd_slow_period = chart_panel->indicator_config_.macd_slow_period;
            temp_config_.macd_signal_period = chart_panel->indicator_config_.macd_signal_period;
            temp_config_.stochastic_k_period = chart_panel->indicator_config_.stochastic_k_period;
            temp_config_.stochastic_d_period = chart_panel->indicator_config_.stochastic_d_period;
            temp_config_.stochastic_slow_period = chart_panel->indicator_config_.stochastic_slow_period;
            temp_config_.atr_period = chart_panel->indicator_config_.atr_period;
        }
    }
}

void ChartPanelSettings::save_settings() {
    if (!chart_panel_ptr_) return;

    ChartPanel* chart_panel = get_chart_panel();
    if (!chart_panel) return;

    // Generate a unique settings key based on the panel's config
    std::string settings_key = "chart_panel_" + std::to_string(reinterpret_cast<uintptr_t>(chart_panel));

    // Create a JSON object to store the settings
    nlohmann::json settings_json;
    settings_json["show_sma_9"] = temp_config_.show_sma_9;
    settings_json["show_sma_20"] = temp_config_.show_sma_20;
    settings_json["show_sma_50"] = temp_config_.show_sma_50;
    settings_json["show_sma_200"] = temp_config_.show_sma_200;
    settings_json["show_ema_9"] = temp_config_.show_ema_9;
    settings_json["show_ema_21"] = temp_config_.show_ema_21;
    settings_json["show_ema_50"] = temp_config_.show_ema_50;
    settings_json["show_ema_200"] = temp_config_.show_ema_200;
    settings_json["show_rsi"] = temp_config_.show_rsi;
    settings_json["show_macd"] = temp_config_.show_macd;
    settings_json["show_stochastic"] = temp_config_.show_stochastic;
    settings_json["show_atr"] = temp_config_.show_atr;
    settings_json["show_volume_profile"] = temp_config_.show_volume_profile;
    settings_json["show_fibonacci"] = temp_config_.show_fibonacci;
    settings_json["show_crosshair_info"] = temp_config_.show_crosshair_info;
    settings_json["rsi_period"] = temp_config_.rsi_period;
    settings_json["rsi_overbought"] = temp_config_.rsi_overbought;
    settings_json["rsi_oversold"] = temp_config_.rsi_oversold;
    settings_json["bollinger_period"] = temp_config_.bollinger_period;
    settings_json["bollinger_std_dev"] = temp_config_.bollinger_std_dev;
    settings_json["macd_fast_period"] = temp_config_.macd_fast_period;
    settings_json["macd_slow_period"] = temp_config_.macd_slow_period;
    settings_json["macd_signal_period"] = temp_config_.macd_signal_period;
    settings_json["stochastic_k_period"] = temp_config_.stochastic_k_period;
    settings_json["stochastic_d_period"] = temp_config_.stochastic_d_period;
    settings_json["stochastic_slow_period"] = temp_config_.stochastic_slow_period;
    settings_json["atr_period"] = temp_config_.atr_period;

    // Write to a file named after the settings key
    std::string filename = "settings/" + settings_key + ".json";

    // Ensure the settings directory exists
    std::filesystem::create_directories("settings/");

    std::ofstream file(filename);
    if (file.is_open()) {
        file << settings_json.dump(4);
        file.close();
        std::cout << "Chart panel settings saved to: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save chart panel settings to: " << filename << std::endl;
    }
}

void ChartPanelSettings::load_settings() {
    if (!chart_panel_ptr_) return;

    ChartPanel* chart_panel = get_chart_panel();
    if (!chart_panel) return;

    // Generate the same settings key based on the panel's config
    std::string settings_key = "chart_panel_" + std::to_string(reinterpret_cast<uintptr_t>(chart_panel));
    std::string filename = "settings/" + settings_key + ".json";

    std::ifstream file(filename);
    if (!file.is_open()) {
        // File doesn't exist, use defaults
        std::cout << "Chart panel settings file not found: " << filename << ", using defaults" << std::endl;
        return;
    }

    try {
        nlohmann::json settings_json;
        file >> settings_json;
        file.close();

        // Load the settings from JSON
        if (settings_json.contains("show_sma_9")) temp_config_.show_sma_9 = settings_json["show_sma_9"];
        if (settings_json.contains("show_sma_20")) temp_config_.show_sma_20 = settings_json["show_sma_20"];
        if (settings_json.contains("show_sma_50")) temp_config_.show_sma_50 = settings_json["show_sma_50"];
        if (settings_json.contains("show_sma_200")) temp_config_.show_sma_200 = settings_json["show_sma_200"];
        if (settings_json.contains("show_ema_9")) temp_config_.show_ema_9 = settings_json["show_ema_9"];
        if (settings_json.contains("show_ema_21")) temp_config_.show_ema_21 = settings_json["show_ema_21"];
        if (settings_json.contains("show_ema_50")) temp_config_.show_ema_50 = settings_json["show_ema_50"];
        if (settings_json.contains("show_ema_200")) temp_config_.show_ema_200 = settings_json["show_ema_200"];
        if (settings_json.contains("show_rsi")) temp_config_.show_rsi = settings_json["show_rsi"];
        if (settings_json.contains("show_macd")) temp_config_.show_macd = settings_json["show_macd"];
        if (settings_json.contains("show_stochastic")) temp_config_.show_stochastic = settings_json["show_stochastic"];
        if (settings_json.contains("show_atr")) temp_config_.show_atr = settings_json["show_atr"];
        if (settings_json.contains("show_volume_profile")) temp_config_.show_volume_profile = settings_json["show_volume_profile"];
        if (settings_json.contains("show_fibonacci")) temp_config_.show_fibonacci = settings_json["show_fibonacci"];
        if (settings_json.contains("show_crosshair_info")) temp_config_.show_crosshair_info = settings_json["show_crosshair_info"];
        if (settings_json.contains("rsi_period")) temp_config_.rsi_period = settings_json["rsi_period"];
        if (settings_json.contains("rsi_overbought")) temp_config_.rsi_overbought = settings_json["rsi_overbought"];
        if (settings_json.contains("rsi_oversold")) temp_config_.rsi_oversold = settings_json["rsi_oversold"];
        if (settings_json.contains("bollinger_period")) temp_config_.bollinger_period = settings_json["bollinger_period"];
        if (settings_json.contains("bollinger_std_dev")) temp_config_.bollinger_std_dev = settings_json["bollinger_std_dev"];
        if (settings_json.contains("macd_fast_period")) temp_config_.macd_fast_period = settings_json["macd_fast_period"];
        if (settings_json.contains("macd_slow_period")) temp_config_.macd_slow_period = settings_json["macd_slow_period"];
        if (settings_json.contains("macd_signal_period")) temp_config_.macd_signal_period = settings_json["macd_signal_period"];
        if (settings_json.contains("stochastic_k_period")) temp_config_.stochastic_k_period = settings_json["stochastic_k_period"];
        if (settings_json.contains("stochastic_d_period")) temp_config_.stochastic_d_period = settings_json["stochastic_d_period"];
        if (settings_json.contains("stochastic_slow_period")) temp_config_.stochastic_slow_period = settings_json["stochastic_slow_period"];
        if (settings_json.contains("atr_period")) temp_config_.atr_period = settings_json["atr_period"];

        std::cout << "Chart panel settings loaded from: " << filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading chart panel settings from " << filename << ": " << e.what() << std::endl;
    }
}

} // namespace BTQuant