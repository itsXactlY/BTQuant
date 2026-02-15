/**
 * BTQuant Compute-to-ImGui Binding Implementation
 *
 * Implementation file for the binding system that connects compute-intensive
 * analytics modules to ImGui UI elements for real-time visualization
 * in professional trading dashboard applications.
 */

#include "ui/compute_to_imgui_bind.h"
#include "rendering/imgui_optimizer.hpp"
#include "imgui.h"
#include <algorithm>
#include <cmath>

namespace BTQuant {
namespace UI {

ComputeToImGuiBind::ComputeToImGuiBind() 
    : m_tpo_engine(nullptr)
    , m_liquidity_detector(nullptr)
    , m_snapshot_pipeline(nullptr)
    , m_initialized(false)
{
    m_initialized = true;
}

ComputeToImGuiBind::~ComputeToImGuiBind() {
    // Clean up resources if needed
}

void ComputeToImGuiBind::bindTPOEngine(const TPOEngine& engine, const char* window_name) {
    m_tpo_engine = &engine;
    
    // Create a visualization entry for TPO data
    BoundVisualization viz;
    viz.window_name = window_name;
    viz.is_visible = true;
    
    // Set up the render callback
    viz.render_callback = [this, &engine, window_name]() {
        if (ImGui::Begin(window_name)) {
            // Visualize TPO data
            visualizeTPOData(engine);
            
            // Render TPO profile histogram
            ImGui::Separator();
            ImGui::Text("TPO Profile Histogram:");
            renderTPOProfileHistogram(engine);
            
            // Render Value Area and POC
            ImGui::Separator();
            ImGui::Text("Value Area & POC:");
            renderValueAreaAndPOC(engine);
        }
        ImGui::End();
    };
    
    m_visualizations.push_back(viz);
}

void ComputeToImGuiBind::bindLiquiditySweepDetector(const LiquiditySweepDetector& detector, const char* window_name) {
    m_liquidity_detector = &detector;
    
    // Create a visualization entry for liquidity sweep data
    BoundVisualization viz;
    viz.window_name = window_name;
    viz.is_visible = true;
    
    // Set up the render callback
    viz.render_callback = [this, &detector, window_name]() {
        if (ImGui::Begin(window_name)) {
            // Visualize liquidity sweeps
            visualizeLiquiditySweeps(detector);
            
            // Render sweep markers
            ImGui::Separator();
            ImGui::Text("Sweep Markers:");
            renderSweepMarkers(detector);
        }
        ImGui::End();
    };
    
    m_visualizations.push_back(viz);
}

void ComputeToImGuiBind::bindLockFreeSnapshotPipeline(const LockFreeSnapshotPipeline& pipeline, const char* window_name) {
    m_snapshot_pipeline = &pipeline;

    // Create a visualization entry for market data
    BoundVisualization viz;
    viz.window_name = window_name;
    viz.is_visible = true;

    // Set up the render callback
    viz.render_callback = [this, &pipeline, window_name]() {
        if (ImGui::Begin(window_name)) {
            // Visualize market data
            visualizeMarketData(pipeline);
        }
        ImGui::End();
    };

    m_visualizations.push_back(viz);
}

void ComputeToImGuiBind::bindMarketTable(double bid_volume, double ask_volume, double last_price,
                                        double bid_price, double ask_price, const char* window_name) {
    // Create a visualization entry for market table
    BoundVisualization viz;
    viz.window_name = window_name;
    viz.is_visible = true;

    // Set up the render callback
    viz.render_callback = [bid_volume, ask_volume, last_price, bid_price, ask_price, window_name]() {
        if (ImGui::Begin(window_name)) {
            // Render the market table with [Buys | Asks | Price | Bids | Sells] format
            renderMarketTable(bid_volume, ask_volume, last_price, bid_price, ask_price);
        }
        ImGui::End();
    };

    m_visualizations.push_back(viz);
}

void ComputeToImGuiBind::render() {
    for (auto& viz : m_visualizations) {
        if (viz.is_visible && viz.render_callback) {
            viz.render_callback();
        }
    }
}

void ComputeToImGuiBind::addCustomVisualization(std::function<void()> callback, const char* window_name) {
    BoundVisualization viz;
    viz.window_name = window_name ? std::string(window_name) : "Custom Visualization";
    viz.render_callback = callback;
    viz.is_visible = true;
    
    m_visualizations.push_back(viz);
}

void ComputeToImGuiBind::update() {
    // Update any internal state if needed
    // For now, this is a placeholder for future functionality
}

void visualizeTPOData(const TPOEngine& engine, float width, float height) {
    // Get TPO data with opacity for visualization
    auto tpo_with_opacity = engine.get_tpo_data_with_opacity();
    
    if (tpo_with_opacity.empty()) {
        ImGui::Text("No TPO data available");
        return;
    }
    
    // Create a simple table to display TPO data
    if (ImGui::BeginTable("TPOData", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
        ImGui::TableSetupColumn("Price Level", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Volume", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Opacity", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableHeadersRow();
        
        int display_count = 0;
        for (const auto& [node, opacity] : tpo_with_opacity) {
            if (display_count >= 20) { // Limit display for performance
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("... and %zu more entries", tpo_with_opacity.size() - display_count);
                break;
            }
            
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%.2f", node.price_level);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%d", node.count);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.2f", node.total_volume);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.2f", opacity);
            
            display_count++;
        }
        
        ImGui::EndTable();
    }
}

void visualizeLiquiditySweeps(const LiquiditySweepDetector& detector, float width, float height) {
    // Get detected sweeps
    auto sweeps = detector.get_detected_sweeps();
    
    if (sweeps.empty()) {
        ImGui::Text("No liquidity sweeps detected");
        return;
    }
    
    // Display sweep information in a table
    if (ImGui::BeginTable("LiquiditySweeps", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Volume", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Severity", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableHeadersRow();
        
        int display_count = 0;
        for (const auto& sweep : sweeps) {
            if (display_count >= 20) { // Limit display for performance
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("... and %zu more sweeps", sweeps.size() - display_count);
                break;
            }
            
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            auto time_t = std::chrono::system_clock::to_time_t(sweep.timestamp);
            ImGui::Text("%s", std::ctime(&time_t));
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.2f", sweep.price_level);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", sweep.is_bid_sweep ? "Bid" : "Ask");
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.2f", sweep.volume_before);
            ImGui::TableSetColumnIndex(4);
            // Calculate severity as the ratio of swept volume to original volume
            double severity = sweep.volume_before > 0 ? sweep.swept_volume / sweep.volume_before : 0.0;
            ImGui::Text("%.2f", severity);
            
            display_count++;
        }
        
        ImGui::EndTable();
    }
}

void visualizeMarketData(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index, 
                        float width, float height) {
    AtomicMarketData data;
    bool success = pipeline.read_market_data_snapshot(symbol_index, data);
    
    if (!success) {
        ImGui::Text("No market data available for symbol index %u", symbol_index);
        return;
    }
    
    // Display market data in a simple format
    ImGui::Text("Symbol Index: %u", symbol_index);
    ImGui::Separator();
    
    ImGui::Text("Price: %.2f", data.price.load());
    ImGui::Text("Volume: %.2f", data.volume.load());
    ImGui::Text("Bid: %.2f @ %.2f", data.bid_price.load(), data.bid_volume.load());
    ImGui::Text("Ask: %.2f @ %.2f", data.ask_price.load(), data.ask_volume.load());
    
    // Show timestamp
    auto timestamp = data.timestamp.load();
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    ImGui::Text("Timestamp: %s", std::ctime(&time_t));
    
    // Show pipeline statistics
    auto stats = pipeline.get_stats();
    ImGui::Separator();
    ImGui::Text("Pipeline Stats:");
    ImGui::Text("Total Updates: %lu", stats.total_updates);
    ImGui::Text("Dropped Updates: %lu", stats.dropped_updates);
    ImGui::Text("Write Head: %u", stats.write_head);
    ImGui::Text("Read Tail: %u", stats.read_tail);
}

void renderTPOProfileHistogram(const TPOEngine& engine, float width, float height) {
    // Get TPO data for histogram
    auto tpo_data = engine.get_tpo_data_with_opacity();
    
    if (tpo_data.empty()) {
        ImGui::Text("No TPO data for histogram");
        return;
    }
    
    // Prepare data for histogram
    std::vector<float> values;
    std::vector<std::string> labels;
    
    for (const auto& [node, opacity] : tpo_data) {
        values.push_back(static_cast<float>(node.count));
        labels.push_back(std::to_string(node.price_level));
    }
    
    if (!values.empty()) {
        // Limit to first 20 values for readability
        size_t max_values = std::min(values.size(), static_cast<size_t>(20));
        std::vector<float> limited_values(values.begin(), values.begin() + max_values);
        std::vector<const char*> limited_labels(max_values);
        
        for (size_t i = 0; i < max_values; ++i) {
            limited_labels[i] = labels[i].c_str();
        }
        
        // Render the histogram using ImGui
        ImGui::PlotHistogram("##TPOHistogram", 
                            limited_values.data(), 
                            static_cast<int>(limited_values.size()),
                            0, 
                            nullptr, 
                            0.0f, 
                            0.0f, 
                            ImVec2(width, height));
        
        // Add price level labels below the histogram
        ImGui::Spacing();
        ImGui::Text("Price Levels:");
        std::string price_labels = "";
        for (size_t i = 0; i < std::min(labels.size(), static_cast<size_t>(20)); ++i) {
            price_labels += labels[i];
            if (i < std::min(labels.size(), static_cast<size_t>(20)) - 1) {
                price_labels += ", ";
            }
        }
        ImGui::TextWrapped("%s", price_labels.c_str());
    }
}

void renderValueAreaAndPOC(const TPOEngine& engine, float width, float height) {
    // Get POC
    double poc = engine.get_current_poc();
    
    // Get Value Area (default 70%)
    auto value_area = engine.get_current_value_area_bounds();
    
    // Display POC and Value Area
    ImGui::Text("Point of Control (POC): %.2f", poc);
    ImGui::Text("Value Area (70%%): %.2f - %.2f", value_area.first, value_area.second);
    
    // Visual representation of POC and Value Area
    ImGui::Separator();
    ImGui::Text("Visual Representation:");
    
    // Simple horizontal bar to represent value area with POC marked
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size(width, 30.0f);
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    // Draw value area background
    ImVec2 value_area_min = ImVec2(canvas_pos.x, canvas_pos.y);
    ImVec2 value_area_max = ImVec2(canvas_pos.x + width, canvas_pos.y + 30.0f);
    draw_list->AddRectFilled(value_area_min, value_area_max, IM_COL32(100, 100, 100, 100));
    
    // Draw POC marker (assuming price range is normalized to fit in width)
    // For simplicity, we'll assume the price range is from 95 to 105 (range of 10)
    double min_price = 95.0;
    double max_price = 105.0;
    double price_range = max_price - min_price;
    
    float poc_x = canvas_pos.x + ((poc - min_price) / price_range) * width;
    draw_list->AddLine(ImVec2(poc_x, canvas_pos.y), ImVec2(poc_x, canvas_pos.y + 30.0f), IM_COL32(255, 255, 0, 255), 2.0f);
    
    // Draw value area bounds
    float va_min_x = canvas_pos.x + ((value_area.first - min_price) / price_range) * width;
    float va_max_x = canvas_pos.x + ((value_area.second - min_price) / price_range) * width;
    
    draw_list->AddLine(ImVec2(va_min_x, canvas_pos.y), ImVec2(va_min_x, canvas_pos.y + 30.0f), IM_COL32(0, 255, 0, 255), 1.0f);
    draw_list->AddLine(ImVec2(va_max_x, canvas_pos.y), ImVec2(va_max_x, canvas_pos.y + 30.0f), IM_COL32(0, 255, 0, 255), 1.0f);
    
    // Advance cursor
    ImGui::Dummy(canvas_size);
}

void renderSweepMarkers(const LiquiditySweepDetector& detector, float width, float height) {
    // Get visual overlays for detected sweeps
    auto visual_overlays = detector.get_visual_overlays();

    if (visual_overlays.empty()) {
        ImGui::Text("No visual overlays for sweeps");
        return;
    }

    // Display sweep markers in a simple list
    ImGui::Text("Detected Sweep Markers:");

    for (size_t i = 0; i < std::min(visual_overlays.size(), static_cast<size_t>(10)); ++i) { // Limit display
        const auto& overlay = visual_overlays[i];
        ImGui::Text("Marker %zu: Pos=(%.1f,%.1f), Radius=%.1f, Color=(%.1f,%.1f,%.1f)",
                   i+1, overlay.x_position, overlay.y_position, overlay.radius,
                   overlay.red, overlay.green, overlay.blue);
    }

    if (visual_overlays.size() > 10) {
        ImGui::Text("... and %zu more markers", visual_overlays.size() - 10);
    }
}

void visualizeMarketDepthTable(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index,
                              float width, float height) {
    AtomicMarketData data;
    bool success = pipeline.read_market_data_snapshot(symbol_index, data);

    if (!success) {
        ImGui::Text("No market data available for symbol index %u", symbol_index);
        return;
    }

    // Get current market data values
    double price = data.price.load();
    double volume = data.volume.load();
    double bid_price = data.bid_price.load();
    double bid_volume = data.bid_volume.load();
    double ask_price = data.ask_price.load();
    double ask_volume = data.ask_volume.load();

    // Create a table to display market depth: Buys | Asks | Price | Bids | Sells
    if (ImGui::BeginTable("MarketDepthTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
        ImGui::TableSetupColumn("Buys", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Asks", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Bids", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Sells", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();

        // Buys column (represents buy-side volume at best bid)
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%.2f", bid_volume);

        // Asks column (represents sell-side volume at best ask)
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.2f", ask_volume);

        // Price column (last traded price)
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.2f", price);

        // Bids column (best bid price)
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.2f", bid_price);

        // Sells column (represents sell-side volume at best ask - equivalent to Asks)
        // In trading context, "Sells" could mean the same as "Asks" - volume available for selling
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%.2f", ask_volume);

        ImGui::EndTable();
    }

    // Show additional market data information
    ImGui::Separator();
    ImGui::Text("Additional Market Data:");
    ImGui::Text("Symbol Index: %u", symbol_index);
    ImGui::Text("Volume: %.2f", volume);

    // Show timestamp
    auto timestamp = data.timestamp.load();
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    ImGui::Text("Timestamp: %s", std::ctime(&time_t));
}

void renderMarketTable(double bid_volume, double ask_volume, double last_price, 
                      double bid_price, double ask_price, float width, float height) {
    // Create a table to display market data: Buys | Asks | Price | Bids | Sells
    if (ImGui::BeginTable("MarketTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
        ImGui::TableSetupColumn("Buys", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Asks", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Bids", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Sells", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();

        // Buys column (represents buy-side volume at best bid)
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%.2f", bid_volume);

        // Asks column (represents sell-side volume at best ask)
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.2f", ask_volume);

        // Price column (last traded price)
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.2f", last_price);

        // Bids column (best bid price)
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.2f", bid_price);

        // Sells column (represents sell-side volume at best ask)
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%.2f", ask_volume);

        ImGui::EndTable();
    }
}

} // namespace UI
} // namespace BTQuant