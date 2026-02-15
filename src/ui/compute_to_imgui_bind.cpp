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
#include "ui/font_manager.hpp"  // Include font manager for monospaced font
#include <algorithm>
#include <cmath>

namespace BTQuant {
namespace UI {

ComputeToImGuiBind::ComputeToImGuiBind() 
    : m_tpo_engine(nullptr)
    , m_liquidity_detector(nullptr)
    , m_snapshot_pipeline(nullptr)
    , m_raw_trade_table(nullptr)
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

void ComputeToImGuiBind::bindOrderBookWithToggle(double bid_volume, double ask_volume, double last_price,
                                                double bid_price, double ask_price, const char* window_name) {
    // Create a visualization entry for order book with toggle
    OrderBookVisualization order_viz;
    order_viz.window_name = window_name;
    order_viz.data = {bid_volume, ask_volume, last_price, bid_price, ask_price};
    order_viz.is_visible = true;
    order_viz.usd_display_mode = true; // Default to USD mode

    // Create a regular visualization with a render callback that accesses the order book data
    BoundVisualization viz;
    viz.window_name = window_name;
    viz.is_visible = true;
    
    // Store the order book visualization to maintain state
    size_t viz_idx = m_order_book_visualizations.size();
    m_order_book_visualizations.push_back(order_viz);

    // Set up the render callback with toggle functionality
    viz.render_callback = [this, viz_idx, window_name]() {
        if (ImGui::Begin(window_name)) {
            // Access the stored order book data and toggle state
            auto& order_book_data = m_order_book_visualizations[viz_idx];
            
            ImGui::Text("Order Book");
            ImGui::SameLine(ImGui::GetWindowWidth() - 150); // Align to right
            
            // Toggle button for USD/COIN
            if (ImGui::RadioButton("USD", order_book_data.usd_display_mode)) {
                order_book_data.usd_display_mode = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("COIN", !order_book_data.usd_display_mode)) {
                order_book_data.usd_display_mode = false;
            }
            
            ImGui::Separator();
            
            // Render the order book with the selected display mode
            if (order_book_data.usd_display_mode) {
                // If USD: Multiply atomic_size (volume) by atomic_last_price (price) during render pass
                double usd_bid_value = order_book_data.data.bid_volume * order_book_data.data.last_price;
                double usd_ask_value = order_book_data.data.ask_volume * order_book_data.data.last_price;
                
                // Render market table with USD values
                if (ImGui::BeginTable("OrderBookTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
                    ImGui::TableSetupColumn("Buys (USD)", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("Asks (USD)", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("Bids (USD)", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("Sells (USD)", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableHeadersRow();

                    ImGui::TableNextRow();

                    // Buys column (represents buy-side volume at best bid in USD)
                    ImGui::TableSetColumnIndex(0);
                    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(order_book_data.data.bid_volume * order_book_data.data.bid_price, "%.2f");

                    // Asks column (represents sell-side volume at best ask in USD)
                    ImGui::TableSetColumnIndex(1);
                    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(order_book_data.data.ask_volume * order_book_data.data.ask_price, "%.2f");

                    // Price column (last traded price)
                    ImGui::TableSetColumnIndex(2);
                    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(order_book_data.data.last_price, "%.2f");

                    // Bids column (best bid price)
                    ImGui::TableSetColumnIndex(3);
                    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(order_book_data.data.bid_price * order_book_data.data.bid_volume, "%.2f"); // USD value

                    // Sells column (represents sell-side volume at best ask in USD)
                    ImGui::TableSetColumnIndex(4);
                    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(order_book_data.data.ask_volume * order_book_data.data.ask_price, "%.2f");

                    ImGui::EndTable();
                }
                
                ImGui::Text("USD Notional Values Displayed");
            } else {
                // COIN mode - display raw values
                renderMarketTable(order_book_data.data.bid_volume, order_book_data.data.ask_volume, 
                                 order_book_data.data.last_price, order_book_data.data.bid_price, 
                                 order_book_data.data.ask_price);
                ImGui::Text("Coin Values Displayed");
            }
        }
        ImGui::End();
    };

    m_visualizations.push_back(viz);
}

void ComputeToImGuiBind::bindRawTradeTable(const RawTradeTable& trade_table, const char* window_name) {
    m_raw_trade_table = &trade_table;

    // Create a visualization entry for raw trade table
    BoundVisualization viz;
    viz.window_name = window_name;
    viz.is_visible = true;

    // Set up the render callback
    viz.render_callback = [this, &trade_table, window_name]() {
        if (ImGui::Begin(window_name)) {
            // Visualize raw trade table
            visualizeRawTradeTable(trade_table);
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

void ComputeToImGuiBind::bindHorizontalBars(const std::vector<float>& values, const std::vector<ImU32>& colors, 
                                          const char* window_name) {
    // Create a visualization entry for horizontal bars
    BoundVisualization viz;
    viz.window_name = window_name ? std::string(window_name) : "Horizontal Bars Visualization";
    viz.is_visible = true;

    // Set up the render callback - make copies of the vectors to capture in lambda
    std::vector<float> values_copy = values;
    std::vector<ImU32> colors_copy = colors;

    viz.render_callback = [values_copy, colors_copy, window_name]() {
        if (ImGui::Begin(window_name)) {
            // Render the horizontal bars visualization
            renderHorizontalBars(values_copy, colors_copy, 400.0f, 300.0f, "Horizontal Bars");
        }
        ImGui::End();
    };

    m_visualizations.push_back(viz);
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
            BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(node.price_level, "%.2f");
            ImGui::TableSetColumnIndex(1);
            BTQuant::UI::FontManager::getInstance().renderNumericalValue(node.count);
            ImGui::TableSetColumnIndex(2);
            BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(node.total_volume, "%.2f");
            ImGui::TableSetColumnIndex(3);
            BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(opacity, "%.2f");
            
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
            BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(sweep.price_level, "%.2f");
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", sweep.is_bid_sweep ? "Bid" : "Ask");
            ImGui::TableSetColumnIndex(3);
            BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(sweep.volume_before, "%.2f");
            ImGui::TableSetColumnIndex(4);
            // Calculate severity as the ratio of swept volume to original volume
            double severity = sweep.volume_before > 0 ? sweep.swept_volume / sweep.volume_before : 0.0;
            BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(severity, "%.2f");
            
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
        ImGui::Text("No market data available for symbol index ");
        ImGui::SameLine();
        BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(symbol_index));
        return;
    }

    // Display market data in a simple format
    ImGui::Text("Symbol Index: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(symbol_index));
    ImGui::Separator();

    ImGui::Text("Price: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.price.load(), "%.2f");
    
    ImGui::Text("Volume: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.volume.load(), "%.2f");

    // If USD: Multiply atomic_size (volume) by atomic_last_price (price) during render pass
    double usd_notional_value = data.volume.load() * data.price.load();
    ImGui::Text("USD Notional Value: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(usd_notional_value, "%.2f");

    ImGui::Text("Bid: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.bid_price.load(), "%.2f");
    ImGui::Text("@ ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.bid_volume.load(), "%.2f");

    ImGui::Text("Ask: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.ask_price.load(), "%.2f");
    ImGui::Text("@ ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.ask_volume.load(), "%.2f");

    // Show timestamp
    auto timestamp = data.timestamp.load();
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    ImGui::Text("Timestamp: %s", std::ctime(&time_t));

    // Show pipeline statistics
    auto stats = pipeline.get_stats();
    ImGui::Separator();
    ImGui::Text("Pipeline Stats:");
    ImGui::Text("Total Updates: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(stats.total_updates));
    
    ImGui::Text("Dropped Updates: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(stats.dropped_updates));
    
    ImGui::Text("Write Head: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(stats.write_head));
    
    ImGui::Text("Read Tail: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(stats.read_tail));
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
    ImGui::Text("Point of Control (POC): ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(poc, "%.2f");
    
    ImGui::Text("Value Area (70%%): ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(value_area.first, "%.2f");
    ImGui::Text(" - ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(value_area.second, "%.2f");
    
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
        ImGui::Text("No market data available for symbol index ");
        ImGui::SameLine();
        BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(symbol_index));
        return;
    }

    // Get current market data values
    double price = data.price.load();
    double volume = data.volume.load();
    double bid_price = data.bid_price.load();
    double bid_volume = data.bid_volume.load();
    double ask_price = data.ask_price.load();
    double ask_volume = data.ask_volume.load();

    // If USD: Multiply atomic_size (volume) by atomic_last_price (price) during render pass
    double usd_notional_value = volume * price;

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
    ImGui::Text("Symbol Index: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(symbol_index));

    ImGui::Text("Volume: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(volume, "%.2f");
    
    ImGui::Text("USD Notional Value: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(usd_notional_value, "%.2f");

    // Show timestamp
    auto timestamp = data.timestamp.load();
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    ImGui::Text("Timestamp: %s", std::ctime(&time_t));
}

void renderMarketTable(double bid_volume, double ask_volume, double last_price,
                      double bid_price, double ask_price, float width, float height) {
    // If USD: Multiply atomic_size (volume) by atomic_last_price (price) during render pass
    // Using bid_volume as atomic_size and last_price as atomic_last_price for this context
    double usd_notional_value = bid_volume * last_price;

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
        BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(bid_volume, "%.2f");

        // Asks column (represents sell-side volume at best ask)
        ImGui::TableSetColumnIndex(1);
        BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(ask_volume, "%.2f");

        // Price column (last traded price)
        ImGui::TableSetColumnIndex(2);
        BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(last_price, "%.2f");

        // Bids column (best bid price)
        ImGui::TableSetColumnIndex(3);
        BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(bid_price, "%.2f");

        // Sells column (represents sell-side volume at best ask)
        ImGui::TableSetColumnIndex(4);
        BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(ask_volume, "%.2f");

        ImGui::EndTable();
    }

    // Display the USD notional value
    ImGui::Separator();
    ImGui::Text("USD Notional Value: %.2f", usd_notional_value);
}

void renderHorizontalBars(const std::vector<float>& values, const std::vector<ImU32>& colors,
                         float width, float height, const char* label) {
    if (values.empty()) {
        ImGui::Text("No data to display");
        return;
    }

    // Create a canvas for drawing the horizontal bars
    ImGui::Text("%s", label);

    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size(width, height);

    // Draw a background rectangle
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                            IM_COL32(30, 30, 30, 200));

    // Calculate dimensions for each bar
    float bar_height = canvas_size.y / values.size();
    float max_value = 0.0f;

    // Find the maximum value to normalize the bars
    for (float val : values) {
        if (std::abs(val) > max_value) max_value = std::abs(val);
    }

    if (max_value == 0.0f) max_value = 1.0f; // Prevent division by zero

    // Draw each horizontal bar
    for (size_t i = 0; i < values.size(); ++i) {
        float normalized_value = std::abs(values[i]) / max_value;
        float bar_width = normalized_value * canvas_size.x;

        // Determine color for this bar
        ImU32 color = (i < colors.size()) ? colors[i] : IM_COL32(255, 255, 255, 255);

        // Calculate position for the bar
        ImVec2 bar_start, bar_end;

        // Center the bars in the canvas vertically
        float y_start = canvas_pos.y + i * bar_height;
        float y_end = canvas_pos.y + (i + 1) * bar_height;

        // For positive values, extend right from left edge; for negative values, extend left from right edge
        if (values[i] >= 0) {
            // Positive values: extend right from left edge
            bar_start = ImVec2(canvas_pos.x, y_start);
            bar_end = ImVec2(canvas_pos.x + bar_width, y_end);
        } else {
            // Negative values: extend left from right edge
            bar_start = ImVec2(canvas_pos.x + canvas_size.x - bar_width, y_start);
            bar_end = ImVec2(canvas_pos.x + canvas_size.x, y_end);
        }

        // Draw the filled rectangle for the bar using DrawList->AddRectFilled as requested
        draw_list->AddRectFilled(bar_start, bar_end, color);

        // Draw a border around the bar for better visibility
        draw_list->AddRect(bar_start, bar_end, IM_COL32(200, 200, 200, 100));
    }

    // Advance the cursor to account for the drawn content
    ImGui::Dummy(canvas_size);
}

void visualizeRawTradeTable(const RawTradeTable& trade_table, float width, float height) {
    // Get trade statistics
    auto stats = trade_table.get_trade_statistics();

    // Display trade statistics at the top
    ImGui::Text("Trade Statistics:");
    ImGui::Text("Total Trades: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(stats.total_trades));
    
    ImGui::Text("Total Volume: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(stats.total_volume, "%.2f");
    
    ImGui::Text("Avg Trade Size: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(stats.avg_trade_size, "%.2f");
    
    ImGui::Text("Largest Trade: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(stats.largest_trade_size, "%.2f");
    
    ImGui::Text("Buy Vol: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(stats.buy_volume, "%.2f");
    ImGui::Text(" (");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(stats.buy_count);
    ImGui::Text(" trades)");
    
    ImGui::Text("Sell Vol: ");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(stats.sell_volume, "%.2f");
    ImGui::Text(" (");
    ImGui::SameLine();
    BTQuant::UI::FontManager::getInstance().renderNumericalValue(stats.sell_count);
    ImGui::Text(" trades)");

    ImGui::Separator();

    // Get recent trades to display
    auto recent_trades = trade_table.get_recent_trades(50); // Limit to 50 for performance

    if (recent_trades.empty()) {
        ImGui::Text("No trades available");
        return;
    }

    // Create a table to display raw trade data
    if (ImGui::BeginTable("RawTradeTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
        ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Volume", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Side", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Trade ID", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (const auto& trade : recent_trades) {
            ImGui::TableNextRow();

            // Time column
            ImGui::TableSetColumnIndex(0);
            auto time_t = std::chrono::system_clock::to_time_t(trade.timestamp);
            std::tm tm_local;
            localtime_r(&time_t, &tm_local); // Use thread-safe version
            char time_str[100];
            std::strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm_local);
            ImGui::Text("%s", time_str);

            // Price column
            ImGui::TableSetColumnIndex(1);
            BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(trade.price, "%.2f");

            // Volume column
            ImGui::TableSetColumnIndex(2);
            BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(trade.volume, "%.2f");

            // Side column with color coding
            ImGui::TableSetColumnIndex(3);
            if (trade.side == 'B' || trade.side == 'b') {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "BUY"); // Green for buy
            } else if (trade.side == 'S' || trade.side == 's') {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "SELL"); // Red for sell
            } else {
                ImGui::Text("N/A");
            }

            // Trade ID column
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%s", trade.trade_id.c_str());
        }

        ImGui::EndTable();
    }

    // Show additional controls
    ImGui::Separator();
    if (ImGui::SmallButton("Clear All Trades")) {
        // Note: In a real implementation, you might want to use a command pattern
        // to avoid modifying data directly from the UI thread
        // For now, we'll just show a notification
        ImGui::Text("Clear command sent");
    }

    if (ImGui::SmallButton("Export to CSV")) {
        ImGui::Text("Export command sent");
    }
}

void ComputeToImGuiBind::bindLiveBidAskButton(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index, const char* window_name) {
    // Create a visualization entry for the live bid/ask button
    BoundVisualization viz;
    viz.window_name = window_name;
    viz.is_visible = true;

    // Set up the render callback that fetches live atomic data
    viz.render_callback = [&pipeline, symbol_index, window_name]() {
        if (ImGui::Begin(window_name)) {
            AtomicMarketData data;
            bool success = pipeline.read_market_data_snapshot(symbol_index, data);

            if (success) {
                // Get the current bid and ask prices from atomic data
                double bid_price = data.bid_price.load();
                double ask_price = data.ask_price.load();

                // Format the button text with live bid/ask
                char button_text[128];
                snprintf(button_text, sizeof(button_text), "Best Bid: %.2f | Best Ask: %.2f", bid_price, ask_price);

                // Create the button with the live bid/ask data in the text
                if (ImGui::Button(button_text)) {
                    // Button click handler - could trigger additional actions if needed
                    // For now, just showing the live data
                }

                // Optionally show additional info
                ImGui::Text("Symbol Index: ");
                ImGui::SameLine();
                BTQuant::UI::FontManager::getInstance().renderNumericalValue(static_cast<int>(symbol_index));

                ImGui::Text("Last Price: ");
                ImGui::SameLine();
                BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.price.load(), "%.2f");

                ImGui::Text("Bid Volume: ");
                ImGui::SameLine();
                BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.bid_volume.load(), "%.2f");

                ImGui::Text("Ask Volume: ");
                ImGui::SameLine();
                BTQuant::UI::FontManager::getInstance().renderFormattedNumericalValue(data.ask_volume.load(), "%.2f");
            } else {
                // If no data available, show a placeholder button
                if (ImGui::Button("No Data Available")) {
                    // Button click handler
                }
            }
        }
        ImGui::End();
    };

    m_visualizations.push_back(viz);
}

bool ComputeToImGuiBind::renderLiveBidAskButton(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index, const char* button_label) {
    AtomicMarketData data;
    bool success = pipeline.read_market_data_snapshot(symbol_index, data);

    if (success) {
        // Get the current bid and ask prices from atomic data
        double bid_price = data.bid_price.load();
        double ask_price = data.ask_price.load();

        // Format the button text with live bid/ask
        char button_text[128];
        snprintf(button_text, sizeof(button_text), "%s: %.2f | %.2f", button_label, bid_price, ask_price);

        // Create the button with the live bid/ask data in the text
        return ImGui::Button(button_text);
    } else {
        // If no data available, show a placeholder button
        return ImGui::Button("No Data Available");
    }
}

void ComputeToImGuiBind::bindMouseTradingInterface(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index, const char* window_name) {
    // Create a visualization entry for the mouse trading interface
    BoundVisualization viz;
    viz.window_name = window_name;
    viz.is_visible = true;

    // Set up the render callback that fetches live atomic data and creates massive trading buttons
    viz.render_callback = [&pipeline, symbol_index, window_name]() {
        if (ImGui::Begin(window_name)) {
            AtomicMarketData data;
            bool success = pipeline.read_market_data_snapshot(symbol_index, data);

            if (success) {
                // Get the current bid and ask prices from atomic data
                double bid_price = data.bid_price.load();
                double ask_price = data.ask_price.load();
                
                // Create a toggle for mouse trading mode
                static bool mouse_trading_enabled = true; // Default to enabled
                
                // Toggle button for mouse trading
                if (ImGui::Checkbox("Enable Mouse Trading", &mouse_trading_enabled)) {
                    // Toggle state changed
                }
                
                ImGui::Separator();
                
                // If Mouse Trading enabled: Render massive BUY MKT / SELL MKT buttons
                if (mouse_trading_enabled) {
                    // Market Buy button with Best Ask - MASSIVE BUTTONS
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.8f, 0.0f, 1.0f)); // Brighter green
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(20, 20)); // Increase padding
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 10)); // Adjust spacing
                    
                    char buy_label[64];
                    snprintf(buy_label, sizeof(buy_label), "BUY MKT\n%.2f", ask_price);
                    if (ImGui::Button(buy_label, ImVec2(200, 100))) {  // MASSIVE button size
                        // Execute market buy order - in a real implementation this would connect to trading interface
                        printf("Executing BUY MKT order at price: %.2f\n", ask_price);
                    }
                    
                    ImGui::PopStyleVar(2);
                    ImGui::PopStyleColor();

                    ImGui::Spacing();

                    // Market Sell button with Best Bid - MASSIVE BUTTONS
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.9f, 0.0f, 0.0f, 1.0f)); // Brighter red
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(20, 20)); // Increase padding
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 10)); // Adjust spacing
                    
                    char sell_label[64];
                    snprintf(sell_label, sizeof(sell_label), "SELL MKT\n%.2f", bid_price);
                    if (ImGui::Button(sell_label, ImVec2(200, 100))) {  // MASSIVE button size
                        // Execute market sell order - in a real implementation this would connect to trading interface
                        printf("Executing SELL MKT order at price: %.2f\n", bid_price);
                    }
                    
                    ImGui::PopStyleVar(2);
                    ImGui::PopStyleColor();

                    ImGui::Separator();
                }
                
                // Show current market data
                ImGui::Text("Current Market Data:");
                ImGui::Text("Best Bid: %.2f", bid_price);
                ImGui::Text("Best Ask: %.2f", ask_price);
                ImGui::Text("Last Price: %.2f", data.price.load());
                ImGui::Text("Bid Volume: %.2f", data.bid_volume.load());
                ImGui::Text("Ask Volume: %.2f", data.ask_volume.load());
            } else {
                // If no data available, show a placeholder
                ImGui::Text("No market data available");
            }
        }
        ImGui::End();
    };

    m_visualizations.push_back(viz);
}

} // namespace UI
} // namespace BTQuant