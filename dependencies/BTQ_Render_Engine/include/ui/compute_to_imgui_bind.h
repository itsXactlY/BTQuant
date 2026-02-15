/**
 * BTQuant Compute-to-ImGui Binding Header
 *
 * Header file for the binding system that connects compute-intensive
 * analytics modules to ImGui UI elements for real-time visualization
 * in professional trading dashboard applications.
 */

#pragma once

#include "analytics/tpoengine.h"
#include "analytics/liquiditysweepdetector.h"
#include "analytics/lockfreesnapshotpipeline.h"
#include <memory>
#include <functional>
#include <vector>
#include <string>

// Forward declaration to avoid including imgui.h in header
struct ImGuiContext;
typedef struct ImGuiContext ImGuiContext;

namespace BTQuant {
namespace UI {

/**
 * @brief Interface for binding compute results to ImGui UI elements
 */
class ComputeToImGuiBind {
public:
    /**
     * @brief Constructor
     */
    ComputeToImGuiBind();

    /**
     * @brief Destructor
     */
    ~ComputeToImGuiBind();

    /**
     * @brief Bind TPO Engine data to ImGui visualization
     * @param engine Reference to the TPO Engine instance
     * @param window_name Name of the ImGui window to render in
     */
    void bindTPOEngine(const TPOEngine& engine, const char* window_name = "TPO Visualization");

    /**
     * @brief Bind Liquidity Sweep Detector data to ImGui visualization
     * @param detector Reference to the Liquidity Sweep Detector instance
     * @param window_name Name of the ImGui window to render in
     */
    void bindLiquiditySweepDetector(const LiquiditySweepDetector& detector, const char* window_name = "Liquidity Sweeps");

    /**
     * @brief Bind Lock-Free Snapshot Pipeline data to ImGui visualization
     * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
     * @param window_name Name of the ImGui window to render in
     */
    void bindLockFreeSnapshotPipeline(const BTQuant::RenderEngine::LockFreeSnapshotPipeline& pipeline, const char* window_name = "Market Data");

    /**
     * @brief Render all bound visualizations
     */
    void render();

    /**
     * @brief Add custom visualization callback
     * @param callback Function to call during render
     * @param window_name Name of the window to render in
     */
    void addCustomVisualization(std::function<void()> callback, const char* window_name);

    /**
     * @brief Update the bound data from compute modules
     */
    void update();

private:
    struct BoundVisualization {
        std::string window_name;
        std::function<void()> render_callback;
        bool is_visible;
    };

    std::vector<BoundVisualization> m_visualizations;

    // References to compute modules (stored as weak references)
    const TPOEngine* m_tpo_engine;
    const LiquiditySweepDetector* m_liquidity_detector;
    const BTQuant::RenderEngine::LockFreeSnapshotPipeline* m_snapshot_pipeline;

    // Internal state
    bool m_initialized;
};

/**
 * @brief Helper function to visualize TPO data in ImGui
 * @param engine Reference to the TPO Engine instance
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void visualizeTPOData(const TPOEngine& engine, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to visualize Liquidity Sweep data in ImGui
 * @param detector Reference to the Liquidity Sweep Detector instance
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void visualizeLiquiditySweeps(const LiquiditySweepDetector& detector, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to visualize Market Data from Snapshot Pipeline in ImGui
 * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
 * @param symbol_index Index of the symbol to visualize
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void visualizeMarketData(const BTQuant::RenderEngine::LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index = 0,
                        float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render TPO profile as a histogram in ImGui
 * @param engine Reference to the TPO Engine instance
 * @param width Width of the histogram
 * @param height Height of the histogram
 */
void renderTPOProfileHistogram(const TPOEngine& engine, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render Value Area and POC in ImGui
 * @param engine Reference to the TPO Engine instance
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void renderValueAreaAndPOC(const TPOEngine& engine, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render sweep detection markers in ImGui
 * @param detector Reference to the Liquidity Sweep Detector instance
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void renderSweepMarkers(const LiquiditySweepDetector& detector, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to visualize market depth table in ImGui
 * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
 * @param symbol_index Index of the symbol to visualize
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void visualizeMarketDepthTable(const BTQuant::RenderEngine::LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index = 0,
                              float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render a specific market table with [Buys | Asks | Price | Bids | Sells] format
 * @param bid_volume Volume at the best bid price
 * @param ask_volume Volume at the best ask price
 * @param last_price Last traded price
 * @param bid_price Best bid price
 * @param ask_price Best ask price
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void renderMarketTable(double bid_volume, double ask_volume, double last_price, 
                      double bid_price, double ask_price, float width = 400.0f, float height = 300.0f);

} // namespace UI
} // namespace BTQuant