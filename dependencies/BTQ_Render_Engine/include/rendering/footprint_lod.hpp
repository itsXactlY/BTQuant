#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>

#include "imgui.h"
#include "implot.h"

// Custom hash function for std::pair used in grid-based aggregation
struct PairHash {
    template<class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

// Forward declaration to avoid circular dependency
namespace BTQuant {
    struct FootprintCell;  // Defined in footprint_panel.hpp
    class FootprintPanel;  // Defined in footprint_panel.hpp
}

namespace BTQuant {
namespace Rendering {

// Level of Detail enumeration for footprint rendering
enum class LODLevel {
    LOW_DETAIL,      // Minimal detail, heatmap only, no text
    MEDIUM_DETAIL,   // Basic detail with borders, no text
    HIGH_DETAIL,     // Full detail with text and labels
    MAX_DETAIL       // Ultra detail with all annotations
};

// Structure to represent LOD transition state for smooth transitions
struct LODTransitionState {
    LODLevel from_lod;
    LODLevel to_lod;
    float transition_progress;  // Value between 0.0 and 1.0 representing transition progress
};

// Structure to hold rendering settings for different LOD levels
struct LODRenderSettings {
    bool render_heatmap = true;
    bool render_borders = true;
    bool render_text = false;
    bool render_labels = false;
    bool render_detailed_annotations = false;
    float alpha_multiplier = 1.0f;
    float border_thickness = 1.0f;
};

// Structure to hold LOD statistics
struct LODStatistics {
    int total_cells = 0;
    int low_detail_cells = 0;
    int medium_detail_cells = 0;
    int high_detail_cells = 0;
    int max_detail_cells = 0;

    float getLowDetailPercentage() const {
        return total_cells > 0 ? (static_cast<float>(low_detail_cells) / total_cells) * 100.0f : 0.0f;
    }

    float getMediumDetailPercentage() const {
        return total_cells > 0 ? (static_cast<float>(medium_detail_cells) / total_cells) * 100.0f : 0.0f;
    }

    float getHighDetailPercentage() const {
        return total_cells > 0 ? (static_cast<float>(high_detail_cells) / total_cells) * 100.0f : 0.0f;
    }

    float getMaxDetailPercentage() const {
        return total_cells > 0 ? (static_cast<float>(max_detail_cells) / total_cells) * 100.0f : 0.0f;
    }
};

// Structure to hold performance metrics for adaptive LOD
struct PerformanceMetrics {
    float frame_time_ms = 0.0f;      // Time taken to render current frame
    int rendered_cells = 0;          // Number of cells rendered in current frame
    float fps = 0.0f;                // Frames per second
    bool performance_degraded = false; // Flag indicating performance issues
};

class FootprintLOD {
public:
    FootprintLOD();

    // Calculate the appropriate LOD level based on cell dimensions and zoom factor
    LODLevel calculateLODLevel(float cell_width_px, float cell_height_px, float zoom_factor) const;

    // Calculate progressive LOD level with smooth transitions between levels
    float calculateProgressiveLOD(float cell_width_px, float cell_height_px, float zoom_factor) const;

    // Calculate the appropriate LOD level with dynamic thresholds based on view range
    LODLevel calculateDynamicLODLevel(float cell_width_px, float cell_height_px,
                                   float zoom_factor, float view_range_x, float view_range_y) const;

    // Calculate LOD level with hierarchical adjustments
    LODLevel calculateHierarchicalLOD(float cell_width_px, float cell_height_px,
                                   float zoom_factor, int hierarchy_level) const;

    // Calculate LOD level with distance-based adjustment (closer cells get more detail)
    LODLevel calculateDistanceBasedLODLevel(float cell_width_px, float cell_height_px,
                                         float zoom_factor, ImVec2 cell_center, ImVec2 view_center) const;

    // Get rendering settings for a specific LOD level
    LODRenderSettings getRenderSettings(LODLevel lod_level) const;

    // Get progressive rendering settings for smooth transitions between LOD levels
    LODRenderSettings getProgressiveRenderSettings(float lod_value, float cell_area_px) const;

    // Get optimized rendering settings considering cell area to prevent overcrowding
    LODRenderSettings getOptimizedRenderSettings(LODLevel lod_level, float cell_area_px) const;

    // Determine if text should be rendered based on cell size and zoom
    bool shouldRenderText(float cell_height_px, float zoom_factor) const;

    // Determine if labels should be rendered based on cell size and zoom
    bool shouldRenderLabels(float cell_height_px, float zoom_factor) const;

    // Determine if detailed annotations should be rendered
    bool shouldRenderDetailedAnnotations(float cell_width_px, float cell_height_px,
                                       float zoom_factor) const;

    // Adjust cell padding based on zoom level for optimal visual representation
    float adjustCellPadding(float base_padding, float zoom_factor) const;

    // Calculate alpha multiplier based on zoom and LOD level
    float calculateAlphaMultiplier(float zoom_factor, LODLevel lod_level) const;

    // Calculate LOD statistics for a set of cells
    LODStatistics calculateLODStatistics(const std::vector<FootprintCell>& cells,
                                       float zoom_factor) const;

    // Calculate adaptive cell size based on zoom and LOD level
    float calculateAdaptiveCellSize(float base_size, float zoom_factor, LODLevel lod_level) const;

    // Calculate LOD transition state for smooth transitions between LOD levels
    LODTransitionState calculateLODTransition(float prev_zoom, float curr_zoom,
                                           float cell_width_px, float cell_height_px) const;

    // Calculate cached LOD level to improve performance
    LODLevel calculateCachedLOD(float cell_width_px, float cell_height_px, float zoom_factor) const;

    // Clear the LOD cache to free memory
    void clearLODCache() const;

    // Calculate zoom-based LOD with consideration for view range
    LODLevel calculateZoomBasedLOD(float cell_width_px, float cell_height_px,
                               float zoom_factor, float view_range_x, float view_range_y) const;

    // Calculate adaptive LOD based on view area to optimize performance
    LODLevel calculateAdaptiveLODBasedOnViewArea(float cell_width_px, float cell_height_px,
                                               float zoom_factor, float view_width, float view_height) const;

    // Determine if a cell should be completely skipped from rendering
    bool shouldCompletelySkipRendering(float cell_width_px, float cell_height_px,
                                    float zoom_factor) const;

    // Get simplified rendering settings for better performance when zoomed out
    LODRenderSettings getSimplifiedRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply performance-based LOD adjustment based on rendering metrics
    void updatePerformanceBasedLOD(const PerformanceMetrics& metrics);

    // Apply adaptive LOD adjustment based on cell density and performance
    void updateAdaptiveLOD(const PerformanceMetrics& metrics, int total_cells_in_view);

    // Cluster nearby cells at low zoom levels to reduce visual clutter
    std::vector<FootprintCell> clusterCells(const std::vector<FootprintCell>& cells,
                                          float zoom_factor) const;

    // Adaptive clustering considering both zoom and cell density
    std::vector<FootprintCell> adaptiveClusterCells(const std::vector<FootprintCell>& cells,
                                                 float zoom_factor,
                                                 int total_cells_in_view) const;

    // Apply LOD-based rendering to a single cell
    void applyLODToCell(const FootprintCell& cell,
                       ImDrawList* draw_list,
                       float zoom_factor,
                       double max_volume,
                       const std::vector<FootprintCell>& diagonal_imbalances,
                       const std::vector<FootprintCell>& stacked_imbalances,
                       const FootprintPanel* panel) const;

    // Apply distance-based LOD rendering to a single cell
    void applyDistanceBasedLODToCell(const FootprintCell& cell,
                                   ImDrawList* draw_list,
                                   float zoom_factor,
                                   double max_volume,
                                   const std::vector<FootprintCell>& diagonal_imbalances,
                                   const std::vector<FootprintCell>& stacked_imbalances,
                                   const FootprintPanel* panel,
                                   ImVec2 view_center) const;

    // Apply simplified LOD rendering to a single cell (optimized for zoomed-out views)
    void applySimplifiedLODToCell(const FootprintCell& cell,
                                 ImDrawList* draw_list,
                                 float zoom_factor,
                                 double max_volume,
                                 const FootprintPanel* panel) const;

    // Calculate enhanced detail LOD for when zoomed in significantly
    LODLevel calculateEnhancedDetailLOD(float cell_width_px, float cell_height_px,
                                     float zoom_factor) const;

    // Get enhanced rendering settings for increased detail when zoomed in
    LODRenderSettings getEnhancedRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply enhanced LOD rendering to a single cell (optimized for zoomed-in views)
    void applyEnhancedLODToCell(const FootprintCell& cell,
                               ImDrawList* draw_list,
                               float zoom_factor,
                               double max_volume,
                               const std::vector<FootprintCell>& diagonal_imbalances,
                               const std::vector<FootprintCell>& stacked_imbalances,
                               const FootprintPanel* panel) const;

    // Apply progressive LOD rendering to a single cell (smooth transitions between levels)
    void applyProgressiveLODToCell(const FootprintCell& cell,
                                  ImDrawList* draw_list,
                                  float zoom_factor,
                                  double max_volume,
                                  const std::vector<FootprintCell>& diagonal_imbalances,
                                  const std::vector<FootprintCell>& stacked_imbalances,
                                  const FootprintPanel* panel) const;

    // Calculate multi-resolution LOD based on zoom level and cell density
    LODLevel calculateMultiResolutionLOD(float cell_width_px, float cell_height_px,
                                      float zoom_factor, int total_cells_in_view) const;

    // Get multi-resolution rendering settings based on cell density
    LODRenderSettings getMultiResolutionRenderSettings(LODLevel lod_level, int total_cells_in_view) const;

    // Apply multi-resolution LOD to cell rendering
    void applyMultiResolutionLODToCell(const FootprintCell& cell,
                                      ImDrawList* draw_list,
                                      float zoom_factor,
                                      double max_volume,
                                      const std::vector<FootprintCell>& diagonal_imbalances,
                                      const std::vector<FootprintCell>& stacked_imbalances,
                                      const FootprintPanel* panel,
                                      int total_cells_in_view) const;

    // Calculate predictive LOD based on expected zoom changes
    LODLevel calculatePredictiveLOD(float cell_width_px, float cell_height_px,
                                float current_zoom, float predicted_zoom) const;

    // Calculate hybrid LOD combining multiple factors
    LODLevel calculateHybridLOD(float cell_width_px, float cell_height_px,
                            float zoom_factor, int total_cells_in_view,
                            const PerformanceMetrics& metrics) const;

    // Calculate adaptive temporal LOD based on how recently the cell was updated
    LODLevel calculateAdaptiveTemporalLOD(float cell_width_px, float cell_height_px,
                                      float zoom_factor, float time_since_update) const;

    // Get temporal LOD rendering settings based on how recently the cell was updated
    LODRenderSettings getTemporalLODRenderSettings(LODLevel lod_level, float time_since_update) const;

    // Calculate contextual LOD based on activity in surrounding cells
    LODLevel calculateContextualLOD(float cell_width_px, float cell_height_px,
                                float zoom_factor, const std::vector<FootprintCell>& nearby_cells) const;

    // Apply contextual LOD to cell rendering
    void applyContextualLODToCell(const FootprintCell& cell,
                                 ImDrawList* draw_list,
                                 float zoom_factor,
                                 double max_volume,
                                 const std::vector<FootprintCell>& diagonal_imbalances,
                                 const std::vector<FootprintCell>& stacked_imbalances,
                                 const FootprintPanel* panel,
                                 const std::vector<FootprintCell>& nearby_cells) const;

    // Calculate foveated LOD (high detail near focus point, decreasing detail further away)
    LODLevel calculateFoveatedLOD(float cell_width_px, float cell_height_px,
                              float zoom_factor, ImVec2 cell_center, ImVec2 focus_point,
                              float focus_radius_inner, float focus_radius_outer) const;

    // Apply foveated LOD to cell rendering
    void applyFoveatedLODToCell(const FootprintCell& cell,
                               ImDrawList* draw_list,
                               float zoom_factor,
                               double max_volume,
                               const std::vector<FootprintCell>& diagonal_imbalances,
                               const std::vector<FootprintCell>& stacked_imbalances,
                               const FootprintPanel* panel,
                               ImVec2 focus_point,
                               float focus_radius_inner,
                               float focus_radius_outer) const;

    // Calculate LOD level specifically optimized for zoomed-out views
    LODLevel calculateZoomOutLOD(float cell_width_px, float cell_height_px,
                               float zoom_factor) const;

    // Calculate LOD level specifically optimized for zoomed-in views
    LODLevel calculateZoomInLOD(float cell_width_px, float cell_height_px,
                              float zoom_factor) const;

    // Get render settings optimized for zoomed-out views
    LODRenderSettings getZoomOutOptimizedRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Get render settings enhanced for zoomed-in views
    LODRenderSettings getZoomInEnhancedRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply zoom-aware LOD to cell rendering (combines zoom-out and zoom-in optimizations)
    void applyZoomInOutLODToCell(const FootprintCell& cell,
                                ImDrawList* draw_list,
                                float zoom_factor,
                                double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                const FootprintPanel* panel) const;

    // Calculate adaptive zoom LOD considering both zoom level and cell density
    LODLevel calculateAdaptiveZoomLOD(float cell_width_px, float cell_height_px,
                                   float zoom_factor, float view_width, float view_height,
                                   int total_cells_in_view) const;

    // Get adaptive zoom render settings based on zoom level and cell count
    LODRenderSettings getAdaptiveZoomRenderSettings(LODLevel lod_level,
                                               float zoom_factor,
                                               int total_cells_in_view) const;

    // Apply adaptive zoom LOD to cell rendering considering view parameters and cell density
    void applyAdaptiveZoomLODToCell(const FootprintCell& cell,
                                   ImDrawList* draw_list,
                                   float zoom_factor,
                                   double max_volume,
                                   const std::vector<FootprintCell>& diagonal_imbalances,
                                   const std::vector<FootprintCell>& stacked_imbalances,
                                   const FootprintPanel* panel,
                                   float view_width,
                                   float view_height,
                                   int total_cells_in_view) const;

    // Calculate smooth zoom LOD to prevent flickering during zoom transitions
    LODLevel calculateSmoothZoomLOD(float cell_width_px, float cell_height_px,
                                float zoom_factor, float prev_zoom_factor) const;

    // Get smooth zoom render settings to prevent flickering
    LODRenderSettings getSmoothZoomRenderSettings(LODLevel lod_level,
                                               float zoom_factor,
                                               float prev_zoom_factor) const;

    // Apply smooth zoom LOD to cell rendering to prevent flickering during zoom transitions
    void applySmoothZoomLODToCell(const FootprintCell& cell,
                                 ImDrawList* draw_list,
                                 float zoom_factor,
                                 float prev_zoom_factor,
                                 double max_volume,
                                 const std::vector<FootprintCell>& diagonal_imbalances,
                                 const std::vector<FootprintCell>& stacked_imbalances,
                                 const FootprintPanel* panel) const;

    // Calculate gradient-based LOD considering volume/activity differences with neighboring cells
    LODLevel calculateGradientBasedLOD(const FootprintCell& cell,
                                   float cell_width_px, float cell_height_px,
                                   float zoom_factor,
                                   const std::vector<FootprintCell>& gradient_neighbors) const;

    // Apply gradient-based LOD to cell rendering considering volume/activity differences with neighboring cells
    void applyGradientBasedLODToCell(const FootprintCell& cell,
                                   ImDrawList* draw_list,
                                   float zoom_factor,
                                   double max_volume,
                                   const std::vector<FootprintCell>& diagonal_imbalances,
                                   const std::vector<FootprintCell>& stacked_imbalances,
                                   const FootprintPanel* panel,
                                   const std::vector<FootprintCell>& gradient_neighbors) const;

    // Calculate zoom-level optimized LOD for better performance across different zoom levels
    LODLevel calculateZoomLevelLOD(float cell_width_px, float cell_height_px,
                               float zoom_factor) const;

    // Get zoom-level optimized render settings
    LODRenderSettings getZoomLevelRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply zoom-level optimized LOD to cell rendering
    void applyZoomLevelLODToCell(const FootprintCell& cell,
                                ImDrawList* draw_list,
                                float zoom_factor,
                                double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                const FootprintPanel* panel) const;

    // Calculate tile-based LOD for managing very large datasets efficiently
    LODLevel calculateTileBasedLOD(float cell_width_px, float cell_height_px,
                               float zoom_factor, int tile_size_px = 256) const;

    // Apply tile-based LOD to cell rendering
    void applyTileBasedLODToCell(const FootprintCell& cell,
                                ImDrawList* draw_list,
                                float zoom_factor,
                                double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                const FootprintPanel* panel,
                                int tile_size_px = 256) const;

    // Calculate enhanced LOD for when zoomed in significantly
    LODLevel calculateEnhancedZoomInLOD(float cell_width_px, float cell_height_px,
                                    float zoom_factor) const;

    // Get enhanced rendering settings for increased detail when zoomed in significantly
    LODRenderSettings getEnhancedZoomInRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply enhanced LOD rendering to a single cell when zoomed in significantly
    void applyEnhancedZoomInLODToCell(const FootprintCell& cell,
                                     ImDrawList* draw_list,
                                     float zoom_factor,
                                     double max_volume,
                                     const std::vector<FootprintCell>& diagonal_imbalances,
                                     const std::vector<FootprintCell>& stacked_imbalances,
                                     const FootprintPanel* panel) const;

    // Calculate simplified LOD for when zoomed out significantly
    LODLevel calculateSimplifiedZoomOutLOD(float cell_width_px, float cell_height_px,
                                       float zoom_factor) const;

    // Get simplified rendering settings for zoomed-out views
    LODRenderSettings getSimplifiedZoomOutRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply simplified LOD rendering to a single cell when zoomed out significantly
    void applySimplifiedZoomOutLODToCell(const FootprintCell& cell,
                                        ImDrawList* draw_list,
                                        float zoom_factor,
                                        double max_volume,
                                        const FootprintPanel* panel) const;

    // Aggregate cells for zoomed-out views to reduce visual clutter
    std::vector<FootprintCell> aggregateCellsForZoomOut(const std::vector<FootprintCell>& cells,
                                                     float zoom_factor,
                                                     float grid_size = 0.1f) const;

    // Calculate advanced zoom-based LOD with smooth transitions
    LODLevel calculateAdvancedZoomLOD(float cell_width_px, float cell_height_px,
                                  float zoom_factor) const;

    // Get advanced zoom-based render settings
    LODRenderSettings getAdvancedZoomRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply advanced zoom-based LOD to cell rendering
    void applyAdvancedZoomLODToCell(const FootprintCell& cell,
                                   ImDrawList* draw_list,
                                   float zoom_factor,
                                   double max_volume,
                                   const std::vector<FootprintCell>& diagonal_imbalances,
                                   const std::vector<FootprintCell>& stacked_imbalances,
                                   const FootprintPanel* panel) const;

    // Calculate multi-scale LOD that adapts to different viewing scales
    LODLevel calculateMultiScaleLOD(float cell_width_px, float cell_height_px,
                                float zoom_factor, float view_scale) const;

    // Get multi-scale render settings
    LODRenderSettings getMultiScaleRenderSettings(LODLevel lod_level, float view_scale) const;

    // Apply multi-scale LOD to cell rendering
    void applyMultiScaleLODToCell(const FootprintCell& cell,
                                 ImDrawList* draw_list,
                                 float zoom_factor,
                                 double max_volume,
                                 const std::vector<FootprintCell>& diagonal_imbalances,
                                 const std::vector<FootprintCell>& stacked_imbalances,
                                 const FootprintPanel* panel,
                                 float view_scale) const;

    // Calculate continuous LOD for smooth transitions between zoom levels
    LODLevel calculateContinuousLOD(float cell_width_px, float cell_height_px,
                               float zoom_factor) const;

    // Get continuous LOD render settings for smooth transitions
    LODRenderSettings getContinuousLODRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply continuous LOD to cell rendering for smooth transitions
    void applyContinuousLODToCell(const FootprintCell& cell,
                                ImDrawList* draw_list,
                                float zoom_factor,
                                double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                const FootprintPanel* panel) const;

    // Helper function to calculate luminance of a color
    float calculateLuminance(ImU32 color) const;

    // Helper function to determine appropriate text color based on background luminance
    ImU32 getTextColorForBackground(ImU32 backgroundColor) const;

    // Apply zoom-out simplification LOD to cell rendering
    void applyZoomOutSimplificationLODToCell(const FootprintCell& cell,
                                           ImDrawList* draw_list,
                                           float zoom_factor,
                                           double max_volume,
                                           const FootprintPanel* panel) const;

    // Calculate edge-based LOD that emphasizes important boundaries
    LODLevel calculateEdgeBasedLOD(float cell_width_px, float cell_height_px,
                              float zoom_factor, bool is_edge_cell) const;

    // Get edge-based render settings
    LODRenderSettings getEdgeBasedRenderSettings(LODLevel lod_level, bool is_edge_cell) const;

    // Apply edge-based LOD to cell rendering
    void applyEdgeBasedLODToCell(const FootprintCell& cell,
                                ImDrawList* draw_list,
                                float zoom_factor,
                                double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                const FootprintPanel* panel,
                                bool is_edge_cell) const;

    // Calculate priority-based LOD that renders important cells with higher detail
    LODLevel calculatePriorityBasedLOD(float cell_width_px, float cell_height_px,
                                  float zoom_factor, int priority_level) const;

    // Get priority-based render settings
    LODRenderSettings getPriorityBasedRenderSettings(LODLevel lod_level, int priority_level) const;

    // Apply priority-based LOD to cell rendering
    void applyPriorityBasedLODToCell(const FootprintCell& cell,
                                    ImDrawList* draw_list,
                                    float zoom_factor,
                                    int priority_level,
                                    double max_volume,
                                    const std::vector<FootprintCell>& diagonal_imbalances,
                                    const std::vector<FootprintCell>& stacked_imbalances,
                                    const FootprintPanel* panel) const;

    // Calculate intelligent zoom LOD that adapts based on user interaction patterns
    LODLevel calculateIntelligentZoomLOD(float cell_width_px, float cell_height_px,
                                    float zoom_factor, float time_spent_at_zoom) const;

    // Get intelligent zoom render settings based on user interaction patterns
    LODRenderSettings getIntelligentZoomRenderSettings(LODLevel lod_level,
                                                  float time_spent_at_zoom) const;

    // Apply intelligent zoom LOD to cell rendering based on user interaction patterns
    void applyIntelligentZoomLODToCell(const FootprintCell& cell,
                                   ImDrawList* draw_list,
                                   float zoom_factor,
                                   float time_spent_at_zoom,
                                   double max_volume,
                                   const std::vector<FootprintCell>& diagonal_imbalances,
                                   const std::vector<FootprintCell>& stacked_imbalances,
                                   const FootprintPanel* panel) const;

    // Calculate smart LOD that balances performance and visual quality based on multiple factors
    LODLevel calculateSmartLOD(float cell_width_px, float cell_height_px,
                          float zoom_factor, int total_cells_in_view,
                          const PerformanceMetrics& metrics) const;

    // Get smart LOD render settings based on multiple factors
    LODRenderSettings getSmartLODRenderSettings(LODLevel lod_level,
                                           int total_cells_in_view,
                                           const PerformanceMetrics& metrics) const;

    // Apply smart LOD to cell rendering balancing performance and visual quality
    void applySmartLODToCell(const FootprintCell& cell,
                            ImDrawList* draw_list,
                            float zoom_factor,
                            int total_cells_in_view,
                            const PerformanceMetrics& metrics,
                            double max_volume,
                            const std::vector<FootprintCell>& diagonal_imbalances,
                            const std::vector<FootprintCell>& stacked_imbalances,
                            const FootprintPanel* panel) const;

    // Calculate zoom-dependent LOD for managing detail based on zoom level
    LODLevel calculateZoomDependentLOD(float cell_width_px, float cell_height_px,
                                  float zoom_factor) const;

    // Get zoom-dependent render settings for managing detail based on zoom level
    LODRenderSettings getZoomDependentRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply zoom-dependent LOD to cell rendering
    void applyZoomDependentLODToCell(const FootprintCell& cell,
                                ImDrawList* draw_list,
                                float zoom_factor,
                                double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                const FootprintPanel* panel) const;

    // Calculate progressive zoom-based LOD for smooth transitions
    LODLevel calculateProgressiveZoomLOD(float cell_width_px, float cell_height_px,
                                    float zoom_factor) const;

    // Get progressive zoom-based render settings for smooth transitions
    LODRenderSettings getProgressiveZoomRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply progressive zoom-based LOD to cell rendering for smooth transitions
    void applyProgressiveZoomLODToCell(const FootprintCell& cell,
                                  ImDrawList* draw_list,
                                  float zoom_factor,
                                  double max_volume,
                                  const std::vector<FootprintCell>& diagonal_imbalances,
                                  const std::vector<FootprintCell>& stacked_imbalances,
                                  const FootprintPanel* panel) const;

    // Calculate core LOD functionality that reduces detail when zoomed out
    // and increases detail when zoomed in, with smooth transitions between levels
    LODLevel calculateCoreLOD(float cell_width_px, float cell_height_px,
                         float zoom_factor) const;

    // Get core LOD render settings
    LODRenderSettings getCoreLODRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply core LOD to cell rendering
    void applyCoreLODToCell(const FootprintCell& cell,
                       ImDrawList* draw_list,
                       float zoom_factor,
                       double max_volume,
                       const std::vector<FootprintCell>& diagonal_imbalances,
                       const std::vector<FootprintCell>& stacked_imbalances,
                       const FootprintPanel* panel) const;

    // Calculate ultimate zoom-based LOD for the best balance of performance and visual quality
    LODLevel calculateUltimateZoomLOD(float cell_width_px, float cell_height_px,
                                 float zoom_factor) const;

    // Get ultimate zoom-based render settings for the best balance of performance and visual quality
    LODRenderSettings getUltimateZoomRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply ultimate zoom-based LOD to cell rendering for the best balance of performance and visual quality
    void applyUltimateZoomLODToCell(const FootprintCell& cell,
                               ImDrawList* draw_list,
                               float zoom_factor,
                               double max_volume,
                               const std::vector<FootprintCell>& diagonal_imbalances,
                               const std::vector<FootprintCell>& stacked_imbalances,
                               const FootprintPanel* panel) const;

    // Calculate main LOD functionality that specifically focuses on reducing detail when zoomed out
    // and increasing detail when zoomed in, with optimized performance characteristics
    LODLevel calculateMainLOD(float cell_width_px, float cell_height_px,
                         float zoom_factor) const;

    // Get main LOD render settings for reducing detail when zoomed out and increasing detail when zoomed in
    LODRenderSettings getMainLODRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply main LOD to cell rendering that reduces detail when zoomed out and increases detail when zoomed in
    void applyMainLODToCell(const FootprintCell& cell,
                       ImDrawList* draw_list,
                       float zoom_factor,
                       double max_volume,
                       const std::vector<FootprintCell>& diagonal_imbalances,
                       const std::vector<FootprintCell>& stacked_imbalances,
                       const FootprintPanel* panel) const;

    // Calculate zoom-based LOD that specifically focuses on reducing detail when zoomed out
    // and increasing detail when zoomed in with optimized performance
    LODLevel calculateZoomBasedLODDetail(float cell_width_px, float cell_height_px,
                                   float zoom_factor) const;

    // Get zoom-based LOD render settings that specifically focuses on reducing detail when zoomed out
    // and increasing detail when zoomed in with optimized performance
    LODRenderSettings getZoomBasedLODDetailRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply zoom-based LOD to cell rendering that specifically focuses on reducing detail when zoomed out
    // and increasing detail when zoomed in with optimized performance
    void applyZoomBasedLODDetailToCell(const FootprintCell& cell,
                                  ImDrawList* draw_list,
                                  float zoom_factor,
                                  double max_volume,
                                  const std::vector<FootprintCell>& diagonal_imbalances,
                                  const std::vector<FootprintCell>& stacked_imbalances,
                                  const FootprintPanel* panel) const;

    // Calculate zoom-based LOD that specifically focuses on reducing detail when zoomed out
    // and increasing detail when zoomed in with optimized performance and smooth transitions
    LODLevel calculateZoomBasedLOD(float cell_width_px, float cell_height_px,
                                 float zoom_factor) const;

    // Get zoom-based LOD render settings for reducing detail when zoomed out and increasing detail when zoomed in
    LODRenderSettings getZoomBasedLODRenderSettings(LODLevel lod_level, float zoom_factor) const;

    // Apply zoom-based LOD to cell rendering that reduces detail when zoomed out and increases detail when zoomed in
    void applyZoomBasedLODToCell(const FootprintCell& cell,
                                ImDrawList* draw_list,
                                float zoom_factor,
                                double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                const FootprintPanel* panel) const;

    // Get enhanced smooth transition render settings for transitioning between LOD levels
    LODRenderSettings getEnhancedSmoothTransitionRenderSettings(LODLevel from_lod, LODLevel to_lod,
                                                              float transition_progress, float zoom_factor) const;

    // Calculate performance-aware LOD that dynamically adjusts based on real-time performance metrics
    LODLevel calculatePerformanceAwareLOD(float cell_width_px, float cell_height_px,
                                       float zoom_factor, const PerformanceMetrics& metrics) const;

    // Get performance-aware render settings
    LODRenderSettings getPerformanceAwareRenderSettings(LODLevel lod_level,
                                                     const PerformanceMetrics& metrics) const;

    // Apply performance-aware LOD to cell rendering
    void applyPerformanceAwareLODToCell(const FootprintCell& cell,
                                      ImDrawList* draw_list,
                                      float zoom_factor,
                                      const PerformanceMetrics& metrics,
                                      double max_volume,
                                      const std::vector<FootprintCell>& diagonal_imbalances,
                                      const std::vector<FootprintCell>& stacked_imbalances,
                                      const FootprintPanel* panel) const;

    // Calculate density-adaptive LOD that adjusts based on the number of cells in the viewport
    LODLevel calculateDensityAdaptiveLOD(float cell_width_px, float cell_height_px,
                                      float zoom_factor, int total_cells_in_view) const;

    // Get density-adaptive render settings
    LODRenderSettings getDensityAdaptiveRenderSettings(LODLevel lod_level, int total_cells_in_view) const;

    // Apply density-adaptive LOD to cell rendering
    void applyDensityAdaptiveLODToCell(const FootprintCell& cell,
                                     ImDrawList* draw_list,
                                     float zoom_factor,
                                     int total_cells_in_view,
                                     double max_volume,
                                     const std::vector<FootprintCell>& diagonal_imbalances,
                                     const std::vector<FootprintCell>& stacked_imbalances,
                                     const FootprintPanel* panel) const;

    // Calculate hybrid zoom-density LOD combining both zoom level and cell density considerations
    LODLevel calculateHybridZoomDensityLOD(float cell_width_px, float cell_height_px,
                                        float zoom_factor, int total_cells_in_view) const;

    // Get hybrid zoom-density render settings
    LODRenderSettings getHybridZoomDensityRenderSettings(LODLevel lod_level,
                                                      float zoom_factor,
                                                      int total_cells_in_view) const;

    // Apply hybrid zoom-density LOD to cell rendering
    void applyHybridZoomDensityLODToCell(const FootprintCell& cell,
                                       ImDrawList* draw_list,
                                       float zoom_factor,
                                       int total_cells_in_view,
                                       double max_volume,
                                       const std::vector<FootprintCell>& diagonal_imbalances,
                                       const std::vector<FootprintCell>& stacked_imbalances,
                                       const FootprintPanel* panel) const;

    // Getter/setter methods for LOD parameters
    void setMinDetailZoom(float zoom) { min_detail_zoom_ = zoom; }
    void setMediumDetailZoom(float zoom) { medium_detail_zoom_ = zoom; }
    void setMaxDetailZoom(float zoom) { max_detail_zoom_ = zoom; }

    void setMinCellSizePx(float size) { min_cell_size_px_ = size; }
    void setMediumCellSizePx(float size) { medium_cell_size_px_ = size; }
    void setMaxCellSizePx(float size) { max_cell_size_px_ = size; }

    void setTextRenderThreshold(float threshold) { text_render_threshold_ = threshold; }
    void setLabelRenderThreshold(float threshold) { label_render_threshold_ = threshold; }
    void setDetailRenderThreshold(float threshold) { detail_render_threshold_ = threshold; }

    void setDistanceLODThreshold(float threshold) { distance_lod_threshold_ = threshold; }
    void setPerformanceTargetFPS(float fps) { target_fps_ = fps; }

    float getMinDetailZoom() const { return min_detail_zoom_; }
    float getMediumDetailZoom() const { return medium_detail_zoom_; }
    float getMaxDetailZoom() const { return max_detail_zoom_; }

    float getMinCellSizePx() const { return min_cell_size_px_; }
    float getMediumCellSizePx() const { return medium_cell_size_px_; }
    float getMaxCellSizePx() const { return max_cell_size_px_; }

    float getTextRenderThreshold() const { return text_render_threshold_; }
    float getLabelRenderThreshold() const { return label_render_threshold_; }
    float getDetailRenderThreshold() const { return detail_render_threshold_; }

    float getDistanceLODThreshold() const { return distance_lod_threshold_; }
    float getPerformanceTargetFPS() const { return target_fps_; }

private:
    // Zoom thresholds for different LOD levels
    float min_detail_zoom_;      // Zoom factor below which low detail is used
    float medium_detail_zoom_;   // Zoom factor below which medium detail is used
    float max_detail_zoom_;      // Zoom factor above which max detail is used

    // Minimum cell size thresholds for different LOD levels (in pixels)
    float min_cell_size_px_;
    float medium_cell_size_px_;
    float max_cell_size_px_;

    // Thresholds for specific rendering elements
    float text_render_threshold_;     // Minimum cell height to render text
    float label_render_threshold_;    // Minimum cell height to render labels
    float detail_render_threshold_;   // Minimum cell size to render detailed annotations

    // Distance-based LOD parameters
    float distance_lod_threshold_;    // Distance from view center where LOD changes

    // Performance-based LOD parameters
    float target_fps_;                // Target FPS for performance-based LOD
    bool performance_lod_enabled_;    // Whether performance-based LOD is enabled

    // Caching for LOD calculations to improve performance
    mutable std::unordered_map<uint64_t, LODLevel> lod_cache_;

    // Helper function to create a unique key for caching
    uint64_t createLODKey(float cell_width_px, float cell_height_px, float zoom_factor) const;
};

} // namespace Rendering
} // namespace BTQuant