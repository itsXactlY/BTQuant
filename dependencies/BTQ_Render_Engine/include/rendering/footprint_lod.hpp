#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>

#include "imgui.h"
#include "implot.h"

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