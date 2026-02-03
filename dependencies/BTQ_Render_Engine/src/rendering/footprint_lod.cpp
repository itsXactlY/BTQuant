#include "rendering/footprint_lod.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

// Include the footprint panel header to get the full definitions
#include "../../include/components/footprint_panel.hpp"

namespace BTQuant {
namespace Rendering {

FootprintLOD::FootprintLOD()
    : min_detail_zoom_(0.1f)
    , medium_detail_zoom_(1.0f)
    , max_detail_zoom_(3.0f)
    , min_cell_size_px_(4.0f)
    , medium_cell_size_px_(12.0f)
    , max_cell_size_px_(24.0f)
    , text_render_threshold_(12.0f)
    , label_render_threshold_(20.0f)
    , detail_render_threshold_(8.0f)
    , distance_lod_threshold_(100.0f)  // 100 pixels from view center
    , target_fps_(60.0f)               // Target 60 FPS
    , performance_lod_enabled_(true) {} // Performance-based LOD enabled by default


float FootprintLOD::calculateProgressiveLOD(float cell_width_px, float cell_height_px,
                                          float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate continuous LOD value instead of discrete levels
    // This enables smooth transitions between LOD levels
    float lod_value = 0.0f;

    if (zoom_factor <= min_detail_zoom_ || min_dimension <= min_cell_size_px_) {
        // Between LOW_DETAIL and MEDIUM_DETAIL
        float zoom_ratio = zoom_factor / min_detail_zoom_;
        float size_ratio = min_dimension / min_cell_size_px_;
        float min_ratio = std::min(zoom_ratio, size_ratio);

        // Interpolate between 0.0 (LOW) and 1.0 (MEDIUM)
        lod_value = std::clamp(min_ratio, 0.0f, 1.0f);
    } else if (zoom_factor <= medium_detail_zoom_ || min_dimension <= medium_cell_size_px_) {
        // Between MEDIUM_DETAIL and HIGH_DETAIL
        float zoom_ratio = (zoom_factor - min_detail_zoom_) / (medium_detail_zoom_ - min_detail_zoom_);
        float size_ratio = (min_dimension - min_cell_size_px_) / (medium_cell_size_px_ - min_cell_size_px_);
        float effective_ratio = std::min(zoom_ratio, size_ratio);

        // Interpolate between 1.0 (MEDIUM) and 2.0 (HIGH)
        lod_value = 1.0f + std::clamp(effective_ratio, 0.0f, 1.0f);
    } else if (zoom_factor <= max_detail_zoom_ || min_dimension <= max_cell_size_px_) {
        // Between HIGH_DETAIL and MAX_DETAIL
        float zoom_ratio = (zoom_factor - medium_detail_zoom_) / (max_detail_zoom_ - medium_detail_zoom_);
        float size_ratio = (min_dimension - medium_cell_size_px_) / (max_cell_size_px_ - medium_cell_size_px_);
        float effective_ratio = std::min(zoom_ratio, size_ratio);

        // Interpolate between 2.0 (HIGH) and 3.0 (MAX)
        lod_value = 2.0f + std::clamp(effective_ratio, 0.0f, 1.0f);
    } else {
        // Beyond MAX_DETAIL
        lod_value = 3.0f + (zoom_factor - max_detail_zoom_) * 0.5f; // Additional detail scaling
    }

    return lod_value;
}

LODRenderSettings FootprintLOD::getProgressiveRenderSettings(float lod_value, float cell_area_px) const {
    LODRenderSettings settings;

    // Interpolate between LOD levels based on the continuous lod_value
    int base_level = static_cast<int>(std::floor(lod_value));
    float fraction = lod_value - base_level;

    // Get settings for the base level
    LODLevel base_lod = static_cast<LODLevel>(std::clamp(base_level, 0, 3));
    LODLevel next_lod = static_cast<LODLevel>(std::clamp(base_level + 1, 0, 3));

    LODRenderSettings base_settings = getRenderSettings(base_lod);
    LODRenderSettings next_settings = getRenderSettings(next_lod);

    // Interpolate between base and next level settings
    settings.render_heatmap = base_settings.render_heatmap || next_settings.render_heatmap;
    settings.render_borders = (base_settings.render_borders && !fraction) || (next_settings.render_borders && fraction > 0.1f);
    settings.render_text = (base_settings.render_text && !fraction) || (next_settings.render_text && fraction > 0.5f);
    settings.render_labels = (base_settings.render_labels && !fraction) || (next_settings.render_labels && fraction > 0.7f);
    settings.render_detailed_annotations = (base_settings.render_detailed_annotations && !fraction) || (next_settings.render_detailed_annotations && fraction > 0.8f);

    // Interpolate numeric values
    settings.alpha_multiplier = base_settings.alpha_multiplier * (1.0f - fraction) + next_settings.alpha_multiplier * fraction;
    settings.border_thickness = base_settings.border_thickness * (1.0f - fraction) + next_settings.border_thickness * fraction;

    // Apply area-based optimizations
    if (cell_area_px < 16.0f) {  // Less than 4x4 pixels
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
    } else if (cell_area_px < 64.0f) {  // Less than 8x8 pixels
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
    } else if (cell_area_px < 144.0f) {  // Less than 12x12 pixels
        settings.render_detailed_annotations = false;
    }

    return settings;
}

LODLevel FootprintLOD::calculateDynamicLODLevel(float cell_width_px, float cell_height_px,
                                              float zoom_factor, float view_range_x, float view_range_y) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate dynamic thresholds based on view range to provide better LOD scaling
    float dynamic_min_threshold = min_cell_size_px_ * (view_range_x * view_range_y > 100.0f ? 0.5f : 1.0f);
    float dynamic_medium_threshold = medium_cell_size_px_ * (view_range_x * view_range_y > 100.0f ? 0.7f : 1.0f);
    float dynamic_max_threshold = max_cell_size_px_ * (view_range_x * view_range_y > 100.0f ? 0.8f : 1.0f);

    // Calculate dynamic zoom thresholds based on view range
    float dynamic_min_zoom = min_detail_zoom_ * (view_range_x * view_range_y > 100.0f ? 1.2f : 1.0f);
    float dynamic_medium_zoom = medium_detail_zoom_ * (view_range_x * view_range_y > 100.0f ? 1.1f : 1.0f);

    if (zoom_factor <= dynamic_min_zoom || min_dimension <= dynamic_min_threshold) {
        return LODLevel::LOW_DETAIL;  // Minimal detail, heatmap only
    } else if (zoom_factor <= dynamic_medium_zoom || min_dimension <= dynamic_medium_threshold) {
        return LODLevel::MEDIUM_DETAIL;  // Basic detail with some labels
    } else if (zoom_factor <= max_detail_zoom_ || min_dimension <= dynamic_max_threshold) {
        return LODLevel::HIGH_DETAIL;  // Full detail with all information
    } else {
        return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
    }
}


LODLevel FootprintLOD::calculateDistanceBasedLODLevel(float cell_width_px, float cell_height_px,
                                                   float zoom_factor, ImVec2 cell_center, ImVec2 view_center) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate distance from cell to view center
    float distance = std::sqrt((cell_center.x - view_center.x) * (cell_center.x - view_center.x) +
                              (cell_center.y - view_center.y) * (cell_center.y - view_center.y));

    // Adjust LOD based on distance - cells closer to center get more detail
    LODLevel base_lod = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);

    // If cell is close to center, potentially increase detail level
    if (distance < distance_lod_threshold_ * 0.5f) {
        // Very close to center - potentially upgrade LOD level
        switch (base_lod) {
            case LODLevel::LOW_DETAIL:
                if (zoom_factor > min_detail_zoom_ * 0.7f) {
                    return LODLevel::MEDIUM_DETAIL;
                }
                break;
            case LODLevel::MEDIUM_DETAIL:
                if (zoom_factor > medium_detail_zoom_ * 0.8f) {
                    return LODLevel::HIGH_DETAIL;
                }
                break;
            default:
                // HIGH_DETAIL and MAX_DETAIL remain unchanged
                break;
        }
    } else if (distance > distance_lod_threshold_ * 1.5f) {
        // Far from center - potentially decrease detail level to improve performance
        switch (base_lod) {
            case LODLevel::MAX_DETAIL:
                return LODLevel::HIGH_DETAIL;
            case LODLevel::HIGH_DETAIL:
                return LODLevel::MEDIUM_DETAIL;
            case LODLevel::MEDIUM_DETAIL:
                if (zoom_factor < medium_detail_zoom_ * 0.7f) {
                    return LODLevel::LOW_DETAIL;
                }
                break;
            default:
                // LOW_DETAIL remains unchanged
                break;
        }
    }

    return base_lod;
}

LODLevel FootprintLOD::calculateHierarchicalLOD(float cell_width_px, float cell_height_px,
                                               float zoom_factor, int hierarchy_level) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Adjust thresholds based on hierarchy level
    float hierarchy_adjustment = 1.0f + (hierarchy_level * 0.2f); // More detail at higher hierarchy levels

    if (zoom_factor <= min_detail_zoom_ * hierarchy_adjustment || min_dimension <= min_cell_size_px_ * hierarchy_adjustment) {
        return LODLevel::LOW_DETAIL;  // Minimal detail, heatmap only
    } else if (zoom_factor <= medium_detail_zoom_ * hierarchy_adjustment || min_dimension <= medium_cell_size_px_ * hierarchy_adjustment) {
        return LODLevel::MEDIUM_DETAIL;  // Basic detail with some labels
    } else if (zoom_factor <= max_detail_zoom_ || min_dimension <= max_cell_size_px_) {
        return LODLevel::HIGH_DETAIL;  // Full detail with all information
    } else {
        return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
    }
}

LODRenderSettings FootprintLOD::getRenderSettings(LODLevel lod_level) const {
    LODRenderSettings settings;

    switch (lod_level) {
        case LODLevel::LOW_DETAIL:
            settings.render_heatmap = true;
            settings.render_borders = false;
            settings.render_text = false;
            settings.render_labels = false;
            settings.render_detailed_annotations = false;
            settings.alpha_multiplier = 1.0f;
            settings.border_thickness = 0.0f;
            break;

        case LODLevel::MEDIUM_DETAIL:
            settings.render_heatmap = true;
            settings.render_borders = true;
            settings.render_text = false;
            settings.render_labels = false;
            settings.render_detailed_annotations = false;
            settings.alpha_multiplier = 0.9f;
            settings.border_thickness = 1.0f;
            break;

        case LODLevel::HIGH_DETAIL:
            settings.render_heatmap = true;
            settings.render_borders = true;
            settings.render_text = true;
            settings.render_labels = true;
            settings.render_detailed_annotations = false;
            settings.alpha_multiplier = 1.0f;
            settings.border_thickness = 1.5f;
            break;

        case LODLevel::MAX_DETAIL:
            settings.render_heatmap = true;
            settings.render_borders = true;
            settings.render_text = true;
            settings.render_labels = true;
            settings.render_detailed_annotations = true;
            settings.alpha_multiplier = 1.0f;
            settings.border_thickness = 2.0f;
            break;
    }

    return settings;
}

LODRenderSettings FootprintLOD::getOptimizedRenderSettings(LODLevel lod_level, float cell_area_px) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Further optimize settings based on cell area to prevent overcrowding
    if (cell_area_px < 16.0f) {  // Less than 4x4 pixels
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
    } else if (cell_area_px < 64.0f) {  // Less than 8x8 pixels
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
    } else if (cell_area_px < 144.0f) {  // Less than 12x12 pixels
        settings.render_detailed_annotations = false;
    }

    return settings;
}

void FootprintLOD::updatePerformanceBasedLOD(const PerformanceMetrics& metrics) {
    if (!performance_lod_enabled_) {
        return;
    }

    // Adjust LOD based on performance metrics
    if (metrics.performance_degraded || (metrics.fps > 0 && metrics.fps < target_fps_ * 0.8f)) {
        // Performance is degraded, reduce detail level thresholds to improve performance
        min_detail_zoom_ = std::min(min_detail_zoom_ * 1.1f, 0.5f);  // Increase low detail threshold
        medium_detail_zoom_ = std::min(medium_detail_zoom_ * 1.1f, 2.0f);  // Increase medium detail threshold
        text_render_threshold_ = std::max(text_render_threshold_ * 1.1f, 15.0f);  // Require larger cells for text
        label_render_threshold_ = std::max(label_render_threshold_ * 1.1f, 25.0f);  // Require larger cells for labels
    } else if (metrics.fps > 0 && metrics.fps > target_fps_ * 1.2f) {
        // Performance is good, can afford to increase detail
        min_detail_zoom_ = std::max(min_detail_zoom_ * 0.95f, 0.05f);  // Decrease low detail threshold
        medium_detail_zoom_ = std::max(medium_detail_zoom_ * 0.95f, 0.8f);  // Decrease medium detail threshold
        text_render_threshold_ = std::max(text_render_threshold_ * 0.95f, 8.0f);  // Allow smaller cells for text
        label_render_threshold_ = std::max(label_render_threshold_ * 0.95f, 15.0f);  // Allow smaller cells for labels
    }

    // Constrain values to reasonable ranges
    min_detail_zoom_ = std::clamp(min_detail_zoom_, 0.05f, 0.5f);
    medium_detail_zoom_ = std::clamp(medium_detail_zoom_, 0.5f, 2.0f);
    max_detail_zoom_ = std::clamp(max_detail_zoom_, 2.0f, 5.0f);
    text_render_threshold_ = std::clamp(text_render_threshold_, 8.0f, 20.0f);
    label_render_threshold_ = std::clamp(label_render_threshold_, 12.0f, 30.0f);
}

void FootprintLOD::updateAdaptiveLOD(const PerformanceMetrics& metrics, int total_cells_in_view) {
    // Adaptive LOD based on both performance and number of cells in view
    float cell_density_factor = static_cast<float>(total_cells_in_view) / 1000.0f; // Normalize to 1000 cells

    // Adjust thresholds based on cell density
    if (cell_density_factor > 2.0f) {  // Very dense view
        min_detail_zoom_ = std::min(min_detail_zoom_ * 1.2f, 0.6f);
        medium_detail_zoom_ = std::min(medium_detail_zoom_ * 1.2f, 2.2f);
        text_render_threshold_ = std::max(text_render_threshold_ * 1.2f, 18.0f);
        label_render_threshold_ = std::max(label_render_threshold_ * 1.2f, 28.0f);
    } else if (cell_density_factor > 1.0f) {  // Dense view
        min_detail_zoom_ = std::min(min_detail_zoom_ * 1.1f, 0.5f);
        medium_detail_zoom_ = std::min(medium_detail_zoom_ * 1.1f, 2.0f);
        text_render_threshold_ = std::max(text_render_threshold_ * 1.1f, 15.0f);
        label_render_threshold_ = std::max(label_render_threshold_ * 1.1f, 25.0f);
    } else if (cell_density_factor < 0.5f) {  // Sparse view
        min_detail_zoom_ = std::max(min_detail_zoom_ * 0.9f, 0.05f);
        medium_detail_zoom_ = std::max(medium_detail_zoom_ * 0.9f, 0.8f);
        text_render_threshold_ = std::max(text_render_threshold_ * 0.9f, 8.0f);
        label_render_threshold_ = std::max(label_render_threshold_ * 0.9f, 15.0f);
    }

    // Apply performance-based adjustments on top of density-based adjustments
    updatePerformanceBasedLOD(metrics);
}

bool FootprintLOD::shouldRenderText(float cell_height_px, float zoom_factor) const {
    // Skip text rendering when cell height < 12px OR when zoomed out significantly
    return (cell_height_px >= text_render_threshold_ && zoom_factor >= min_detail_zoom_);
}

std::vector<FootprintCell> FootprintLOD::clusterCells(const std::vector<FootprintCell>& cells,
                                                    float zoom_factor) const {
    std::vector<FootprintCell> clustered_cells;

    // Only cluster when zoomed out significantly
    if (zoom_factor > medium_detail_zoom_ * 0.5f) {
        // At higher zoom levels, return original cells without clustering
        return cells;
    }

    // Define clustering distance based on zoom level
    float cluster_distance = 2.0f / zoom_factor;  // Larger distance when zoomed out

    // Create a copy of the input cells
    std::vector<FootprintCell> working_cells = cells;

    // Simple clustering algorithm: merge nearby cells
    for (size_t i = 0; i < working_cells.size(); ++i) {
        const FootprintCell& current_cell = working_cells[i];

        // Skip if this cell has already been processed (marked by setting width to 0)
        if (current_cell.width <= 0) continue;

        // Find nearby cells to cluster with
        FootprintCell clustered_cell = current_cell;
        int cluster_count = 1;

        for (size_t j = i + 1; j < working_cells.size(); ++j) {
            const FootprintCell& other_cell = working_cells[j];

            // Skip if this cell has already been processed
            if (other_cell.width <= 0) continue;

            // Calculate distance between cells
            float dx = std::abs(current_cell.x - other_cell.x);
            float dy = std::abs(current_cell.y - other_cell.y);

            // If cells are close enough, cluster them
            if (dx < cluster_distance && dy < cluster_distance) {
                // Merge the cells by combining their volumes and positions
                clustered_cell.bid_volume += other_cell.bid_volume;
                clustered_cell.ask_volume += other_cell.ask_volume;
                clustered_cell.trade_count += other_cell.trade_count;

                // Average the position
                clustered_cell.x = (clustered_cell.x * cluster_count + other_cell.x) / (cluster_count + 1);
                clustered_cell.y = (clustered_cell.y * cluster_count + other_cell.y) / (cluster_count + 1);

                // Update dimensions to encompass both cells
                clustered_cell.width = std::max(clustered_cell.width, std::abs(other_cell.x - clustered_cell.x) * 2.0);
                clustered_cell.height = std::max(clustered_cell.height, std::abs(other_cell.y - clustered_cell.y) * 2.0);

                // Mark this cell as processed
                working_cells[j].width = 0; // Mark as processed by setting width to 0
                cluster_count++;
            }
        }

        // Add the clustered cell to the result
        clustered_cells.push_back(clustered_cell);
    }

    return clustered_cells;
}

std::vector<FootprintCell> FootprintLOD::adaptiveClusterCells(const std::vector<FootprintCell>& cells,
                                                                         float zoom_factor,
                                                                         int total_cells_in_view) const {
    std::vector<FootprintCell> clustered_cells;

    // Determine if clustering is needed based on zoom and cell density
    float cell_density = static_cast<float>(total_cells_in_view) / 1000.0f; // Normalize to 1000 cells
    bool should_cluster = (zoom_factor < medium_detail_zoom_ * 0.7f) || (cell_density > 1.5f);

    if (!should_cluster) {
        return cells;
    }

    // Adjust clustering distance based on cell density
    float base_cluster_distance = 2.0f / zoom_factor;
    float density_factor = std::min(cell_density, 3.0f); // Cap at 3x density
    float cluster_distance = base_cluster_distance * density_factor;

    // Create a copy of the input cells
    std::vector<FootprintCell> working_cells = cells;

    // More efficient clustering algorithm using spatial partitioning concept
    for (size_t i = 0; i < working_cells.size(); ++i) {
        const FootprintCell& current_cell = working_cells[i];

        // Skip if this cell has already been processed
        if (current_cell.width <= 0) continue;

        // Find nearby cells to cluster with
        FootprintCell clustered_cell = current_cell;
        double total_weight = 1.0; // Weight based on volume for weighted average
        double weighted_x = current_cell.x * (current_cell.bid_volume + current_cell.ask_volume + 1);
        double weighted_y = current_cell.y * (current_cell.bid_volume + current_cell.ask_volume + 1);

        for (size_t j = i + 1; j < working_cells.size(); ++j) {
            const FootprintCell& other_cell = working_cells[j];

            // Skip if this cell has already been processed
            if (other_cell.width <= 0) continue;

            // Calculate distance between cells
            float dx = std::abs(current_cell.x - other_cell.x);
            float dy = std::abs(current_cell.y - other_cell.y);

            // If cells are close enough, cluster them
            if (dx < cluster_distance && dy < cluster_distance) {
                // Merge the cells by combining their volumes and positions
                clustered_cell.bid_volume += other_cell.bid_volume;
                clustered_cell.ask_volume += other_cell.ask_volume;
                clustered_cell.trade_count += other_cell.trade_count;

                // Weighted average for position based on volume
                double cell_weight = other_cell.bid_volume + other_cell.ask_volume + 1;
                weighted_x += other_cell.x * cell_weight;
                weighted_y += other_cell.y * cell_weight;
                total_weight += cell_weight;

                // Update dimensions to encompass both cells
                clustered_cell.width = std::max(clustered_cell.width, std::abs(other_cell.x - clustered_cell.x) * 2.0);
                clustered_cell.height = std::max(clustered_cell.height, std::abs(other_cell.y - clustered_cell.y) * 2.0);

                // Mark this cell as processed
                working_cells[j].width = 0; // Mark as processed by setting width to 0
            }
        }

        // Calculate final weighted position
        if (total_weight > 1.0) {
            clustered_cell.x = static_cast<float>(weighted_x / total_weight);
            clustered_cell.y = static_cast<float>(weighted_y / total_weight);
        }

        // Add the clustered cell to the result
        clustered_cells.push_back(clustered_cell);
    }

    return clustered_cells;
}

bool FootprintLOD::shouldRenderLabels(float cell_height_px, float zoom_factor) const {
    // Labels require more space, only render when cell is sufficiently large
    return (cell_height_px >= label_render_threshold_ && zoom_factor >= medium_detail_zoom_);
}

bool FootprintLOD::shouldRenderDetailedAnnotations(float cell_width_px, 
                                                float cell_height_px, 
                                                float zoom_factor) const {
    // Detailed annotations (like delta indicators) need sufficient space
    float min_dimension = std::min(cell_width_px, cell_height_px);
    return (min_dimension >= detail_render_threshold_ && zoom_factor >= medium_detail_zoom_);
}

float FootprintLOD::adjustCellPadding(float base_padding, float zoom_factor) const {
    // Adjust cell padding based on zoom level for better visual representation
    if (zoom_factor >= 1.0f) {
        // When zoomed in: reduce padding to show more detail
        float zoom_effect = std::log10(zoom_factor * 1.5f + 1.0f) * 0.2f;
        return std::max(0.05f, base_padding - zoom_effect);
    } else {
        // When zoomed out: increase padding to show more cells, approaching squares
        float zoom_effect = std::pow(1.0f / (zoom_factor * 1.5f), 0.8f) - 1.0f;
        return std::min(0.48f, base_padding + zoom_effect * 0.15f);
    }
}

float FootprintLOD::calculateAdaptiveCellSize(float base_size, float zoom_factor, LODLevel lod_level) const {
    // Calculate adaptive cell size based on zoom and LOD level
    float size_multiplier = 1.0f;

    switch (lod_level) {
        case LODLevel::LOW_DETAIL:
            // At low detail, slightly reduce cell size to allow more cells to be visible
            size_multiplier = 0.8f;
            break;
        case LODLevel::MEDIUM_DETAIL:
            // At medium detail, use standard size
            size_multiplier = 1.0f;
            break;
        case LODLevel::HIGH_DETAIL:
            // At high detail, slightly increase size for better visibility
            size_multiplier = 1.1f;
            break;
        case LODLevel::MAX_DETAIL:
            // At max detail, increase size for maximum clarity
            size_multiplier = 1.2f;
            break;
    }

    // Also adjust based on zoom factor
    if (zoom_factor < 0.5f) {
        size_multiplier *= 0.9f;  // Further reduce at very low zoom
    } else if (zoom_factor > 2.0f) {
        size_multiplier *= 1.1f;  // Slightly increase at high zoom
    }

    return base_size * size_multiplier;
}

LODTransitionState FootprintLOD::calculateLODTransition(float prev_zoom, float curr_zoom,
                                                      float cell_width_px, float cell_height_px) const {
    // Calculate transition state to enable smooth LOD transitions
    LODTransitionState state;

    LODLevel prev_lod_level = calculateLODLevel(cell_width_px, cell_height_px, prev_zoom);
    LODLevel curr_lod_level = calculateLODLevel(cell_width_px, cell_height_px, curr_zoom);

    state.from_lod = prev_lod_level;
    state.to_lod = curr_lod_level;

    // Calculate transition progress (0.0 to 1.0)
    float zoom_diff = std::abs(curr_zoom - prev_zoom);
    state.transition_progress = std::min(1.0f, zoom_diff * 5.0f);  // Scale factor for sensitivity

    return state;
}

float FootprintLOD::calculateAlphaMultiplier(float zoom_factor, LODLevel lod_level) const {
    // Adjust alpha based on zoom level and LOD to maintain visual coherence
    switch (lod_level) {
        case LODLevel::LOW_DETAIL:
            // At low detail, use slightly reduced alpha to indicate lower importance
            return 0.8f;
        case LODLevel::MEDIUM_DETAIL:
            return 0.9f;
        case LODLevel::HIGH_DETAIL:
            return 1.0f;
        case LODLevel::MAX_DETAIL:
            return 1.0f;
        default:
            return 0.8f;
    }
}

LODStatistics FootprintLOD::calculateLODStatistics(const std::vector<FootprintCell>& cells,
                                                 float zoom_factor) const {
    LODStatistics stats = {};
    stats.total_cells = static_cast<int>(cells.size());
    
    for (const auto& cell : cells) {
        // Convert cell coordinates to pixel dimensions (this would typically happen in the render loop)
        // For now, we'll simulate the calculation
        float cell_width_px = cell.width * 10.0f;  // Placeholder conversion
        float cell_height_px = cell.height * 10.0f; // Placeholder conversion
        
        LODLevel lod_level = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
        
        switch (lod_level) {
            case LODLevel::LOW_DETAIL:
                stats.low_detail_cells++;
                break;
            case LODLevel::MEDIUM_DETAIL:
                stats.medium_detail_cells++;
                break;
            case LODLevel::HIGH_DETAIL:
                stats.high_detail_cells++;
                break;
            case LODLevel::MAX_DETAIL:
                stats.max_detail_cells++;
                break;
        }
    }
    
    return stats;
}

void FootprintLOD::applyLODToCell(const FootprintCell& cell,
                                ImDrawList* draw_list,
                                float zoom_factor,
                                double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level
    LODLevel lod_level = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getOptimizedRenderSettings(lod_level, cell_area_px);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Render heatmap/fill based on LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells even at lower LOD levels
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

void FootprintLOD::applyDistanceBasedLODToCell(const FootprintCell& cell,
                                             ImDrawList* draw_list,
                                             float zoom_factor,
                                             double max_volume,
                                             const std::vector<FootprintCell>& diagonal_imbalances,
                                             const std::vector<FootprintCell>& stacked_imbalances,
                                             const FootprintPanel* panel,
                                             ImVec2 view_center) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Calculate cell center for distance-based LOD
    ImVec2 cell_center = {(p1.x + p2.x) * 0.5f, (p1.y + p2.y) * 0.5f};

    // Determine LOD level using distance-based calculation
    LODLevel lod_level = calculateDistanceBasedLODLevel(cell_width_px, cell_height_px, zoom_factor, cell_center, view_center);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getOptimizedRenderSettings(lod_level, cell_area_px);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Render heatmap/fill based on LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells even at lower LOD levels
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

uint64_t FootprintLOD::createLODKey(float cell_width_px, float cell_height_px, float zoom_factor) const {
    // Create a unique key by combining the three float values into a 64-bit integer
    // This is a simple hash-like approach for caching
    union {
        float f;
        uint32_t i;
    } converter_w, converter_h, converter_z;

    converter_w.f = cell_width_px;
    converter_h.f = cell_height_px;
    converter_z.f = zoom_factor;

    // Combine the bits to create a unique key
    uint64_t key = ((uint64_t)converter_w.i << 32) | (uint64_t)converter_h.i;
    // XOR with zoom factor to further differentiate
    key ^= ((uint64_t)converter_z.f * 1000000); // Scale zoom to get more variation

    return key;
}

LODLevel FootprintLOD::calculateLODLevel(float cell_width_px, float cell_height_px,
                                       float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate LOD based on both cell size and zoom level
    // Using a more sophisticated approach that considers both factors
    if (zoom_factor <= min_detail_zoom_ || min_dimension <= min_cell_size_px_) {
        return LODLevel::LOW_DETAIL;  // Minimal detail, heatmap only
    } else if (zoom_factor <= medium_detail_zoom_ || min_dimension <= medium_cell_size_px_) {
        return LODLevel::MEDIUM_DETAIL;  // Basic detail with some labels
    } else if (zoom_factor <= max_detail_zoom_ || min_dimension <= max_cell_size_px_) {
        return LODLevel::HIGH_DETAIL;  // Full detail with all information
    } else {
        return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
    }
}

LODLevel FootprintLOD::calculateCachedLOD(float cell_width_px, float cell_height_px, float zoom_factor) const {
    uint64_t key = createLODKey(cell_width_px, cell_height_px, zoom_factor);

    // Check if we have a cached result
    auto it = lod_cache_.find(key);
    if (it != lod_cache_.end()) {
        return it->second;
    }

    // Calculate the LOD level and cache it
    LODLevel lod_level = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);

    // Only cache if we have room (prevent unlimited memory growth)
    if (lod_cache_.size() < 10000) {  // Limit cache size to prevent memory issues
        lod_cache_[key] = lod_level;
    }

    return lod_level;
}

void FootprintLOD::clearLODCache() const {
    lod_cache_.clear();
}

LODLevel FootprintLOD::calculateZoomBasedLOD(float cell_width_px, float cell_height_px,
                                           float zoom_factor, float view_range_x, float view_range_y) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate view area to determine density
    float view_area = view_range_x * view_range_y;

    // Adjust thresholds based on view density
    float density_factor = std::min(view_area / 1000.0f, 2.0f); // Cap at 2x

    // Calculate base thresholds adjusted for density
    float adjusted_min_threshold = min_cell_size_px_ * density_factor;
    float adjusted_medium_threshold = medium_cell_size_px_ * density_factor;
    float adjusted_max_threshold = max_cell_size_px_ * density_factor;

    // Calculate LOD based on adjusted thresholds
    if (zoom_factor <= min_detail_zoom_ || min_dimension <= adjusted_min_threshold) {
        return LODLevel::LOW_DETAIL;  // Minimal detail, heatmap only
    } else if (zoom_factor <= medium_detail_zoom_ || min_dimension <= adjusted_medium_threshold) {
        return LODLevel::MEDIUM_DETAIL;  // Basic detail with some labels
    } else if (zoom_factor <= max_detail_zoom_ || min_dimension <= adjusted_max_threshold) {
        return LODLevel::HIGH_DETAIL;  // Full detail with all information
    } else {
        return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
    }
}

LODLevel FootprintLOD::calculateAdaptiveLODBasedOnViewArea(float cell_width_px, float cell_height_px,
                                                         float zoom_factor, float view_width, float view_height) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate the view area in pixels to determine how much detail we can afford
    float view_area_px = view_width * view_height;
    int estimated_cells_in_view = static_cast<int>(view_area_px / (min_cell_size_px_ * min_cell_size_px_));

    // Adjust LOD based on how many cells we expect to render
    if (estimated_cells_in_view > 10000) {  // Too many cells to render in detail
        // Force low detail to maintain performance
        if (min_dimension <= min_cell_size_px_ * 0.5f) {
            return LODLevel::LOW_DETAIL;  // Skip rendering entirely at very low resolution
        } else {
            return LODLevel::MEDIUM_DETAIL;  // Minimal rendering
        }
    } else if (estimated_cells_in_view > 5000) {  // Many cells, reduce detail
        if (zoom_factor <= medium_detail_zoom_ * 0.7f || min_dimension <= medium_cell_size_px_ * 0.7f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::HIGH_DETAIL;
        }
    } else {  // Reasonable number of cells, allow more detail
        if (zoom_factor <= min_detail_zoom_ || min_dimension <= min_cell_size_px_) {
            return LODLevel::LOW_DETAIL;
        } else if (zoom_factor <= medium_detail_zoom_ || min_dimension <= medium_cell_size_px_) {
            return LODLevel::MEDIUM_DETAIL;
        } else if (zoom_factor <= max_detail_zoom_ || min_dimension <= max_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else {
            return LODLevel::MAX_DETAIL;
        }
    }
}

bool FootprintLOD::shouldCompletelySkipRendering(float cell_width_px, float cell_height_px,
                                              float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Skip rendering entirely when cells are too small to be meaningful
    // This occurs when zoomed out significantly
    return (zoom_factor <= min_detail_zoom_ * 0.3f && min_dimension <= min_cell_size_px_ * 0.5f);
}

LODRenderSettings FootprintLOD::getSimplifiedRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Further simplify settings when zoomed out significantly
    if (zoom_factor < min_detail_zoom_ * 0.5f) {
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
        settings.render_borders = false;  // Even borders might be too much at extreme zoom out
    } else if (zoom_factor < medium_detail_zoom_ * 0.7f) {
        settings.render_detailed_annotations = false;
        settings.render_labels = false;
    }

    return settings;
}

void FootprintLOD::applySimplifiedLODToCell(const FootprintCell& cell,
                                           ImDrawList* draw_list,
                                           float zoom_factor,
                                           double max_volume,
                                           const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Early exit if cell should not be rendered at all
    if (shouldCompletelySkipRendering(cell_width_px, cell_height_px, zoom_factor)) {
        return;
    }

    // Determine LOD level
    LODLevel lod_level = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    LODRenderSettings settings = getSimplifiedRenderSettings(lod_level, zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Render heatmap/fill based on simplified LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on simplified LOD settings
    if (settings.render_borders) {
        unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
        ImU32 border_color = IM_COL32(255, 255, 255, border_alpha);
        draw_list->AddRect(p1, p2, border_color, 0.0f, 0, settings.border_thickness);
    }
}

LODLevel FootprintLOD::calculateEnhancedDetailLOD(float cell_width_px, float cell_height_px,
                                                float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // When zoomed in significantly, allow for even more detailed representations
    if (zoom_factor > max_detail_zoom_ * 1.5f) {
        // At very high zoom levels, provide maximum detail
        if (min_dimension > max_cell_size_px_ * 2.0f) {
            return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
        } else if (min_dimension > medium_cell_size_px_ * 1.5f) {
            return LODLevel::HIGH_DETAIL;  // Full detail with all information
        } else {
            return LODLevel::MEDIUM_DETAIL;  // Basic detail with some labels
        }
    } else if (zoom_factor > max_detail_zoom_) {
        // High zoom but not extremely high
        if (min_dimension > max_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;  // Full detail with all information
        } else {
            return LODLevel::MEDIUM_DETAIL;  // Basic detail with some labels
        }
    } else {
        // Standard zoom levels - use regular calculation
        return calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    }
}

LODRenderSettings FootprintLOD::getEnhancedRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // When zoomed in significantly, enhance the detail level
    if (zoom_factor > max_detail_zoom_ * 1.2f) {
        // Enable additional features when zoomed in
        settings.render_detailed_annotations = true;
        settings.render_labels = true;

        if (zoom_factor > max_detail_zoom_ * 2.0f) {
            // At extreme zoom in, add even more detail
            settings.alpha_multiplier = 1.1f;  // Slightly more opaque
            settings.border_thickness = 2.5f;  // Thicker borders for better visibility
            settings.render_text = true;       // Ensure text is rendered
        }
    }

    return settings;
}

void FootprintLOD::applyEnhancedLODToCell(const FootprintCell& cell,
                                         ImDrawList* draw_list,
                                         float zoom_factor,
                                         double max_volume,
                                         const std::vector<FootprintCell>& diagonal_imbalances,
                                         const std::vector<FootprintCell>& stacked_imbalances,
                                         const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level with enhanced detail calculation
    LODLevel lod_level = calculateEnhancedDetailLOD(cell_width_px, cell_height_px, zoom_factor);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getEnhancedRenderSettings(lod_level, zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Render heatmap/fill based on enhanced LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on enhanced LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells even more when zoomed in
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on enhanced LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on enhanced LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

void FootprintLOD::applyProgressiveLODToCell(const FootprintCell& cell,
                                           ImDrawList* draw_list,
                                           float zoom_factor,
                                           double max_volume,
                                           const std::vector<FootprintCell>& diagonal_imbalances,
                                           const std::vector<FootprintCell>& stacked_imbalances,
                                           const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Calculate progressive LOD value
    float lod_value = calculateProgressiveLOD(cell_width_px, cell_height_px, zoom_factor);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getProgressiveRenderSettings(lod_value, cell_area_px);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD (using the interpolated value)
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, static_cast<LODLevel>(static_cast<int>(lod_value)));

    // Render heatmap/fill based on progressive LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on progressive LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells even at lower LOD levels
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on progressive LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on progressive LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on progressive LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

LODLevel FootprintLOD::calculateMultiResolutionLOD(float cell_width_px, float cell_height_px,
                                                float zoom_factor, int total_cells_in_view) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate cell density factor based on total cells in view
    float density_factor = static_cast<float>(total_cells_in_view) / 5000.0f; // Normalize to 5000 cells

    // Adjust thresholds based on cell density
    float density_adjusted_min_threshold = min_cell_size_px_ * (1.0f + density_factor * 0.5f);
    float density_adjusted_medium_threshold = medium_cell_size_px_ * (1.0f + density_factor * 0.3f);
    float density_adjusted_max_threshold = max_cell_size_px_ * (1.0f + density_factor * 0.1f);

    // Calculate zoom-based adjustment
    float zoom_factor_adjustment = 1.0f;
    if (zoom_factor < 0.3f) {
        zoom_factor_adjustment = 0.7f; // Reduce detail when heavily zoomed out
    } else if (zoom_factor > 2.0f) {
        zoom_factor_adjustment = 1.2f; // Allow more detail when zoomed in
    }

    // Apply adjustments to thresholds
    density_adjusted_min_threshold *= zoom_factor_adjustment;
    density_adjusted_medium_threshold *= zoom_factor_adjustment;
    density_adjusted_max_threshold *= zoom_factor_adjustment;

    // Determine LOD level based on adjusted thresholds
    if (zoom_factor <= min_detail_zoom_ * 0.8f || min_dimension <= density_adjusted_min_threshold) {
        return LODLevel::LOW_DETAIL;  // Minimal detail, heatmap only
    } else if (zoom_factor <= medium_detail_zoom_ * 0.9f || min_dimension <= density_adjusted_medium_threshold) {
        return LODLevel::MEDIUM_DETAIL;  // Basic detail with some labels
    } else if (zoom_factor <= max_detail_zoom_ || min_dimension <= density_adjusted_max_threshold) {
        return LODLevel::HIGH_DETAIL;  // Full detail with all information
    } else {
        return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
    }
}

LODRenderSettings FootprintLOD::getMultiResolutionRenderSettings(LODLevel lod_level, int total_cells_in_view) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Adjust settings based on cell density
    float density_factor = static_cast<float>(total_cells_in_view) / 5000.0f; // Normalize to 5000 cells

    if (density_factor > 1.5f) {  // Very high density
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier *= 0.9f; // Slightly reduce alpha to avoid visual clutter
    } else if (density_factor > 1.0f) {  // High density
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier *= 0.95f;
    } else if (density_factor < 0.3f) {  // Low density
        settings.render_detailed_annotations = true; // Show more detail when not crowded
        settings.alpha_multiplier = 1.0f;
    }

    return settings;
}

void FootprintLOD::applyMultiResolutionLODToCell(const FootprintCell& cell,
                                               ImDrawList* draw_list,
                                               float zoom_factor,
                                               double max_volume,
                                               const std::vector<FootprintCell>& diagonal_imbalances,
                                               const std::vector<FootprintCell>& stacked_imbalances,
                                               const FootprintPanel* panel,
                                               int total_cells_in_view) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using multi-resolution calculation
    LODLevel lod_level = calculateMultiResolutionLOD(cell_width_px, cell_height_px, zoom_factor, total_cells_in_view);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getMultiResolutionRenderSettings(lod_level, total_cells_in_view);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply density-based alpha adjustment
    float density_factor = static_cast<float>(total_cells_in_view) / 5000.0f;
    if (density_factor > 1.0f) {
        alpha_multiplier *= (1.0f - (density_factor - 1.0f) * 0.2f); // Reduce alpha in high density
    }

    // Render heatmap/fill based on multi-resolution LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and density
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on multi-resolution LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells even at lower LOD levels
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on multi-resolution LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on multi-resolution LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

LODLevel FootprintLOD::calculatePredictiveLOD(float cell_width_px, float cell_height_px,
                                          float current_zoom, float predicted_zoom) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate the expected cell dimensions at the predicted zoom level
    float expected_cell_width = cell_width_px * (predicted_zoom / current_zoom);
    float expected_cell_height = cell_height_px * (predicted_zoom / current_zoom);
    float expected_min_dimension = std::min(expected_cell_width, expected_cell_height);

    // Use the expected dimensions to determine LOD level
    if (predicted_zoom <= min_detail_zoom_ || expected_min_dimension <= min_cell_size_px_) {
        return LODLevel::LOW_DETAIL;  // Minimal detail, heatmap only
    } else if (predicted_zoom <= medium_detail_zoom_ || expected_min_dimension <= medium_cell_size_px_) {
        return LODLevel::MEDIUM_DETAIL;  // Basic detail with some labels
    } else if (predicted_zoom <= max_detail_zoom_ || expected_min_dimension <= max_cell_size_px_) {
        return LODLevel::HIGH_DETAIL;  // Full detail with all information
    } else {
        return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
    }
}

LODLevel FootprintLOD::calculateHybridLOD(float cell_width_px, float cell_height_px,
                                      float zoom_factor, int total_cells_in_view,
                                      const PerformanceMetrics& metrics) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate LOD based on multiple factors
    LODLevel zoom_based_lod = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    LODLevel density_based_lod = calculateMultiResolutionLOD(cell_width_px, cell_height_px, zoom_factor, total_cells_in_view);
    LODLevel performance_based_lod = zoom_based_lod; // Default to zoom-based

    // Adjust performance-based LOD based on metrics
    if (metrics.performance_degraded || (metrics.fps > 0 && metrics.fps < target_fps_ * 0.8f)) {
        // Performance is degraded, reduce detail
        switch (zoom_based_lod) {
            case LODLevel::MAX_DETAIL:
                performance_based_lod = LODLevel::HIGH_DETAIL;
                break;
            case LODLevel::HIGH_DETAIL:
                performance_based_lod = LODLevel::MEDIUM_DETAIL;
                break;
            case LODLevel::MEDIUM_DETAIL:
                performance_based_lod = LODLevel::LOW_DETAIL;
                break;
            default:
                performance_based_lod = LODLevel::LOW_DETAIL;
                break;
        }
    } else {
        performance_based_lod = zoom_based_lod;
    }

    // Combine the different LOD levels by taking the most conservative (lowest detail) approach
    // to ensure performance and visual clarity
    LODLevel hybrid_lod = zoom_based_lod;

    // Choose the LOD level that represents the lowest detail among all approaches
    if (static_cast<int>(density_based_lod) < static_cast<int>(hybrid_lod)) {
        hybrid_lod = density_based_lod;
    }
    if (static_cast<int>(performance_based_lod) < static_cast<int>(hybrid_lod)) {
        hybrid_lod = performance_based_lod;
    }

    return hybrid_lod;
}

// Advanced LOD methods for improved performance and visual quality

LODLevel FootprintLOD::calculateAdaptiveTemporalLOD(float cell_width_px, float cell_height_px,
                                                  float zoom_factor, float time_since_update) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate base LOD level
    LODLevel base_lod = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);

    // Adjust LOD based on how recently the cell was updated
    // Cells that haven't changed recently can be rendered with lower detail
    if (time_since_update > 5.0f) {  // More than 5 seconds since last update
        // Reduce detail for stale cells
        switch (base_lod) {
            case LODLevel::MAX_DETAIL:
                return LODLevel::HIGH_DETAIL;
            case LODLevel::HIGH_DETAIL:
                return LODLevel::MEDIUM_DETAIL;
            case LODLevel::MEDIUM_DETAIL:
                return LODLevel::LOW_DETAIL;
            default:
                return base_lod;
        }
    } else if (time_since_update < 0.5f) {  // Updated in the last 0.5 seconds
        // Increase detail for actively changing cells
        switch (base_lod) {
            case LODLevel::LOW_DETAIL:
                return LODLevel::MEDIUM_DETAIL;
            case LODLevel::MEDIUM_DETAIL:
                return LODLevel::HIGH_DETAIL;
            default:
                return base_lod;
        }
    }

    return base_lod;
}

LODRenderSettings FootprintLOD::getTemporalLODRenderSettings(LODLevel lod_level, float time_since_update) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Adjust settings based on how recently the cell was updated
    if (time_since_update > 3.0f) {  // Older data
        settings.alpha_multiplier *= 0.7f;  // Make older data more transparent
        settings.render_detailed_annotations = false;  // Skip detailed annotations for old data
    } else if (time_since_update < 1.0f) {  // Recent data
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 1.2f, 1.0f);  // Make recent data more prominent
    }

    return settings;
}

LODLevel FootprintLOD::calculateContextualLOD(float cell_width_px, float cell_height_px,
                                            float zoom_factor, const std::vector<FootprintCell>& nearby_cells) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate base LOD level
    LODLevel base_lod = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);

    // Analyze nearby cells to determine if this cell is in an area of interest
    int high_activity_neighbors = 0;
    int total_neighbors = static_cast<int>(nearby_cells.size());

    // Count neighbors with high activity (large volume differences)
    for (const auto& neighbor : nearby_cells) {
        double volume_sum = neighbor.bid_volume + neighbor.ask_volume;
        double volume_diff = std::abs(neighbor.bid_volume - neighbor.ask_volume);

        if (volume_sum > 0 && (volume_diff / volume_sum) > 0.3) {  // High imbalance
            high_activity_neighbors++;
        }
    }

    // Adjust LOD based on contextual importance
    if (total_neighbors > 0 && (static_cast<float>(high_activity_neighbors) / total_neighbors) > 0.5f) {
        // This cell is in an area of high activity, increase detail
        switch (base_lod) {
            case LODLevel::LOW_DETAIL:
                return LODLevel::MEDIUM_DETAIL;
            case LODLevel::MEDIUM_DETAIL:
                return LODLevel::HIGH_DETAIL;
            default:
                return base_lod;
        }
    } else if (high_activity_neighbors == 0) {
        // This cell is in a quiet area, reduce detail
        switch (base_lod) {
            case LODLevel::MAX_DETAIL:
                return LODLevel::HIGH_DETAIL;
            case LODLevel::HIGH_DETAIL:
                return LODLevel::MEDIUM_DETAIL;
            case LODLevel::MEDIUM_DETAIL:
                return LODLevel::LOW_DETAIL;
            default:
                return base_lod;
        }
    }

    return base_lod;
}

void FootprintLOD::applyContextualLODToCell(const FootprintCell& cell,
                                          ImDrawList* draw_list,
                                          float zoom_factor,
                                          double max_volume,
                                          const std::vector<FootprintCell>& diagonal_imbalances,
                                          const std::vector<FootprintCell>& stacked_imbalances,
                                          const FootprintPanel* panel,
                                          const std::vector<FootprintCell>& nearby_cells) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using contextual calculation
    LODLevel lod_level = calculateContextualLOD(cell_width_px, cell_height_px, zoom_factor, nearby_cells);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getOptimizedRenderSettings(lod_level, cell_area_px);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Render heatmap/fill based on contextual LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on contextual LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells even at lower LOD levels
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on contextual LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on contextual LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

LODLevel FootprintLOD::calculateFoveatedLOD(float cell_width_px, float cell_height_px,
                                          float zoom_factor, ImVec2 cell_center, ImVec2 focus_point,
                                          float focus_radius_inner, float focus_radius_outer) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate distance from cell to focus point
    float distance_to_focus = std::sqrt((cell_center.x - focus_point.x) * (cell_center.x - focus_point.x) +
                                       (cell_center.y - focus_point.y) * (cell_center.y - focus_point.y));

    // Determine base LOD level
    LODLevel base_lod = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);

    // Apply foveated rendering: high detail near focus point, decreasing detail further away
    if (distance_to_focus <= focus_radius_inner) {
        // Inside inner radius - maximum detail regardless of zoom
        return LODLevel::MAX_DETAIL;
    } else if (distance_to_focus <= focus_radius_outer) {
        // In transition zone - interpolate between max detail and base LOD
        float t = (distance_to_focus - focus_radius_inner) / (focus_radius_outer - focus_radius_inner);

        // Convert LOD levels to integers for interpolation
        int base_int = static_cast<int>(base_lod);
        int max_int = static_cast<int>(LODLevel::MAX_DETAIL);

        int interpolated_lod = static_cast<int>(max_int + (base_int - max_int) * t);
        return static_cast<LODLevel>(std::max(interpolated_lod, 0)); // Clamp to valid range
    } else {
        // Outside outer radius - use base LOD or potentially reduce further
        return base_lod;
    }
}

void FootprintLOD::applyFoveatedLODToCell(const FootprintCell& cell,
                                         ImDrawList* draw_list,
                                         float zoom_factor,
                                         double max_volume,
                                         const std::vector<FootprintCell>& diagonal_imbalances,
                                         const std::vector<FootprintCell>& stacked_imbalances,
                                         const FootprintPanel* panel,
                                         ImVec2 focus_point,
                                         float focus_radius_inner,
                                         float focus_radius_outer) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Calculate cell center for foveated LOD
    ImVec2 cell_center = {(p1.x + p2.x) * 0.5f, (p1.y + p2.y) * 0.5f};

    // Determine LOD level using foveated calculation
    LODLevel lod_level = calculateFoveatedLOD(cell_width_px, cell_height_px, zoom_factor,
                                            cell_center, focus_point, focus_radius_inner, focus_radius_outer);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getOptimizedRenderSettings(lod_level, cell_area_px);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Render heatmap/fill based on foveated LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on foveated LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells even at lower LOD levels
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on foveated LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on foveated LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

LODLevel FootprintLOD::calculateZoomOutLOD(float cell_width_px, float cell_height_px,
                                         float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // When zoomed out significantly, use more aggressive LOD reduction
    if (zoom_factor < 0.1f) {
        // Extremely zoomed out - only show heatmap for most important cells
        if (min_dimension > min_cell_size_px_ * 0.3f) {
            return LODLevel::LOW_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL; // Skip rendering entirely for tiny cells
        }
    } else if (zoom_factor < 0.3f) {
        // Highly zoomed out - reduce detail significantly
        if (min_dimension > min_cell_size_px_ * 0.5f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL; // Skip rendering for very small cells
        }
    } else if (zoom_factor < 0.7f) {
        // Moderately zoomed out - use medium detail
        if (min_dimension > medium_cell_size_px_ * 0.7f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else {
        // Normal zoom levels - use standard calculation
        return calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    }
}

LODLevel FootprintLOD::calculateZoomInLOD(float cell_width_px, float cell_height_px,
                                        float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // When zoomed in significantly, allow for more detailed representations
    if (zoom_factor > max_detail_zoom_ * 2.0f) {
        // Extremely zoomed in - provide maximum detail
        if (min_dimension > max_cell_size_px_ * 3.0f) {
            return LODLevel::MAX_DETAIL; // Ultra detail with additional annotations
        } else if (min_dimension > max_cell_size_px_ * 1.5f) {
            return LODLevel::HIGH_DETAIL; // Full detail with all information
        } else {
            return LODLevel::HIGH_DETAIL; // Still high detail due to zoom level
        }
    } else if (zoom_factor > max_detail_zoom_ * 1.5f) {
        // Highly zoomed in - provide extra detail
        if (min_dimension > max_cell_size_px_ * 2.0f) {
            return LODLevel::HIGH_DETAIL; // Full detail with all information
        } else if (min_dimension > medium_cell_size_px_ * 1.5f) {
            return LODLevel::HIGH_DETAIL; // Still high detail due to size
        } else {
            return LODLevel::MEDIUM_DETAIL; // Basic detail with some labels
        }
    } else if (zoom_factor > max_detail_zoom_) {
        // Slightly above max zoom - provide high detail
        if (min_dimension > max_cell_size_px_) {
            return LODLevel::HIGH_DETAIL; // Full detail with all information
        } else {
            return LODLevel::MEDIUM_DETAIL; // Basic detail with some labels
        }
    } else {
        // Standard zoom levels - use regular calculation
        return calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    }
}

LODRenderSettings FootprintLOD::getZoomOutOptimizedRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // When zoomed out, further optimize settings to improve performance
    if (zoom_factor < 0.3f) {
        // At high zoom out, disable text and labels to reduce visual clutter
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;

        // Reduce alpha slightly to make visualization clearer when many cells are present
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 0.9f, 1.0f);
    } else if (zoom_factor < 0.7f) {
        // At moderate zoom out, consider disabling detailed annotations
        settings.render_detailed_annotations = false;
    }

    return settings;
}

LODRenderSettings FootprintLOD::getZoomInEnhancedRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // When zoomed in significantly, enhance the detail level
    if (zoom_factor > max_detail_zoom_ * 1.5f) {
        // Enable additional features when zoomed in
        settings.render_detailed_annotations = true;
        settings.render_labels = true;
        settings.render_text = true;

        if (zoom_factor > max_detail_zoom_ * 2.5f) {
            // At extreme zoom in, add even more detail
            settings.alpha_multiplier = 1.15f;  // Slightly more opaque
            settings.border_thickness = 3.0f;   // Thicker borders for better visibility
            settings.render_text = true;        // Ensure text is rendered
        } else if (zoom_factor > max_detail_zoom_ * 1.8f) {
            settings.alpha_multiplier = 1.1f;
            settings.border_thickness = 2.5f;
        }
    }

    return settings;
}

void FootprintLOD::applyZoomInOutLODToCell(const FootprintCell& cell,
                                         ImDrawList* draw_list,
                                         float zoom_factor,
                                         double max_volume,
                                         const std::vector<FootprintCell>& diagonal_imbalances,
                                         const std::vector<FootprintCell>& stacked_imbalances,
                                         const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level based on zoom direction (in/out)
    LODLevel lod_level;
    if (zoom_factor < 0.7f) {
        // Zoomed out - use zoom-out optimized LOD
        lod_level = calculateZoomOutLOD(cell_width_px, cell_height_px, zoom_factor);
    } else {
        // Zoomed in or normal - use zoom-in enhanced LOD
        lod_level = calculateZoomInLOD(cell_width_px, cell_height_px, zoom_factor);
    }

    float cell_area_px = cell_width_px * cell_height_px;

    // Get appropriate render settings based on zoom level
    LODRenderSettings settings;
    if (zoom_factor < 0.7f) {
        settings = getZoomOutOptimizedRenderSettings(lod_level, zoom_factor);
    } else {
        settings = getZoomInEnhancedRenderSettings(lod_level, zoom_factor);
    }

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply zoom-based alpha adjustment
    if (zoom_factor < 0.5f) {
        alpha_multiplier *= 0.85f; // Reduce alpha when heavily zoomed out
    } else if (zoom_factor > max_detail_zoom_ * 1.5f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.1f, 1.2f); // Increase alpha when zoomed in
    }

    // Render heatmap/fill based on zoom-aware LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on zoom-aware LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on zoom-aware LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on zoom-aware LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

LODLevel FootprintLOD::calculateAdaptiveZoomLOD(float cell_width_px, float cell_height_px,
                                              float zoom_factor, float view_width, float view_height,
                                              int total_cells_in_view) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate view density to determine appropriate LOD level
    float view_area = view_width * view_height;
    float cell_density = static_cast<float>(total_cells_in_view) / view_area;

    // Adjust LOD based on both zoom level and cell density
    if (zoom_factor < 0.2f || cell_density > 0.005f) {  // Very low zoom or very high density
        // Use lowest detail to maintain performance
        if (min_dimension > min_cell_size_px_ * 0.2f) {
            return LODLevel::LOW_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL; // Skip rendering entirely
        }
    } else if (zoom_factor < 0.5f || cell_density > 0.002f) {  // Low zoom or high density
        if (min_dimension > min_cell_size_px_ * 0.5f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (zoom_factor > max_detail_zoom_ * 2.0f) {  // Very high zoom
        // Allow maximum detail when zoomed in significantly
        if (min_dimension > max_cell_size_px_ * 2.0f) {
            return LODLevel::MAX_DETAIL;
        } else if (min_dimension > medium_cell_size_px_ * 1.5f) {
            return LODLevel::HIGH_DETAIL;
        } else {
            return LODLevel::HIGH_DETAIL; // Still high detail due to zoom level
        }
    } else if (zoom_factor > max_detail_zoom_) {  // High zoom
        if (min_dimension > max_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else {
            return LODLevel::MEDIUM_DETAIL;
        }
    } else {  // Normal zoom levels
        return calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    }
}

LODRenderSettings FootprintLOD::getAdaptiveZoomRenderSettings(LODLevel lod_level,
                                                           float zoom_factor,
                                                           int total_cells_in_view) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Adjust settings based on zoom level and cell count
    if (zoom_factor < 0.3f || total_cells_in_view > 5000) {
        // When zoomed out or many cells, simplify rendering
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier *= 0.8f;
    } else if (zoom_factor > max_detail_zoom_ * 1.5f) {
        // When zoomed in, enhance detail
        settings.render_detailed_annotations = true;
        settings.render_labels = true;
        settings.render_text = true;
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 1.1f, 1.2f);
        settings.border_thickness = std::max(settings.border_thickness * 1.2f, 2.0f);
    }

    return settings;
}

void FootprintLOD::applyAdaptiveZoomLODToCell(const FootprintCell& cell,
                                            ImDrawList* draw_list,
                                            float zoom_factor,
                                            double max_volume,
                                            const std::vector<FootprintCell>& diagonal_imbalances,
                                            const std::vector<FootprintCell>& stacked_imbalances,
                                            const FootprintPanel* panel,
                                            float view_width,
                                            float view_height,
                                            int total_cells_in_view) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using adaptive zoom calculation
    LODLevel lod_level = calculateAdaptiveZoomLOD(cell_width_px, cell_height_px,
                                                zoom_factor, view_width, view_height,
                                                total_cells_in_view);
    float cell_area_px = cell_width_px * cell_height_px;

    // Get adaptive render settings
    LODRenderSettings settings = getAdaptiveZoomRenderSettings(lod_level, zoom_factor, total_cells_in_view);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply additional zoom-based alpha adjustment
    if (zoom_factor < 0.4f) {
        alpha_multiplier *= 0.8f; // Reduce alpha when zoomed out
    } else if (zoom_factor > max_detail_zoom_ * 2.0f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.15f, 1.3f); // Increase alpha when zoomed in
    }

    // Render heatmap/fill based on adaptive zoom LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on adaptive zoom LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on adaptive zoom LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on adaptive zoom LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

// Additional LOD methods for improved zoom-based detail management

LODLevel FootprintLOD::calculateSmoothZoomLOD(float cell_width_px, float cell_height_px,
                                            float zoom_factor, float prev_zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate base LOD level
    LODLevel current_lod = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);

    // Calculate previous LOD level
    LODLevel prev_lod = calculateLODLevel(cell_width_px, cell_height_px, prev_zoom_factor);

    // Smooth transitions between zoom levels to prevent flickering
    float zoom_change = std::abs(zoom_factor - prev_zoom_factor);

    // If zoom changed significantly, allow the new LOD
    // If zoom changed slightly, maintain some stability to prevent flickering
    if (zoom_change < 0.05f && current_lod != prev_lod) {
        // Small zoom change, maintain stability - use the lower detail level to be conservative
        return static_cast<LODLevel>(std::min(static_cast<int>(current_lod), static_cast<int>(prev_lod)));
    }

    return current_lod;
}

LODRenderSettings FootprintLOD::getSmoothZoomRenderSettings(LODLevel lod_level,
                                                          float zoom_factor,
                                                          float prev_zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Apply smoothing based on zoom change magnitude
    float zoom_change = std::abs(zoom_factor - prev_zoom_factor);

    if (zoom_change < 0.05f) {
        // Small zoom change - gradually adjust settings to prevent sudden changes
        settings.alpha_multiplier *= 0.95f; // Slightly reduce to account for transition
    } else if (zoom_change > 0.2f) {
        // Large zoom change - allow full settings
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 1.05f, 1.2f);
    }

    return settings;
}

void FootprintLOD::applySmoothZoomLODToCell(const FootprintCell& cell,
                                          ImDrawList* draw_list,
                                          float zoom_factor,
                                          float prev_zoom_factor,
                                          double max_volume,
                                          const std::vector<FootprintCell>& diagonal_imbalances,
                                          const std::vector<FootprintCell>& stacked_imbalances,
                                          const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using smooth zoom calculation
    LODLevel lod_level = calculateSmoothZoomLOD(cell_width_px, cell_height_px, zoom_factor, prev_zoom_factor);
    float cell_area_px = cell_width_px * cell_height_px;

    // Get smooth zoom render settings
    LODRenderSettings settings = getSmoothZoomRenderSettings(lod_level, zoom_factor, prev_zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply zoom-based alpha adjustment
    if (zoom_factor < 0.4f) {
        alpha_multiplier *= 0.8f; // Reduce alpha when zoomed out
    } else if (zoom_factor > max_detail_zoom_ * 2.0f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.15f, 1.3f); // Increase alpha when zoomed in
    }

    // Render heatmap/fill based on smooth zoom LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on smooth zoom LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on smooth zoom LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on smooth zoom LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

LODLevel FootprintLOD::calculateGradientBasedLOD(const FootprintCell& cell,
                                               float cell_width_px, float cell_height_px,
                                               float zoom_factor,
                                               const std::vector<FootprintCell>& gradient_neighbors) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate base LOD level
    LODLevel base_lod = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);

    // Analyze gradient of volume/activity around this cell
    if (!gradient_neighbors.empty()) {
        double avg_volume = 0.0;
        double max_volume = 0.0;

        for (const auto& neighbor : gradient_neighbors) {
            double vol = neighbor.bid_volume + neighbor.ask_volume;
            avg_volume += vol;
            if (vol > max_volume) max_volume = vol;
        }

        avg_volume /= gradient_neighbors.size();
        double center_volume = cell.bid_volume + cell.ask_volume;

        // If this cell has significantly higher volume than neighbors, increase detail
        if (avg_volume > 0 && center_volume > avg_volume * 1.5) {
            switch (base_lod) {
                case LODLevel::LOW_DETAIL:
                    return LODLevel::MEDIUM_DETAIL;
                case LODLevel::MEDIUM_DETAIL:
                    return LODLevel::HIGH_DETAIL;
                default:
                    return base_lod;
            }
        }
        // If this cell has significantly lower volume than neighbors, decrease detail
        else if (avg_volume > 0 && center_volume < avg_volume * 0.3) {
            switch (base_lod) {
                case LODLevel::HIGH_DETAIL:
                    return LODLevel::MEDIUM_DETAIL;
                case LODLevel::MEDIUM_DETAIL:
                    return LODLevel::LOW_DETAIL;
                default:
                    return base_lod;
            }
        }
    }

    return base_lod;
}

void FootprintLOD::applyGradientBasedLODToCell(const FootprintCell& cell,
                                             ImDrawList* draw_list,
                                             float zoom_factor,
                                             double max_volume,
                                             const std::vector<FootprintCell>& diagonal_imbalances,
                                             const std::vector<FootprintCell>& stacked_imbalances,
                                             const FootprintPanel* panel,
                                             const std::vector<FootprintCell>& gradient_neighbors) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using gradient-based calculation
    LODLevel lod_level = calculateGradientBasedLOD(cell, cell_width_px, cell_height_px, zoom_factor, gradient_neighbors);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getOptimizedRenderSettings(lod_level, cell_area_px);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply zoom-based alpha adjustment
    if (zoom_factor < 0.4f) {
        alpha_multiplier *= 0.8f; // Reduce alpha when zoomed out
    } else if (zoom_factor > max_detail_zoom_ * 2.0f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.15f, 1.3f); // Increase alpha when zoomed in
    }

    // Render heatmap/fill based on gradient-based LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on gradient-based LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on gradient-based LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on gradient-based LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

// Additional methods for improved zoom-based LOD management

LODLevel FootprintLOD::calculateZoomLevelLOD(float cell_width_px, float cell_height_px,
                                          float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Define more granular zoom levels for better control
    if (zoom_factor < 0.05f) {
        // Extremely zoomed out - only show most important information
        return LODLevel::LOW_DETAIL;
    } else if (zoom_factor < 0.1f) {
        // Very zoomed out - minimal detail
        if (min_dimension > min_cell_size_px_ * 0.2f) {
            return LODLevel::LOW_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL; // Skip rendering
        }
    } else if (zoom_factor < 0.25f) {
        // Moderately zoomed out - basic detail
        if (min_dimension > min_cell_size_px_ * 0.5f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (zoom_factor < 0.5f) {
        // Somewhat zoomed out - medium detail
        if (min_dimension > medium_cell_size_px_ * 0.7f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (zoom_factor < 1.0f) {
        // Near normal zoom - use standard calculation
        if (min_dimension > medium_cell_size_px_) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (zoom_factor < 2.0f) {
        // Zoomed in - high detail
        if (min_dimension > medium_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else if (min_dimension > min_cell_size_px_) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else {
        // Highly zoomed in - maximum detail
        if (min_dimension > max_cell_size_px_) {
            return LODLevel::MAX_DETAIL;
        } else if (min_dimension > medium_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else {
            return LODLevel::MEDIUM_DETAIL;
        }
    }
}

LODRenderSettings FootprintLOD::getZoomLevelRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Adjust settings based on zoom level for optimal performance and visual quality
    if (zoom_factor < 0.25f) {
        // When zoomed out, simplify rendering to improve performance
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;

        // Reduce alpha slightly to prevent visual clutter
        settings.alpha_multiplier *= 0.85f;
    } else if (zoom_factor > 2.0f) {
        // When zoomed in, enhance detail
        settings.render_detailed_annotations = true;
        settings.render_labels = true;
        settings.render_text = true;

        // Increase alpha and border thickness for better visibility
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 1.1f, 1.2f);
        settings.border_thickness = std::max(settings.border_thickness * 1.2f, 2.0f);
    }

    return settings;
}

void FootprintLOD::applyZoomLevelLODToCell(const FootprintCell& cell,
                                        ImDrawList* draw_list,
                                        float zoom_factor,
                                        double max_volume,
                                        const std::vector<FootprintCell>& diagonal_imbalances,
                                        const std::vector<FootprintCell>& stacked_imbalances,
                                        const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using zoom-level optimized calculation
    LODLevel lod_level = calculateZoomLevelLOD(cell_width_px, cell_height_px, zoom_factor);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getZoomLevelRenderSettings(lod_level, zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply additional zoom-based alpha adjustment
    if (zoom_factor < 0.2f) {
        alpha_multiplier *= 0.7f; // Reduce alpha when heavily zoomed out
    } else if (zoom_factor > 2.5f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.2f, 1.3f); // Increase alpha when zoomed in
    }

    // Render heatmap/fill based on zoom-level optimized LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on zoom-level optimized LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on zoom-level optimized LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on zoom-level optimized LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

// Implementation for tile-based LOD for very large datasets
LODLevel FootprintLOD::calculateTileBasedLOD(float cell_width_px, float cell_height_px,
                                          float zoom_factor, int tile_size_px) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate how many cells fit in a tile at this zoom level
    float cells_per_tile = static_cast<float>(tile_size_px) / std::max(cell_width_px, cell_height_px);

    if (cells_per_tile > 4.0f) {
        // Many cells per tile - use low detail to avoid clutter
        return LODLevel::LOW_DETAIL;
    } else if (cells_per_tile > 2.0f) {
        // Moderate number of cells per tile - use medium detail
        if (min_dimension > min_cell_size_px_) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (cells_per_tile > 0.5f) {
        // Few cells per tile - use high detail
        if (min_dimension > medium_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else {
            return LODLevel::MEDIUM_DETAIL;
        }
    } else {
        // Very few cells per tile - use maximum detail
        if (min_dimension > max_cell_size_px_) {
            return LODLevel::MAX_DETAIL;
        } else {
            return LODLevel::HIGH_DETAIL;
        }
    }
}

// Implementation for enhanced detail when zoomed in significantly
LODLevel FootprintLOD::calculateEnhancedZoomInLOD(float cell_width_px, float cell_height_px,
                                               float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // When zoomed in significantly, allow for even more detailed representations
    if (zoom_factor > max_detail_zoom_ * 3.0f) {
        // At extreme zoom in, provide ultra-high detail
        if (min_dimension > max_cell_size_px_ * 4.0f) {
            return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
        } else if (min_dimension > max_cell_size_px_ * 2.0f) {
            return LODLevel::MAX_DETAIL;  // Still ultra detail
        } else if (min_dimension > medium_cell_size_px_ * 1.5f) {
            return LODLevel::HIGH_DETAIL; // High detail
        } else {
            return LODLevel::HIGH_DETAIL; // High detail due to zoom level
        }
    } else if (zoom_factor > max_detail_zoom_ * 2.0f) {
        // Very high zoom in - provide high detail
        if (min_dimension > max_cell_size_px_ * 3.0f) {
            return LODLevel::MAX_DETAIL;  // Ultra detail
        } else if (min_dimension > max_cell_size_px_ * 1.5f) {
            return LODLevel::HIGH_DETAIL; // High detail
        } else {
            return LODLevel::HIGH_DETAIL; // High detail due to zoom level
        }
    } else if (zoom_factor > max_detail_zoom_ * 1.2f) {
        // High zoom in - provide high detail
        if (min_dimension > max_cell_size_px_ * 2.0f) {
            return LODLevel::HIGH_DETAIL;  // High detail
        } else if (min_dimension > medium_cell_size_px_ * 1.2f) {
            return LODLevel::HIGH_DETAIL;  // High detail due to size
        } else {
            return LODLevel::MEDIUM_DETAIL; // Medium detail
        }
    } else {
        // Standard zoom levels - use regular calculation
        return calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    }
}

LODRenderSettings FootprintLOD::getEnhancedZoomInRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // When zoomed in significantly, enhance the detail level
    if (zoom_factor > max_detail_zoom_ * 2.0f) {
        // Enable additional features when zoomed in
        settings.render_detailed_annotations = true;
        settings.render_labels = true;
        settings.render_text = true;

        if (zoom_factor > max_detail_zoom_ * 3.0f) {
            // At extreme zoom in, add even more detail
            settings.alpha_multiplier = 1.25f;  // More opaque
            settings.border_thickness = 3.5f;   // Much thicker borders for better visibility
            settings.render_text = true;        // Ensure text is rendered
        } else if (zoom_factor > max_detail_zoom_ * 2.5f) {
            settings.alpha_multiplier = 1.2f;   // More opaque
            settings.border_thickness = 3.0f;   // Thicker borders for better visibility
        } else {
            settings.alpha_multiplier = 1.15f;  // Slightly more opaque
            settings.border_thickness = 2.5f;   // Thicker borders for better visibility
        }
    }

    return settings;
}

void FootprintLOD::applyEnhancedZoomInLODToCell(const FootprintCell& cell,
                                             ImDrawList* draw_list,
                                             float zoom_factor,
                                             double max_volume,
                                             const std::vector<FootprintCell>& diagonal_imbalances,
                                             const std::vector<FootprintCell>& stacked_imbalances,
                                             const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level with enhanced zoom-in calculation
    LODLevel lod_level = calculateEnhancedZoomInLOD(cell_width_px, cell_height_px, zoom_factor);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getEnhancedZoomInRenderSettings(lod_level, zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Render heatmap/fill based on enhanced zoom-in LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on enhanced zoom-in LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells even more when zoomed in
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 2.0f, p1.y - 2.0f);
                ImVec2 offset_p2(p2.x + 2.0f, p2.y + 2.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness * 0.7f);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on enhanced zoom-in LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on enhanced zoom-in LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 4.0f;  // Slightly taller when zoomed in
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 2.0f);  // Position lower when zoomed in

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

void FootprintLOD::applyTileBasedLODToCell(const FootprintCell& cell,
                                        ImDrawList* draw_list,
                                        float zoom_factor,
                                        double max_volume,
                                        const std::vector<FootprintCell>& diagonal_imbalances,
                                        const std::vector<FootprintCell>& stacked_imbalances,
                                        const FootprintPanel* panel,
                                        int tile_size_px) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using tile-based calculation
    LODLevel lod_level = calculateTileBasedLOD(cell_width_px, cell_height_px, zoom_factor, tile_size_px);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getOptimizedRenderSettings(lod_level, cell_area_px);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply zoom-based alpha adjustment
    if (zoom_factor < 0.3f) {
        alpha_multiplier *= 0.8f; // Reduce alpha when zoomed out
    } else if (zoom_factor > max_detail_zoom_ * 2.0f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.15f, 1.3f); // Increase alpha when zoomed in
    }

    // Render heatmap/fill based on tile-based LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on tile-based LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on tile-based LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on tile-based LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = 3.0f;
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - 1.0f);

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

// Implementation for simplified representations when zoomed out significantly
LODLevel FootprintLOD::calculateSimplifiedZoomOutLOD(float cell_width_px, float cell_height_px,
                                                  float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // When zoomed out significantly, use more aggressive LOD reduction
    if (zoom_factor < 0.05f) {
        // Extremely zoomed out - only show heatmap for most important cells
        if (min_dimension > min_cell_size_px_ * 0.1f) {
            return LODLevel::LOW_DETAIL;  // Show only heatmap
        } else {
            return LODLevel::LOW_DETAIL;  // Skip rendering entirely for tiny cells
        }
    } else if (zoom_factor < 0.15f) {
        // Highly zoomed out - reduce detail significantly
        if (min_dimension > min_cell_size_px_ * 0.3f) {
            return LODLevel::MEDIUM_DETAIL;  // Show heatmap and basic borders
        } else {
            return LODLevel::LOW_DETAIL;  // Skip rendering for very small cells
        }
    } else if (zoom_factor < 0.4f) {
        // Moderately zoomed out - use medium detail
        if (min_dimension > medium_cell_size_px_ * 0.5f) {
            return LODLevel::MEDIUM_DETAIL;  // Show heatmap and basic borders
        } else {
            return LODLevel::LOW_DETAIL;  // Only heatmap
        }
    } else {
        // Normal zoom levels - use standard calculation
        return calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);
    }
}

LODRenderSettings FootprintLOD::getSimplifiedZoomOutRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // When zoomed out, further optimize settings to improve performance and reduce visual clutter
    if (zoom_factor < 0.15f) {
        // At high zoom out, disable text, labels and detailed annotations to reduce visual clutter
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;

        // Reduce alpha slightly to make visualization clearer when many cells are present
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 0.7f, 0.9f);

        // Use thinner borders to reduce visual noise
        settings.border_thickness = std::max(settings.border_thickness * 0.5f, 0.5f);
    } else if (zoom_factor < 0.4f) {
        // At moderate zoom out, consider disabling detailed annotations
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 0.85f, 1.0f);
    }

    return settings;
}

void FootprintLOD::applySimplifiedZoomOutLODToCell(const FootprintCell& cell,
                                                ImDrawList* draw_list,
                                                float zoom_factor,
                                                double max_volume,
                                                const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Early exit if cell should not be rendered at all
    if (shouldCompletelySkipRendering(cell_width_px, cell_height_px, zoom_factor)) {
        return;
    }

    // Determine LOD level using simplified zoom-out calculation
    LODLevel lod_level = calculateSimplifiedZoomOutLOD(cell_width_px, cell_height_px, zoom_factor);
    LODRenderSettings settings = getSimplifiedZoomOutRenderSettings(lod_level, zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply additional zoom-based alpha adjustment for zoomed-out views
    if (zoom_factor < 0.2f) {
        alpha_multiplier *= 0.6f;  // Reduce alpha significantly when heavily zoomed out
    } else if (zoom_factor < 0.4f) {
        alpha_multiplier *= 0.8f;  // Reduce alpha moderately when zoomed out
    }

    // Render heatmap/fill based on simplified zoom-out LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on simplified zoom-out LOD settings
    if (settings.render_borders) {
        // Use a simpler border approach when zoomed out
        unsigned char border_alpha = static_cast<unsigned char>(10 * alpha_multiplier);  // Reduced alpha for borders
        ImU32 border_color = IM_COL32(255, 255, 255, border_alpha);
        float thickness = settings.border_thickness;

        // Draw a simpler border when zoomed out
        draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
    }
}

// Implementation for aggregated cell representation when zoomed out
std::vector<FootprintCell> FootprintLOD::aggregateCellsForZoomOut(const std::vector<FootprintCell>& cells,
                                                              float zoom_factor,
                                                              float grid_size) const {
    if (zoom_factor > 0.5f) {
        // Not zoomed out enough, return original cells
        return cells;
    }

    std::vector<FootprintCell> aggregated_cells;
    std::unordered_map<std::pair<int, int>, FootprintCell, PairHash> grid_map;

    // Grid-based aggregation
    for (const auto& cell : cells) {
        // Calculate grid position
        int grid_x = static_cast<int>(cell.x / grid_size);
        int grid_y = static_cast<int>(cell.y / grid_size);

        auto grid_key = std::make_pair(grid_x, grid_y);

        if (grid_map.find(grid_key) == grid_map.end()) {
            // Create new aggregated cell using the constructor
            FootprintCell aggregated_cell(
                grid_x * grid_size + grid_size * 0.5f,  // x: Center in grid
                grid_y * grid_size + grid_size * 0.5f,  // y: Center in grid
                grid_size * 0.9f,                       // width: Slightly smaller to avoid overlap
                grid_size * 0.9f,                       // height: Slightly smaller to avoid overlap
                cell.bid_volume,                        // bid_volume
                cell.ask_volume,                        // ask_volume
                cell.trade_count,                       // trade_count
                cell.vwap                               // vwap
            );

            // Copy additional properties
            aggregated_cell.delta = cell.delta;
            aggregated_cell.buy_trade_count = cell.buy_trade_count;
            aggregated_cell.sell_trade_count = cell.sell_trade_count;
            aggregated_cell.max_single_trade_volume = cell.max_single_trade_volume;
            aggregated_cell.start_time_ns = cell.start_time_ns;
            aggregated_cell.end_time_ns = cell.end_time_ns;

            grid_map[grid_key] = aggregated_cell;
        } else {
            // Aggregate with existing cell in grid
            FootprintCell& existing_cell = grid_map[grid_key];

            // Sum volumes
            existing_cell.bid_volume += cell.bid_volume;
            existing_cell.ask_volume += cell.ask_volume;
            existing_cell.trade_count += cell.trade_count;

            // Average delta based on total volume
            double total_vol = existing_cell.bid_volume + existing_cell.ask_volume;
            if (total_vol > 0) {
                existing_cell.delta = ((existing_cell.delta * (total_vol - cell.bid_volume - cell.ask_volume)) +
                                      (cell.delta * (cell.bid_volume + cell.ask_volume))) / total_vol;
            }
        }
    }

    // Convert map to vector
    for (const auto& pair : grid_map) {
        aggregated_cells.push_back(pair.second);
    }

    return aggregated_cells;
}

// Implementation for advanced zoom-based LOD with smooth transitions
LODLevel FootprintLOD::calculateAdvancedZoomLOD(float cell_width_px, float cell_height_px,
                                              float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate zoom level with more granular thresholds for smoother transitions
    if (zoom_factor < 0.02f) {
        // Extremely zoomed out - only show most important information
        return LODLevel::LOW_DETAIL;
    } else if (zoom_factor < 0.08f) {
        // Very zoomed out - minimal detail
        if (min_dimension > min_cell_size_px_ * 0.1f) {
            return LODLevel::LOW_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL; // Skip rendering
        }
    } else if (zoom_factor < 0.2f) {
        // Moderately zoomed out - basic detail
        if (min_dimension > min_cell_size_px_ * 0.4f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (zoom_factor < 0.4f) {
        // Somewhat zoomed out - medium detail
        if (min_dimension > medium_cell_size_px_ * 0.6f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (zoom_factor < 0.8f) {
        // Near normal zoom - use standard calculation
        if (min_dimension > medium_cell_size_px_ * 0.8f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (zoom_factor < 1.5f) {
        // Near normal zoom - high detail
        if (min_dimension > medium_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else if (min_dimension > min_cell_size_px_) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (zoom_factor < 3.0f) {
        // Zoomed in - maximum detail
        if (min_dimension > max_cell_size_px_) {
            return LODLevel::MAX_DETAIL;
        } else if (min_dimension > medium_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else {
            return LODLevel::MEDIUM_DETAIL;
        }
    } else {
        // Highly zoomed in - ultra maximum detail
        if (min_dimension > max_cell_size_px_ * 2.0f) {
            return LODLevel::MAX_DETAIL;  // Ultra detail with additional annotations
        } else if (min_dimension > max_cell_size_px_) {
            return LODLevel::HIGH_DETAIL; // High detail with most annotations
        } else {
            return LODLevel::HIGH_DETAIL; // High detail due to zoom level
        }
    }
}

LODRenderSettings FootprintLOD::getAdvancedZoomRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Adjust settings based on zoom level for optimal performance and visual quality
    if (zoom_factor < 0.2f) {
        // When zoomed out, simplify rendering to improve performance
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;

        // Reduce alpha slightly to prevent visual clutter
        settings.alpha_multiplier *= 0.8f;

        // Use thinner borders to reduce visual noise
        settings.border_thickness = std::max(settings.border_thickness * 0.6f, 0.5f);
    } else if (zoom_factor > 1.5f) {
        // When zoomed in, enhance detail
        settings.render_detailed_annotations = true;
        settings.render_labels = true;
        settings.render_text = true;

        // Increase alpha and border thickness for better visibility
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 1.15f, 1.25f);
        settings.border_thickness = std::max(settings.border_thickness * 1.3f, 2.2f);
    } else if (zoom_factor > 3.0f) {
        // When extremely zoomed in, add even more detail
        settings.render_detailed_annotations = true;
        settings.render_labels = true;
        settings.render_text = true;

        // Further increase alpha and border thickness for maximum visibility
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 1.25f, 1.35f);
        settings.border_thickness = std::max(settings.border_thickness * 1.5f, 2.5f);
    }

    return settings;
}

void FootprintLOD::applyAdvancedZoomLODToCell(const FootprintCell& cell,
                                           ImDrawList* draw_list,
                                           float zoom_factor,
                                           double max_volume,
                                           const std::vector<FootprintCell>& diagonal_imbalances,
                                           const std::vector<FootprintCell>& stacked_imbalances,
                                           const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using advanced zoom calculation
    LODLevel lod_level = calculateAdvancedZoomLOD(cell_width_px, cell_height_px, zoom_factor);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getAdvancedZoomRenderSettings(lod_level, zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply additional zoom-based alpha adjustment
    if (zoom_factor < 0.1f) {
        alpha_multiplier *= 0.6f; // Reduce alpha when heavily zoomed out
    } else if (zoom_factor > 3.5f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.3f, 1.4f); // Increase alpha when extremely zoomed in
    } else if (zoom_factor > 2.0f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.2f, 1.3f); // Increase alpha when zoomed in
    }

    // Render heatmap/fill based on advanced zoom LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on advanced zoom LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 2.0f, p1.y - 2.0f);
                ImVec2 offset_p2(p2.x + 2.0f, p2.y + 2.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness * 0.8f);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on advanced zoom LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on advanced zoom LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = (zoom_factor > 2.0f) ? 5.0f : 3.0f;  // Taller bars when zoomed in
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - (zoom_factor > 2.0f ? 2.0f : 1.0f));  // Position lower when zoomed in

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

// Implementation for multi-scale LOD that adapts to different viewing scales
LODLevel FootprintLOD::calculateMultiScaleLOD(float cell_width_px, float cell_height_px,
                                            float zoom_factor, float view_scale) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate effective zoom considering both zoom factor and view scale
    float effective_zoom = zoom_factor * view_scale;

    // Define different LOD thresholds based on the effective scale
    if (effective_zoom < 0.05f) {
        // Extremely zoomed out at global scale - only show heatmap
        return LODLevel::LOW_DETAIL;
    } else if (effective_zoom < 0.15f) {
        // Global scale - minimal detail
        if (min_dimension > min_cell_size_px_ * 0.2f) {
            return LODLevel::LOW_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL; // Skip rendering
        }
    } else if (effective_zoom < 0.4f) {
        // Regional scale - basic detail
        if (min_dimension > min_cell_size_px_ * 0.6f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (effective_zoom < 0.8f) {
        // Local scale - medium detail
        if (min_dimension > medium_cell_size_px_ * 0.8f) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else if (effective_zoom < 2.0f) {
        // Detailed scale - high detail
        if (min_dimension > medium_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else if (min_dimension > min_cell_size_px_) {
            return LODLevel::MEDIUM_DETAIL;
        } else {
            return LODLevel::LOW_DETAIL;
        }
    } else {
        // Micro scale - maximum detail
        if (min_dimension > max_cell_size_px_) {
            return LODLevel::MAX_DETAIL;
        } else if (min_dimension > medium_cell_size_px_) {
            return LODLevel::HIGH_DETAIL;
        } else {
            return LODLevel::MEDIUM_DETAIL;
        }
    }
}

LODRenderSettings FootprintLOD::getMultiScaleRenderSettings(LODLevel lod_level, float view_scale) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Adjust settings based on the view scale
    if (view_scale < 0.3f) {
        // Global view - simplify rendering
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier *= 0.7f;
    } else if (view_scale > 2.0f) {
        // Detailed view - enhance rendering
        settings.render_detailed_annotations = true;
        settings.render_labels = true;
        settings.render_text = true;
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 1.2f, 1.3f);
        settings.border_thickness = std::max(settings.border_thickness * 1.4f, 2.3f);
    }

    return settings;
}

void FootprintLOD::applyMultiScaleLODToCell(const FootprintCell& cell,
                                         ImDrawList* draw_list,
                                         float zoom_factor,
                                         double max_volume,
                                         const std::vector<FootprintCell>& diagonal_imbalances,
                                         const std::vector<FootprintCell>& stacked_imbalances,
                                         const FootprintPanel* panel,
                                         float view_scale) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using multi-scale calculation
    LODLevel lod_level = calculateMultiScaleLOD(cell_width_px, cell_height_px, zoom_factor, view_scale);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getMultiScaleRenderSettings(lod_level, view_scale);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply view-scale based alpha adjustment
    if (view_scale < 0.5f) {
        alpha_multiplier *= 0.75f; // Reduce alpha for global views
    } else if (view_scale > 1.5f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.15f, 1.25f); // Increase alpha for detailed views
    }

    // Render heatmap/fill based on multi-scale LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and view scale
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on multi-scale LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.5f, p1.y - 1.5f);
                ImVec2 offset_p2(p2.x + 1.5f, p2.y + 1.5f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness * 0.9f);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on multi-scale LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on multi-scale LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = (view_scale > 1.5f) ? 4.0f : 3.0f;  // Taller bars in detailed views
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - (view_scale > 1.5f ? 1.5f : 1.0f));  // Position lower in detailed views

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

// Implementation for continuous LOD that provides smooth transitions between zoom levels
LODLevel FootprintLOD::calculateContinuousLOD(float cell_width_px, float cell_height_px,
                                           float zoom_factor) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate continuous LOD value that smoothly transitions between discrete levels
    // This creates a continuous value between LOD levels for smooth transitions

    // Base thresholds with some overlap to ensure smooth transitions
    float low_to_medium_threshold = min_detail_zoom_ * 0.7f;
    float medium_to_high_threshold = medium_detail_zoom_ * 0.8f;
    float high_to_max_threshold = max_detail_zoom_ * 0.9f;

    float size_low_threshold = min_cell_size_px_ * 0.8f;
    float size_medium_threshold = medium_cell_size_px_ * 0.9f;
    float size_high_threshold = max_cell_size_px_;

    // Determine which range the cell falls into
    if (zoom_factor <= low_to_medium_threshold || min_dimension <= size_low_threshold) {
        return LODLevel::LOW_DETAIL;
    } else if (zoom_factor <= medium_to_high_threshold || min_dimension <= size_medium_threshold) {
        // In transition zone between low and medium
        float zoom_ratio = (zoom_factor - low_to_medium_threshold) / (medium_to_high_threshold - low_to_medium_threshold);
        float size_ratio = (min_dimension - size_low_threshold) / (size_medium_threshold - size_low_threshold);

        // Use the higher of the two ratios to determine LOD
        float ratio = std::max(zoom_ratio, size_ratio);
        return (ratio > 0.5f) ? LODLevel::MEDIUM_DETAIL : LODLevel::LOW_DETAIL;
    } else if (zoom_factor <= high_to_max_threshold || min_dimension <= size_high_threshold) {
        // In transition zone between medium and high
        float zoom_ratio = (zoom_factor - medium_to_high_threshold) / (high_to_max_threshold - medium_to_high_threshold);
        float size_ratio = (min_dimension - size_medium_threshold) / (size_high_threshold - size_medium_threshold);

        float ratio = std::max(zoom_ratio, size_ratio);
        return (ratio > 0.5f) ? LODLevel::HIGH_DETAIL : LODLevel::MEDIUM_DETAIL;
    } else {
        // In transition zone between high and max
        float zoom_ratio = (zoom_factor - high_to_max_threshold) / (max_detail_zoom_ * 1.5f - high_to_max_threshold);
        float size_ratio = (min_dimension - size_high_threshold) / (max_cell_size_px_ * 1.5f - size_high_threshold);

        float ratio = std::max(zoom_ratio, size_ratio);
        return (ratio > 0.3f) ? LODLevel::MAX_DETAIL : LODLevel::HIGH_DETAIL;
    }
}

LODRenderSettings FootprintLOD::getContinuousLODRenderSettings(LODLevel lod_level, float zoom_factor) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Apply continuous adjustments based on zoom level for smooth transitions
    if (zoom_factor < 0.1f) {
        // Very zoomed out - minimize visual complexity
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier = 0.6f;
        settings.border_thickness = 0.5f;
    } else if (zoom_factor < 0.3f) {
        // Moderately zoomed out - basic detail
        settings.render_text = false;
        settings.render_labels = false;
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier = 0.7f;
        settings.border_thickness = 0.8f;
    } else if (zoom_factor < 0.7f) {
        // Approaching normal zoom - medium detail
        settings.render_text = false;
        settings.render_labels = true;
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier = 0.85f;
        settings.border_thickness = 1.0f;
    } else if (zoom_factor < 1.5f) {
        // Normal zoom - high detail
        settings.render_text = true;
        settings.render_labels = true;
        settings.render_detailed_annotations = false;
        settings.alpha_multiplier = 1.0f;
        settings.border_thickness = 1.5f;
    } else if (zoom_factor < 3.0f) {
        // Zoomed in - maximum detail
        settings.render_text = true;
        settings.render_labels = true;
        settings.render_detailed_annotations = true;
        settings.alpha_multiplier = 1.1f;
        settings.border_thickness = 2.0f;
    } else {
        // Highly zoomed in - ultra detail
        settings.render_text = true;
        settings.render_labels = true;
        settings.render_detailed_annotations = true;
        settings.alpha_multiplier = 1.2f;
        settings.border_thickness = 2.5f;
    }

    return settings;
}

void FootprintLOD::applyContinuousLODToCell(const FootprintCell& cell,
                                       ImDrawList* draw_list,
                                       float zoom_factor,
                                       double max_volume,
                                       const std::vector<FootprintCell>& diagonal_imbalances,
                                       const std::vector<FootprintCell>& stacked_imbalances,
                                       const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using continuous calculation
    LODLevel lod_level = calculateContinuousLOD(cell_width_px, cell_height_px, zoom_factor);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getContinuousLODRenderSettings(lod_level, zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply zoom-based alpha adjustment for continuous transitions
    if (zoom_factor < 0.2f) {
        alpha_multiplier *= 0.6f; // Reduce alpha when heavily zoomed out
    } else if (zoom_factor > 2.5f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.2f, 1.3f); // Increase alpha when zoomed in
    }

    // Render heatmap/fill based on continuous LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on continuous LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.0f, p1.y - 1.0f);
                ImVec2 offset_p2(p2.x + 1.0f, p2.y + 1.0f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on continuous LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on continuous LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = (zoom_factor > 2.0f) ? 4.0f : 3.0f;  // Taller bars when zoomed in
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - (zoom_factor > 2.0f ? 1.5f : 1.0f));

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

// Implementation for zoom-based simplification when zoomed out
void FootprintLOD::applyZoomOutSimplificationLODToCell(const FootprintCell& cell,
                                                      ImDrawList* draw_list,
                                                      float zoom_factor,
                                                      double max_volume,
                                                      const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Early exit if cell should not be rendered at all
    if (shouldCompletelySkipRendering(cell_width_px, cell_height_px, zoom_factor)) {
        return;
    }

    // Determine LOD level using simplified zoom-out calculation
    LODLevel lod_level = calculateSimplifiedZoomOutLOD(cell_width_px, cell_height_px, zoom_factor);
    LODRenderSettings settings = getSimplifiedZoomOutRenderSettings(lod_level, zoom_factor);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply additional zoom-based alpha adjustment for zoomed-out views
    if (zoom_factor < 0.2f) {
        alpha_multiplier *= 0.6f;  // Reduce alpha significantly when heavily zoomed out
    } else if (zoom_factor < 0.4f) {
        alpha_multiplier *= 0.8f;  // Reduce alpha moderately when zoomed out
    }

    // Render heatmap/fill based on simplified zoom-out LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and zoom level
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        // When zoomed out significantly, use a simplified representation
        if (zoom_factor < 0.1f) {
            // At extreme zoom out, draw a smaller centered rectangle to reduce visual clutter
            ImVec2 center_p1 = ImVec2((p1.x + p2.x) * 0.5f - 2.0f, (p1.y + p2.y) * 0.5f - 2.0f);
            ImVec2 center_p2 = ImVec2((p1.x + p2.x) * 0.5f + 2.0f, (p1.y + p2.y) * 0.5f + 2.0f);
            draw_list->AddRectFilled(center_p1, center_p2, color_with_lod_alpha);
        } else {
            draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
        }
    }

    // Render borders based on simplified zoom-out LOD settings
    if (settings.render_borders) {
        // Use a simpler border approach when zoomed out
        unsigned char border_alpha = static_cast<unsigned char>(10 * alpha_multiplier);  // Reduced alpha for borders
        ImU32 border_color = IM_COL32(255, 255, 255, border_alpha);
        float thickness = settings.border_thickness;

        // Draw a simpler border when zoomed out
        if (zoom_factor < 0.1f) {
            // At extreme zoom out, use a minimal border
            ImVec2 center_p1 = ImVec2((p1.x + p2.x) * 0.5f - 2.5f, (p1.y + p2.y) * 0.5f - 2.5f);
            ImVec2 center_p2 = ImVec2((p1.x + p2.x) * 0.5f + 2.5f, (p1.y + p2.y) * 0.5f + 2.5f);
            draw_list->AddRect(center_p1, center_p2, border_color, 0.0f, 0, thickness * 0.5f);
        } else {
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }
}

// Implementation for intelligent zoom-based LOD that dynamically adjusts based on user interaction patterns
LODLevel FootprintLOD::calculateIntelligentZoomLOD(float cell_width_px, float cell_height_px,
                                                float zoom_factor, float time_spent_at_zoom) const {
    float min_dimension = std::min(cell_width_px, cell_height_px);

    // Calculate base LOD level
    LODLevel base_lod = calculateLODLevel(cell_width_px, cell_height_px, zoom_factor);

    // Adjust based on how long the user has spent at this zoom level
    // If user spends more time at a zoom level, they likely want more detail
    if (time_spent_at_zoom > 3.0f) {  // User has been at this zoom for more than 3 seconds
        switch (base_lod) {
            case LODLevel::LOW_DETAIL:
                if (zoom_factor > min_detail_zoom_ * 0.8f || min_dimension > min_cell_size_px_ * 0.8f) {
                    return LODLevel::MEDIUM_DETAIL;  // Increase detail for engaged viewing
                }
                break;
            case LODLevel::MEDIUM_DETAIL:
                if (zoom_factor > medium_detail_zoom_ * 0.9f || min_dimension > medium_cell_size_px_ * 0.9f) {
                    return LODLevel::HIGH_DETAIL;  // Increase detail for engaged viewing
                }
                break;
            default:
                return base_lod;  // Already high detail
        }
    }

    // If zooming rapidly (less than 0.5 seconds at current zoom), reduce detail for performance
    if (time_spent_at_zoom < 0.5f) {
        switch (base_lod) {
            case LODLevel::MAX_DETAIL:
                return LODLevel::HIGH_DETAIL;  // Reduce for performance during navigation
            case LODLevel::HIGH_DETAIL:
                return LODLevel::MEDIUM_DETAIL;  // Reduce for performance during navigation
            default:
                return base_lod;  // Keep as is
        }
    }

    return base_lod;
}

LODRenderSettings FootprintLOD::getIntelligentZoomRenderSettings(LODLevel lod_level,
                                                              float time_spent_at_zoom) const {
    LODRenderSettings settings = getRenderSettings(lod_level);

    // Adjust settings based on user engagement
    if (time_spent_at_zoom > 3.0f) {
        // User is engaged, enhance detail
        settings.render_detailed_annotations = true;
        settings.render_labels = true;
        settings.render_text = true;
        settings.alpha_multiplier = std::min(settings.alpha_multiplier * 1.1f, 1.2f);
        settings.border_thickness = std::max(settings.border_thickness * 1.1f, 1.8f);
    } else if (time_spent_at_zoom < 0.5f) {
        // User is navigating quickly, optimize for performance
        settings.render_detailed_annotations = false;
        if (lod_level == LODLevel::LOW_DETAIL) {
            settings.render_labels = false;
        }
        settings.alpha_multiplier = std::max(settings.alpha_multiplier * 0.9f, 0.7f);
    }

    return settings;
}

void FootprintLOD::applyIntelligentZoomLODToCell(const FootprintCell& cell,
                                               ImDrawList* draw_list,
                                               float zoom_factor,
                                               float time_spent_at_zoom,
                                               double max_volume,
                                               const std::vector<FootprintCell>& diagonal_imbalances,
                                               const std::vector<FootprintCell>& stacked_imbalances,
                                               const FootprintPanel* panel) const {
    // Calculate cell dimensions in pixels
    ImVec2 p1 = ImPlot::PlotToPixels(cell.x - cell.width * 0.48, cell.y - cell.height * 0.48);
    ImVec2 p2 = ImPlot::PlotToPixels(cell.x + cell.width * 0.48, cell.y + cell.height * 0.48);

    float cell_width_px = std::abs(p2.x - p1.x);
    float cell_height_px = std::abs(p2.y - p1.y);

    // Determine LOD level using intelligent zoom calculation
    LODLevel lod_level = calculateIntelligentZoomLOD(cell_width_px, cell_height_px, zoom_factor, time_spent_at_zoom);
    float cell_area_px = cell_width_px * cell_height_px;
    LODRenderSettings settings = getIntelligentZoomRenderSettings(lod_level, time_spent_at_zoom);

    // Get cell color from panel
    ImU32 cell_color = panel->getCellColor(cell, max_volume);

    // Get cell label from panel
    std::string cell_label = panel->getCellLabel(cell);

    // Apply alpha multiplier based on LOD
    float alpha_multiplier = calculateAlphaMultiplier(zoom_factor, lod_level);

    // Apply time-based alpha adjustment
    if (time_spent_at_zoom < 0.3f) {
        alpha_multiplier *= 0.8f;  // Reduce alpha during rapid zooming
    } else if (time_spent_at_zoom > 5.0f) {
        alpha_multiplier = std::min(alpha_multiplier * 1.15f, 1.3f);  // Increase alpha for engaged viewing
    }

    // Render heatmap/fill based on intelligent zoom LOD settings
    if (settings.render_heatmap) {
        // Modify alpha based on LOD and engagement time
        unsigned char original_alpha = (cell_color >> 24) & 0xFF;
        unsigned char new_alpha = static_cast<unsigned char>(original_alpha * alpha_multiplier);
        ImU32 color_with_lod_alpha = (cell_color & 0x00FFFFFF) | (new_alpha << 24);

        draw_list->AddRectFilled(p1, p2, color_with_lod_alpha);
    }

    // Render borders based on intelligent zoom LOD settings
    if (settings.render_borders) {
        // Check for imbalances to determine border color/type
        bool is_diagonal = false;
        bool is_stacked = false;

        for (const auto& diag_cell : diagonal_imbalances) {
            if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
                is_diagonal = true;
                break;
            }
        }

        for (const auto& stack_cell : stacked_imbalances) {
            if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
                is_stacked = true;
                break;
            }
        }

        ImU32 border_color;
        float thickness = settings.border_thickness;

        if (is_diagonal || is_stacked) {
            // Highlight imbalanced cells with special colors
            if (is_diagonal && is_stacked) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);

                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                ImVec2 offset_p1(p1.x - 1.5f, p1.y - 1.5f);
                ImVec2 offset_p2(p2.x + 1.5f, p2.y + 1.5f);
                draw_list->AddRect(offset_p1, offset_p2, border_color, 0.0f, 0, thickness * 0.8f);
            } else if (is_diagonal) {
                border_color = IM_COL32(255, 255, 0, 255);  // Yellow for diagonal
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            } else if (is_stacked) {
                border_color = IM_COL32(0, 255, 255, 255);  // Cyan for stacked
                draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
            }
        } else {
            // Regular border based on LOD
            unsigned char border_alpha = static_cast<unsigned char>(13 * alpha_multiplier);
            border_color = IM_COL32(255, 255, 255, border_alpha);
            draw_list->AddRect(p1, p2, border_color, 0.0f, 0, thickness);
        }
    }

    // Render text/labels based on intelligent zoom LOD settings
    if (settings.render_text && shouldRenderText(cell_height_px, zoom_factor)) {
        if (settings.render_labels && shouldRenderLabels(cell_height_px, zoom_factor)) {
            ImVec2 text_size = ImGui::CalcTextSize(cell_label.c_str());

            // Center text in cell
            ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f,
                           (p1.y + p2.y - text_size.y) * 0.5f);

            draw_list->AddText(text_pos, IM_COL32_WHITE, cell_label.c_str());
        }
    }

    // Render detailed annotations based on intelligent zoom LOD settings
    if (settings.render_detailed_annotations &&
        shouldRenderDetailedAnnotations(cell_width_px, cell_height_px, zoom_factor)) {

        // Example: render delta indicator if enabled and conditions met
        if (panel->getShowDeltaIndicator()) {
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            if (max_vol > 0.0) {
                double normalized_delta = cell.delta / max_vol;

                if (std::abs(normalized_delta) > panel->getDeltaThreshold()) {
                    float bar_height = (time_spent_at_zoom > 3.0f) ? 4.5f : 3.0f;  // Taller bars for engaged viewing
                    float bar_width = (p2.x - p1.x) * 0.8f;
                    ImVec2 bar_pos(p1.x + (p2.x - p1.x - bar_width) * 0.5f,
                                  p2.y - bar_height - (time_spent_at_zoom > 3.0f ? 1.8f : 1.0f));

                    ImU32 bar_color = normalized_delta > 0 ? IM_COL32(0, 255, 0, 200)   // Green
                                                           : IM_COL32(255, 0, 0, 200);  // Red

                    draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
                                           ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
                                           bar_color);
                }
            }
        }
    }
}

} // namespace Rendering
} // namespace BTQuant