#include "rendering/footprint_lod.hpp"

#include <algorithm>
#include <cmath>

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

    float prev_lod_level = calculateLODLevel(cell_width_px, cell_height_px, prev_zoom);
    float curr_lod_level = calculateLODLevel(cell_width_px, cell_height_px, curr_zoom);

    state.from_lod = static_cast<LODLevel>(static_cast<int>(prev_lod_level));
    state.to_lod = static_cast<LODLevel>(static_cast<int>(curr_lod_level));

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
    LODRenderSettings settings = getRenderSettings(lod_level);

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
    LODRenderSettings settings = getRenderSettings(lod_level);

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

} // namespace Rendering
} // namespace BTQuant