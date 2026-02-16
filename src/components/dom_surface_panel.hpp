#pragma once

/**
 * @file dom_surface_panel.hpp
 * @brief Liquidity Surface panel with GPU-accelerated rendering
 *
 * This module provides a transparent ImGui panel that displays liquidity data
 * in a 5-column table format while rendering GPUMemoryManager textures
 * in the background.
 *
 * Table columns: [Buys | Asks | Price | Bids | Sells]
 */

#include <vector>

// Forward declaration
namespace BTQuant {
class GPUMemoryManager;
}

namespace pubbtquant::components {

/**
 * @brief Renders the Liquidity Surface panel with transparent background
 * and GPUMemoryManager texture displayed behind a 5-column ImGuiTable.
 *
 * @param gpu_memory_manager Reference to GPUMemoryManager for texture access
 * @param buys Vector of buy volumes
 * @param asks Vector of ask volumes
 * @param prices Vector of price levels
 * @param bids Vector of bid volumes
 * @param sells Vector of sell volumes
 */
void RenderLiquiditySurface(
    BTQuant::GPUMemoryManager& gpu_memory_manager,
    const std::vector<double>& buys,
    const std::vector<double>& asks,
    const std::vector<double>& prices,
    const std::vector<double>& bids,
    const std::vector<double>& sells);

/**
 * @brief Renders a liquidity surface panel with integrated GPU texture display.
 *
 * @param gpu_memory_manager Pointer to GPUMemoryManager instance
 * @param width Panel width in pixels (0 = auto-fit)
 * @param height Panel height in pixels (0 = auto-fit)
 */
void RenderLiquiditySurfacePanel(
    BTQuant::GPUMemoryManager* gpu_memory_manager,
    float width = 0.0f,
    float height = 0.0f);

}  // namespace pubbtquant::components
