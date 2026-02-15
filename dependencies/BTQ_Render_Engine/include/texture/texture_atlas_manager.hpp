#pragma once

#include "../include/vulkan_base_types.hpp"
#include <unordered_map>
#include <string>
#include <vector>

// Forward declaration or include for ImTextureID
#include "imgui.h"

namespace BTQuant {

/**
 * @brief Manages texture atlases for exchange icons and other UI elements
 */
class TextureAtlasManager {
public:
    /**
     * @brief Constructor
     * @param memory_manager Reference to the GPU memory manager
     */
    TextureAtlasManager(GPUMemoryManager& memory_manager);

    /**
     * @brief Initialize the exchange icon texture atlas
     * @param icon_data Vector of icon pixel data (RGBA format)
     * @param icon_width Width of each individual icon
     * @param icon_height Height of each individual icon
     * @param exchange_names Names of exchanges corresponding to each icon
     */
    void initializeExchangeIconAtlas(const std::vector<std::vector<uint8_t>>& icon_data,
                                    uint32_t icon_width, uint32_t icon_height,
                                    const std::vector<std::string>& exchange_names);

    /**
     * @brief Get the texture atlas allocation for exchange icons
     * @return ImageAllocation containing the atlas
     */
    const ImageAllocation& getExchangeIconAtlas() const { return exchange_icon_atlas_; }

    /**
     * @brief Get UV coordinates for a specific exchange icon
     * @param exchange_name Name of the exchange
     * @return UV coordinates {u_min, v_min, u_max, v_max}
     */
    std::array<float, 4> getExchangeIconUV(const std::string& exchange_name) const;

    /**
     * @brief Get the ImTextureID for ImGui rendering
     * @return ImTextureID for use with ImGui::Image
     */
    ImTextureID getImGuiTextureID() const;

private:
    GPUMemoryManager& memory_manager_;
    ImageAllocation exchange_icon_atlas_;
    std::unordered_map<std::string, std::array<float, 4>> exchange_uv_map_; // Maps exchange name to UV coordinates
    uint32_t icon_width_;
    uint32_t icon_height_;
    uint32_t atlas_cols_;
    uint32_t atlas_rows_;
};

} // namespace BTQuant