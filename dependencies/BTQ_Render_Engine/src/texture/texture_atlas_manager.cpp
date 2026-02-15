#include "../../include/texture/texture_atlas_manager.hpp"
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"

namespace BTQuant {

TextureAtlasManager::TextureAtlasManager(GPUMemoryManager& memory_manager)
    : memory_manager_(memory_manager), 
      icon_width_(0), 
      icon_height_(0), 
      atlas_cols_(0), 
      atlas_rows_(0) {
}

void TextureAtlasManager::initializeExchangeIconAtlas(const std::vector<std::vector<uint8_t>>& icon_data,
                                                    uint32_t icon_width, uint32_t icon_height,
                                                    const std::vector<std::string>& exchange_names) {
    if (icon_data.empty() || exchange_names.empty() || icon_data.size() != exchange_names.size()) {
        return; // Invalid input
    }

    icon_width_ = icon_width;
    icon_height_ = icon_height;

    // Calculate grid dimensions (square layout)
    size_t num_icons = icon_data.size();
    atlas_cols_ = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(num_icons))));
    atlas_rows_ = static_cast<uint32_t>(std::ceil(static_cast<float>(num_icons) / atlas_cols_));

    // Create the texture atlas
    exchange_icon_atlas_ = memory_manager_.create_texture_atlas(icon_data, icon_width_, icon_height_, atlas_cols_, atlas_rows_);

    // Populate the UV coordinate map
    exchange_uv_map_.clear();
    for (size_t i = 0; i < exchange_names.size(); ++i) {
        uint32_t col = i % atlas_cols_;
        uint32_t row = i / atlas_cols_;

        // Calculate UV coordinates for this icon in the atlas
        float u_min = static_cast<float>(col * icon_width_) / static_cast<float>(exchange_icon_atlas_.width);
        float v_min = static_cast<float>(row * icon_height_) / static_cast<float>(exchange_icon_atlas_.height);
        float u_max = static_cast<float>((col + 1) * icon_width_) / static_cast<float>(exchange_icon_atlas_.width);
        float v_max = static_cast<float>((row + 1) * icon_height_) / static_cast<float>(exchange_icon_atlas_.height);

        exchange_uv_map_[exchange_names[i]] = {u_min, v_min, u_max, v_max};
    }
}

std::array<float, 4> TextureAtlasManager::getExchangeIconUV(const std::string& exchange_name) const {
    auto it = exchange_uv_map_.find(exchange_name);
    if (it != exchange_uv_map_.end()) {
        return it->second;
    }
    
    // Return default UV coordinates if exchange not found
    return {0.0f, 0.0f, 1.0f, 1.0f};
}

ImTextureID TextureAtlasManager::getImGuiTextureID() const {
    // In a real implementation, we would need to register the texture with ImGui's descriptor system
    // This requires access to the descriptor pool and proper descriptor creation
    // For now, we return a placeholder that would be replaced with the actual implementation
    
    // The actual implementation would create a descriptor set for the texture
    // and return the appropriate ImTextureID that ImGui can use
    // This is typically done through ImGui_ImplVulkan_AddTexture function
    return reinterpret_cast<ImTextureID>(exchange_icon_atlas_.view);
}

} // namespace BTQuant