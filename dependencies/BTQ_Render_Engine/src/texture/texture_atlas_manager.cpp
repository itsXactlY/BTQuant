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

    // Store icon data for later upload
    icon_data_ = icon_data;
    exchange_names_ = exchange_names;

    // Create the texture atlas (this creates the image but doesn't upload data yet)
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

ImTextureID TextureAtlasManager::getImGuiTextureID(VulkanCore* vulkan_core) const {
    if (!vulkan_core) {
        // If no VulkanCore provided, return the raw image view as fallback
        return reinterpret_cast<ImTextureID>(exchange_icon_atlas_.view);
    }
    
    // Register the texture with ImGui using ImGui_ImplVulkan_AddTexture
    // This creates an ImTextureID that can be used with ImGui::Image
    VkSampler texture_sampler = vulkan_core->get_default_sampler();
    
    ImTextureID texture_id = ImGui_ImplVulkan_AddTexture(
        texture_sampler,
        exchange_icon_atlas_.view,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
    
    return texture_id;
}

void TextureAtlasManager::uploadTextureData(VulkanCore* vulkan_core) {
    if (!vulkan_core || icon_data_.empty()) {
        return; // Nothing to upload
    }

    // Calculate total size needed for the atlas (assuming RGBA format)
    uint32_t atlas_width = icon_width_ * atlas_cols_;
    uint32_t atlas_height = icon_height_ * atlas_rows_;
    VkDeviceSize image_size = atlas_width * atlas_height * 4; // 4 bytes per pixel (RGBA)

    // Create staging buffer to transfer image data
    BufferAllocation staging_buffer = memory_manager_.allocate_staging_buffer(image_size);

    // Copy icon data to staging buffer in atlas layout
    uint8_t* data_ptr = static_cast<uint8_t*>(staging_buffer.mapped_ptr);

    // Initialize the entire atlas to transparent black
    memset(data_ptr, 0, image_size);

    // Place each icon in its grid position
    for (size_t i = 0; i < icon_data_.size(); ++i) {
        if (i >= atlas_cols_ * atlas_rows_) break; // Don't exceed atlas capacity

        uint32_t col = i % atlas_cols_;
        uint32_t row = i / atlas_cols_;

        uint32_t dest_x = col * icon_width_;
        uint32_t dest_y = row * icon_height_;

        // Copy each row of the icon to the appropriate position in the atlas
        for (uint32_t y = 0; y < icon_height_; ++y) {
            uint32_t src_offset = y * icon_width_ * 4; // 4 bytes per pixel
            uint32_t dst_row_start = ((dest_y + y) * atlas_width + dest_x) * 4;

            if (src_offset + (icon_width_ * 4) <= icon_data_[i].size()) {
                memcpy(&data_ptr[dst_row_start], &icon_data_[i][src_offset], icon_width_ * 4);
            }
        }
    }

    // Get a command buffer from the VulkanCore for the transfer
    VkCommandBuffer command_buffer = vulkan_core->begin_single_time_commands();
    
    // Perform the buffer to image copy using the GPUMemoryManager's method
    VkResult result = memory_manager_.copy_buffer_to_image(command_buffer, staging_buffer.buffer, exchange_icon_atlas_.image, atlas_width, atlas_height);
    
    // End the single-time commands
    vulkan_core->end_single_time_commands(command_buffer);

    // Clean up staging buffer
    memory_manager_.deallocate_buffer(staging_buffer);
}

} // namespace BTQuant