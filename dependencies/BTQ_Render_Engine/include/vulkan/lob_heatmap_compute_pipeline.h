#pragma once

#include "vulkan_base_types.hpp"
#include <memory>
#include <vector>

// Forward declaration
struct VmaAllocation_T;
typedef struct VmaAllocation_T* VmaAllocation;

namespace btq {
namespace vulkan {

// Define VulkanCore as an alias to the actual class in BTQuant namespace
using VulkanCore = BTQuant::VulkanCore;

struct HeatmapParams {
    uint32_t width;
    uint32_t height;
    float maxLiquidity;
    float timeOffset;
    bool invertYAxis;
};

class LOBHeatmapComputePipeline {
public:
    explicit LOBHeatmapComputePipeline(VulkanCore* vulkan_core);
    ~LOBHeatmapComputePipeline();

    void initialize(uint32_t width, uint32_t height);
    void destroy();

    void update_parameters(uint32_t cols, uint32_t rows, float max_liquidity);
    void dispatch(VkCommandBuffer command_buffer, uint32_t width, uint32_t height);

    VkDeviceMemory get_output_image_memory() const { return output_image_memory_; }
    VkImage get_output_image() const { return output_image_; }
    VkImageView get_output_image_view() const { return output_image_view_; }
    VkSampler get_output_sampler() const { return sampler_; }
    // Note: VmaAllocation is not exposed since we use GPUMemoryManager

private:
    VulkanCore* vulkan_core_;
    
    // Pipeline objects
    VkPipeline pipeline_;
    VkPipelineLayout pipeline_layout_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkDescriptorSet descriptor_set_;
    
    // Output image resources
    VkImage output_image_;
    VkImageView output_image_view_;
    VkDeviceMemory output_image_memory_;
    VkSampler sampler_;
    
    // Descriptor pool
    VkDescriptorPool descriptor_pool_;
    
    // Current parameters
    HeatmapParams current_params_;
    
    void create_descriptor_set_layout();
    void create_pipeline_layout();
    void create_compute_pipeline();
    void create_output_image(uint32_t width, uint32_t height);
    void create_image_view();
    void create_sampler();
    void create_descriptor_pool();
    void create_descriptor_sets();
    
    std::vector<uint32_t> load_shader_spirv(const std::string& filename);
};

} // namespace vulkan
} // namespace btq