#include "vulkan/lob_heatmap_compute_pipeline.h"
#include "shader_spirv.hpp"
#include <fstream>
#include <iostream>
#include <cstring>

namespace btq {
namespace vulkan {

LOBHeatmapComputePipeline::LOBHeatmapComputePipeline(VulkanCore* vulkan_core)
    : vulkan_core_(vulkan_core), pipeline_(VK_NULL_HANDLE), pipeline_layout_(VK_NULL_HANDLE),
      descriptor_set_layout_(VK_NULL_HANDLE), descriptor_set_(VK_NULL_HANDLE),
      output_image_(VK_NULL_HANDLE), output_image_view_(VK_NULL_HANDLE),
      output_image_memory_(VK_NULL_HANDLE),
      sampler_(VK_NULL_HANDLE), descriptor_pool_(VK_NULL_HANDLE) {
    current_params_ = {};
}

LOBHeatmapComputePipeline::~LOBHeatmapComputePipeline() {
    destroy();
}

void LOBHeatmapComputePipeline::destroy() {
    if (vulkan_core_ && vulkan_core_->get_device() != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkan_core_->get_device(), pipeline_, nullptr);
        vkDestroyPipelineLayout(vulkan_core_->get_device(), pipeline_layout_, nullptr);
        vkDestroyDescriptorSetLayout(vulkan_core_->get_device(), descriptor_set_layout_, nullptr);
        
        if (output_image_ != VK_NULL_HANDLE) {
            // Use the memory manager to destroy the image
            BTQuant::ImageAllocation allocation;
            allocation.image = output_image_;
            allocation.memory = output_image_memory_;
            allocation.view = output_image_view_;
            vulkan_core_->get_memory_manager().deallocate_image(allocation);
            
            output_image_ = VK_NULL_HANDLE;
            output_image_memory_ = VK_NULL_HANDLE;
            output_image_view_ = VK_NULL_HANDLE;
        }
        
        if (output_image_view_ != VK_NULL_HANDLE) {
            vkDestroyImageView(vulkan_core_->get_device(), output_image_view_, nullptr);
            output_image_view_ = VK_NULL_HANDLE;
        }
        
        if (sampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(vulkan_core_->get_device(), sampler_, nullptr);
            sampler_ = VK_NULL_HANDLE;
        }
        
        if (descriptor_pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(vulkan_core_->get_device(), descriptor_pool_, nullptr);
            descriptor_pool_ = VK_NULL_HANDLE;
        }
    }
}

void LOBHeatmapComputePipeline::initialize(uint32_t width, uint32_t height) {
    create_descriptor_set_layout();
    create_pipeline_layout();
    create_compute_pipeline();
    create_output_image(width, height);
    create_sampler();
    create_descriptor_pool();
    create_descriptor_sets();
}

void LOBHeatmapComputePipeline::create_descriptor_set_layout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);

    // Binding 0: SSBO for order book snapshots
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = nullptr;

    // Binding 1: Storage image for heatmap output
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].pImmutableSamplers = nullptr;

    // Binding 2: SSBO for atomic depth buffer
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(vulkan_core_->get_device(), &layout_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout for LOB heatmap pipeline");
    }
}

void LOBHeatmapComputePipeline::create_pipeline_layout() {
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;

    // Push constant for heatmap parameters
    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = sizeof(HeatmapParams);

    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;

    if (vkCreatePipelineLayout(vulkan_core_->get_device(), &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout for LOB heatmap pipeline");
    }
}

void LOBHeatmapComputePipeline::create_compute_pipeline() {
    // Load the SPIR-V code for the compute shader
    auto compute_shader_code = load_shader_spirv("dependencies/BTQ_Render_Engine/shaders/spirv/lob_heatmap.spv");

    if (compute_shader_code.empty()) {
        // Try alternative path
        compute_shader_code = load_shader_spirv("shaders/spirv/lob_heatmap.spv");
    }

    if (compute_shader_code.empty()) {
        // If compiled SPIR-V doesn't exist, we'll need to compile the shader
        std::cerr << "Warning: Could not load compiled shader from expected paths." << std::endl;
        std::cerr << "Expected paths: dependencies/BTQ_Render_Engine/shaders/spirv/lob_heatmap.spv or shaders/spirv/lob_heatmap.spv" << std::endl;
        
        // For now, we'll throw an error to indicate the need for shader compilation
        throw std::runtime_error("LOB heatmap compute shader SPIR-V not found. Please ensure the shader is compiled.");
    }

    VkShaderModuleCreateInfo shader_module_create_info{};
    shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_create_info.codeSize = compute_shader_code.size() * sizeof(uint32_t);
    shader_module_create_info.pCode = compute_shader_code.data();

    VkShaderModule compute_shader_module;
    if (vkCreateShaderModule(vulkan_core_->get_device(), &shader_module_create_info, nullptr, &compute_shader_module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute shader module");
    }

    VkPipelineShaderStageCreateInfo shader_stage_create_info{};
    shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_create_info.module = compute_shader_module;
    shader_stage_create_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_create_info{};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.layout = pipeline_layout_;
    pipeline_create_info.stage = shader_stage_create_info;

    if (vkCreateComputePipelines(vulkan_core_->get_device(), VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline for LOB heatmap");
    }

    vkDestroyShaderModule(vulkan_core_->get_device(), compute_shader_module, nullptr);
}

void LOBHeatmapComputePipeline::create_output_image(uint32_t width, uint32_t height) {
    // Use the memory manager to create the image
    BTQuant::ImageAllocation allocation = vulkan_core_->get_memory_manager().allocate_image(
        width, 
        height, 
        VK_FORMAT_R32G32B32A32_SFLOAT, // RGBA32F for high precision color data
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    if (allocation.image == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to create output image for LOB heatmap");
    }

    output_image_ = allocation.image;
    output_image_memory_ = allocation.memory;
    output_image_view_ = allocation.view; // This will be set by allocate_image
    // Note: We don't store output_image_allocation_ since we're using GPUMemoryManager
}

// create_image_view() is not needed since GPUMemoryManager::allocate_image() creates the view

void LOBHeatmapComputePipeline::create_sampler() {
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sampler_info.unnormalizedCoordinates = VK_TRUE; // Use pixel coordinates
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 1.0f;
    sampler_info.anisotropyEnable = VK_FALSE;

    if (vkCreateSampler(vulkan_core_->get_device(), &sampler_info, nullptr, &sampler_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler for LOB heatmap output");
    }
}

void LOBHeatmapComputePipeline::create_descriptor_pool() {
    std::vector<VkDescriptorPoolSize> pool_sizes(3);
    
    // Storage buffer for order book snapshots
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 1;
    
    // Storage image for heatmap output
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = 1;
    
    // Storage buffer for atomic depth buffer
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[2].descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = 1;

    if (vkCreateDescriptorPool(vulkan_core_->get_device(), &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool for LOB heatmap pipeline");
    }
}

void LOBHeatmapComputePipeline::create_descriptor_sets() {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout_;

    if (vkAllocateDescriptorSets(vulkan_core_->get_device(), &alloc_info, &descriptor_set_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set for LOB heatmap pipeline");
    }

    // Update descriptor sets later when actual buffers are bound
}

void LOBHeatmapComputePipeline::update_parameters(uint32_t cols, uint32_t rows, float max_liquidity) {
    current_params_.width = cols;
    current_params_.height = rows;
    current_params_.maxLiquidity = max_liquidity;
    current_params_.timeOffset = 0.0f; // Default time offset
    current_params_.invertYAxis = true; // Default to inverting Y axis
}

void LOBHeatmapComputePipeline::dispatch(VkCommandBuffer command_buffer, uint32_t width, uint32_t height) {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    
    // Push heatmap parameters
    vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HeatmapParams), &current_params_);
    
    // Dispatch the compute shader
    // Using 1x64 local workgroup size as specified in the shader
    uint32_t group_count_x = 1;
    uint32_t group_count_y = (height + 63) / 64;  // Round up to nearest multiple of 64
    uint32_t group_count_z = 1;
    
    vkCmdDispatch(command_buffer, group_count_x, group_count_y, group_count_z);
    
    // Memory barrier to ensure the storage image is ready for use
    VkMemoryBarrier memory_barrier{};
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.pNext = nullptr;
    memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        1,
        &memory_barrier,
        0,
        nullptr,
        0,
        nullptr
    );
}

std::vector<uint32_t> LOBHeatmapComputePipeline::load_shader_spirv(const std::string& filename) {
    // This is a simplified version - in practice, you'd want to load the actual compiled SPIR-V
    // For now, we'll simulate loading by returning an empty vector to trigger the fallback
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Could not open shader file: " << filename << std::endl;
        return std::vector<uint32_t>();
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

} // namespace vulkan
} // namespace btq