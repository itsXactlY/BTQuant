// Phase 1 LOB Heatmap Compute Pipeline — STUB
// Real implementation deferred. The compute shader SPIR-V is embedded in
// <shader_spirv.hpp> and the class declaration is part of the Phase 1
// interface contract. This stub satisfies the linker so the dashboard
// builds without GPU compute wired up.

#include "vulkan/lob_heatmap_compute_pipeline.hpp"

#include <cstring>
#include <iostream>

namespace BTQuant {

void LobHeatmapComputePipeline::initialize(VkDevice device,
                                           VkPhysicalDevice physical_device,
                                           VkDescriptorPool pool,
                                           GPUMemoryManager& mem) {
    device_ = device;
    // In production:
    //   1. Create compute shader module from LOB_HEATMAP_COMPUTE_SPIRV
    //   2. Create descriptor set layout (SSBO @ 0, storage image @ 1)
    //   3. Create compute pipeline
    //   4. create_output_image(device, physical_device) → 1024x256 rgba16f
    //   5. mem.add_texture(...) wraps ImGui_ImplVulkan_AddTexture (the spec
    //      needs GPUMemoryManager::add_texture() — not yet present, so the
    //      stub leaves imgui_texture_ default-constructed).
    // STUB: log and return. The pipeline handles a no-op render path.
    std::cerr << "[LobHeatmapComputePipeline] STUB: initialize() "
              << "— Phase 1 compute not yet wired. Image stays default."
              << std::endl;
}

void LobHeatmapComputePipeline::dispatch(VkCommandBuffer /*cmd*/,
                                         uint32_t /*columns*/,
                                         uint32_t /*rows*/) {
    // STUB: no-op. In production: vkCmdBindPipeline, bind descriptor set,
    // vkCmdPushConstants (HeatmapPushConstants{ max_volume, alpha }),
    // vkCmdDispatch(columns/16, rows/16, 1).
}

void LobHeatmapComputePipeline::transition_to_general(VkCommandBuffer /*cmd*/) {
    // STUB: no-op. Real implementation inserts VkImageMemoryBarrier
    // SHADER_READ_ONLY_OPTIMAL → GENERAL with src/dst access masks per spec 1.5.
}

void LobHeatmapComputePipeline::transition_to_shader_read(VkCommandBuffer /*cmd*/) {
    // STUB: no-op. Reverse of transition_to_general per spec 1.5.
}

void LobHeatmapComputePipeline::set_push_constants(float max_volume, float alpha) {
    push_constants_.max_volume = max_volume;
    push_constants_.alpha      = alpha;
}

void LobHeatmapComputePipeline::bind_ssbo(const BufferAllocation& ssbo_alloc) {
    ssbo_allocation_ = ssbo_alloc;
}

void LobHeatmapComputePipeline::create_output_image(VkDevice /*device*/,
                                                    VkPhysicalDevice /*physical_device*/) {
    // STUB: no-op. Real implementation: vkCreateImage (1024x256 rgba16f),
    // allocate via MemoryPool, vkBindImageMemory, vkCreateImageView,
    // vkCreateSampler (linear filtering, clamp-to-edge).
}

void LobHeatmapComputePipeline::destroy(VkDevice /*device*/) {
    // STUB: real implementation tears down image/view/sampler/pipeline/layout.
    output_image_         = VK_NULL_HANDLE;
    output_image_view_    = VK_NULL_HANDLE;
    output_sampler_       = VK_NULL_HANDLE;
    pipeline_             = VK_NULL_HANDLE;
    pipeline_layout_      = VK_NULL_HANDLE;
    descriptor_set_layout_= VK_NULL_HANDLE;
    descriptor_set_       = VK_NULL_HANDLE;
}

}  // namespace BTQuant
