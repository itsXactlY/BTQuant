#include "heatmap_compute.hpp"

#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include "embedded_shaders.hpp"

namespace btquant::renderer {

namespace {

uint32_t findMemoryType(VkPhysicalDevice physicalDevice,
                         uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("HeatmapCompute: no suitable memory type");
}

struct ConfigUbo {
    float price_low;
    float price_high;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_trades;
};

}  // namespace

HeatmapCompute::~HeatmapCompute() { shutdown(); }

std::optional<std::string>
HeatmapCompute::initialize(VkDevice device,
                            VkPhysicalDevice physicalDevice,
                            VkCommandPool commandPool,
                            VkQueue computeQueue,
                            uint32_t computeQueueFamily,
                            const HeatmapConfig& cfg) {
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_commandPool = commandPool;
    m_computeQueue = computeQueue;
    m_computeQueueFamily = computeQueueFamily;
    m_cfg = cfg;

    if (auto err = createInputBuffer()) return err;
    if (auto err = createOutputImage()) return err;
    if (auto err = createConfigBuffer()) return err;
    if (auto err = buildDescriptorSetLayout()) return err;
    if (auto err = buildComputePipeline()) return err;
    if (auto err = buildDescriptorSet()) return err;

    // Register with ImGui_ImplVulkan for sampling in the main render pass.
    // Image layout must be SHADER_READ_ONLY_OPTIMAL after this — ImGui takes
    // care of the transition internally when binding the descriptor set.
    // (New ImGui API: ImGui_ImplVulkan_AddTexture returns VkDescriptorSet;
    // ImTextureID is typedef'd to ImU64 — reinterpret_cast between them.)
    m_imguiTextureId = reinterpret_cast<ImTextureID>(
        ImGui_ImplVulkan_AddTexture(
            m_outputSampler, m_outputView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

    std::printf("[HeatmapCompute] ready: %ux%u RGBA8 storage image, %u trade capacity\n",
                m_cfg.image_width, m_cfg.image_height,
                static_cast<uint32_t>(m_inputCapacity / sizeof(TradeInput)));
    return std::nullopt;
}

std::optional<std::string> HeatmapCompute::createInputBuffer() {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = m_inputCapacity;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_device, &info, nullptr, &m_inputBuffer) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateBuffer(input) failed";
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(m_device, m_inputBuffer, &req);
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemoryType(
        m_physicalDevice, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(m_device, &alloc, nullptr, &m_inputMemory) != VK_SUCCESS) {
        return "HeatmapCompute: vkAllocateMemory(input) failed";
    }
    if (vkBindBufferMemory(m_device, m_inputBuffer, m_inputMemory, 0) != VK_SUCCESS) {
        return "HeatmapCompute: vkBindBufferMemory(input) failed";
    }
    return std::nullopt;
}

std::optional<std::string> HeatmapCompute::createOutputImage() {
    // R8G8B8A8_UNORM storage image that doubles as a sampled image for ImGui.
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {m_cfg.image_width, m_cfg.image_height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(m_device, &info, nullptr, &m_outputImage) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateImage(output) failed";
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(m_device, m_outputImage, &req);
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemoryType(
        m_physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(m_device, &alloc, nullptr, &m_outputMemory) != VK_SUCCESS) {
        return "HeatmapCompute: vkAllocateMemory(output) failed";
    }
    if (vkBindImageMemory(m_device, m_outputImage, m_outputMemory, 0) != VK_SUCCESS) {
        return "HeatmapCompute: vkBindImageMemory(output) failed";
    }

    // Image view.
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_outputImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(m_device, &viewInfo, nullptr, &m_outputView) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateImageView failed";
    }

    // Linear sampler — ImGui draws textures 1:1, no mipmap, no anisotropy.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_outputSampler) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateSampler failed";
    }

    // Transition UNDEFINED → SHADER_READ_ONLY_OPTIMAL so ImGui can sample.
    // (We use a one-shot command buffer for the transition.)
    VkCommandBufferAllocateInfo cbAlloc{};
    cbAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAlloc.commandPool = m_commandPool;
    cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;
    VkCommandBuffer cmd{};
    vkAllocateCommandBuffers(m_device, &cbAlloc, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_outputImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(m_computeQueue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_computeQueue);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);

    return std::nullopt;
}

std::optional<std::string> HeatmapCompute::createConfigBuffer() {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(ConfigUbo);
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_device, &info, nullptr, &m_configBuffer) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateBuffer(config) failed";
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(m_device, m_configBuffer, &req);
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemoryType(
        m_physicalDevice, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(m_device, &alloc, nullptr, &m_configMemory) != VK_SUCCESS) {
        return "HeatmapCompute: vkAllocateMemory(config) failed";
    }
    if (vkBindBufferMemory(m_device, m_configBuffer, m_configMemory, 0) != VK_SUCCESS) {
        return "HeatmapCompute: vkBindBufferMemory(config) failed";
    }
    return std::nullopt;
}

std::optional<std::string> HeatmapCompute::buildDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 3;
    info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_device, &info, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateDescriptorSetLayout failed";
    }
    return std::nullopt;
}

std::optional<std::string> HeatmapCompute::buildComputePipeline() {
    // Shader module from embedded SPIR-V.
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = embedded_shaders::HEATMAP_COMP_SPIRV_WORDS * sizeof(uint32_t);
    sm.pCode = embedded_shaders::HEATMAP_COMP_SPIRV;
    if (vkCreateShaderModule(m_device, &sm, nullptr, &m_shaderModule) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateShaderModule failed";
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = m_shaderModule;
    stage.pName = "main";

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_descriptorSetLayout;
    if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreatePipelineLayout failed";
    }

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.layout = m_pipelineLayout;
    pipeInfo.stage = stage;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &m_pipeline) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateComputePipelines failed";
    }
    return std::nullopt;
}

std::optional<std::string> HeatmapCompute::buildDescriptorSet() {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[0].descriptorCount = 1;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        return "HeatmapCompute: vkCreateDescriptorPool failed";
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
        return "HeatmapCompute: vkAllocateDescriptorSets failed";
    }

    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = m_inputBuffer;
    bufInfo.offset = 0;
    bufInfo.range = m_inputCapacity;

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler = VK_NULL_HANDLE;
    imgInfo.imageView = m_outputView;  // image view, not the raw image handle
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo cfgInfo{};
    cfgInfo.buffer = m_configBuffer;
    cfgInfo.offset = 0;
    cfgInfo.range = sizeof(ConfigUbo);

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &bufInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &imgInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = m_descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].pBufferInfo = &cfgInfo;

    vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
    return std::nullopt;
}

void HeatmapCompute::updateTrades(const TradeInput* trades, uint32_t count) {
    if (!m_inputBuffer || count == 0) return;
    const uint32_t maxCount = static_cast<uint32_t>(m_inputCapacity / sizeof(TradeInput));
    if (count > maxCount) count = maxCount;
    m_lastDispatchedCount = count;

    void* mapped = nullptr;
    if (vkMapMemory(m_device, m_inputMemory, 0, count * sizeof(TradeInput), 0, &mapped) != VK_SUCCESS) {
        return;
    }
    std::memcpy(mapped, trades, count * sizeof(TradeInput));
    vkUnmapMemory(m_device, m_inputMemory);

    // Update config UBO with current count.
    ConfigUbo cfg{};
    cfg.price_low = m_cfg.price_low;
    cfg.price_high = m_cfg.price_high;
    cfg.image_width = m_cfg.image_width;
    cfg.image_height = m_cfg.image_height;
    cfg.num_trades = count;
    void* cfgMapped = nullptr;
    if (vkMapMemory(m_device, m_configMemory, 0, sizeof(ConfigUbo), 0, &cfgMapped) == VK_SUCCESS) {
        std::memcpy(cfgMapped, &cfg, sizeof(ConfigUbo));
        vkUnmapMemory(m_device, m_configMemory);
    }
}

void HeatmapCompute::dispatch(VkCommandBuffer cmd) {
    if (m_lastDispatchedCount == 0) return;

    // Image layout transition: SHADER_READ_ONLY_OPTIMAL (after last sample)
    //  → GENERAL (for compute write) before dispatch.
    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = m_outputImage;
    toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneral.subresourceRange.levelCount = 1;
    toGeneral.subresourceRange.layerCount = 1;
    toGeneral.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    const uint32_t gx = (m_cfg.image_width + 15) / 16;
    const uint32_t gy = (m_cfg.image_height + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Transition GENERAL → SHADER_READ_ONLY_OPTIMAL for the next ImGui sample.
    VkImageMemoryBarrier toRead{};
    toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.image = m_outputImage;
    toRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toRead.subresourceRange.levelCount = 1;
    toRead.subresourceRange.layerCount = 1;
    toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toRead);
}

void HeatmapCompute::shutdown() {
    if (m_device == VK_NULL_HANDLE) return;
    destroySizeDependentResources();
    if (m_pipeline) vkDestroyPipeline(m_device, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    if (m_shaderModule) vkDestroyShaderModule(m_device, m_shaderModule, nullptr);
    if (m_descriptorPool) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    if (m_descriptorSetLayout) vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    if (m_configBuffer) vkDestroyBuffer(m_device, m_configBuffer, nullptr);
    if (m_configMemory) vkFreeMemory(m_device, m_configMemory, nullptr);
    if (m_inputBuffer) vkDestroyBuffer(m_device, m_inputBuffer, nullptr);
    if (m_inputMemory) vkFreeMemory(m_device, m_inputMemory, nullptr);
    m_pipeline = VK_NULL_HANDLE;
    m_pipelineLayout = VK_NULL_HANDLE;
    m_shaderModule = VK_NULL_HANDLE;
    m_descriptorPool = VK_NULL_HANDLE;
    m_descriptorSetLayout = VK_NULL_HANDLE;
    m_descriptorSet = VK_NULL_HANDLE;
    m_configBuffer = VK_NULL_HANDLE;
    m_configMemory = VK_NULL_HANDLE;
    m_inputBuffer = VK_NULL_HANDLE;
    m_inputMemory = VK_NULL_HANDLE;
    m_outputSampler = VK_NULL_HANDLE;
    m_outputView = VK_NULL_HANDLE;
    m_outputImage = VK_NULL_HANDLE;
    m_outputMemory = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

void HeatmapCompute::destroySizeDependentResources() {
    if (m_device == VK_NULL_HANDLE) {
        // Nothing to free — initialize() never ran or already cleaned up.
        m_imguiTextureId = 0;
        return;
    }
    if (m_imguiTextureId) {
        ImGui_ImplVulkan_RemoveTexture(
            reinterpret_cast<VkDescriptorSet>(m_imguiTextureId));
        m_imguiTextureId = 0;
    }
    if (m_outputSampler) vkDestroySampler(m_device, m_outputSampler, nullptr);
    if (m_outputView) vkDestroyImageView(m_device, m_outputView, nullptr);
    if (m_outputImage) vkDestroyImage(m_device, m_outputImage, nullptr);
    if (m_outputMemory) vkFreeMemory(m_device, m_outputMemory, nullptr);
    m_outputSampler = VK_NULL_HANDLE;
    m_outputView = VK_NULL_HANDLE;
    m_outputImage = VK_NULL_HANDLE;
    m_outputMemory = VK_NULL_HANDLE;

    // Free the descriptor set (it references the destroyed view). Reset the
    // pool so subsequent buildDescriptorSet() can allocate a fresh set that
    // points at the new view.
    if (m_descriptorSet != VK_NULL_HANDLE && m_descriptorPool != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(m_device, m_descriptorPool, 1, &m_descriptorSet);
        m_descriptorSet = VK_NULL_HANDLE;
    }
}

void HeatmapCompute::setSize(uint32_t newSize) {
    if (newSize == 0) newSize = 1;  // floor
    if (newSize == m_cfg.image_width) return;  // no-op fast path
    if (m_device == VK_NULL_HANDLE) {
        // Not initialized yet — defer until initialize() is called.
        m_cfg.image_width = newSize;
        m_cfg.image_height = newSize;
        return;
    }

    // Tear down size-dependent Vulkan resources (image, view, memory, sampler,
    // ImGui texture registration, descriptor set).
    destroySizeDependentResources();

    // Update config and rebuild.
    m_cfg.image_width = newSize;
    m_cfg.image_height = newSize;

    if (auto err = createOutputImage()) {
        std::fprintf(stderr, "[HeatmapCompute] setSize(%u) failed at createOutputImage: %s\n",
                     newSize, err->c_str());
        return;
    }
    m_imguiTextureId = reinterpret_cast<ImTextureID>(
        ImGui_ImplVulkan_AddTexture(
            m_outputSampler, m_outputView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

    if (auto err = buildDescriptorSet()) {
        std::fprintf(stderr, "[HeatmapCompute] setSize(%u) failed at buildDescriptorSet: %s\n",
                     newSize, err->c_str());
        return;
    }
    std::fprintf(stderr, "[HeatmapCompute] resized to %ux%u\n", newSize, newSize);
}

}  // namespace btquant::renderer
