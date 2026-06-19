#ifndef BTQUANT_HEATMAP_COMPUTE_HPP
#define BTQUANT_HEATMAP_COMPUTE_HPP

#include <vulkan/vulkan.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <imgui.h>  // for ImTextureID

namespace btquant::renderer {

// Single compute-shader-driven heatmap aggregator.
//
// Per-frame:
//   1. Caller pushes recent trades (TradeInput array) via updateTrades().
//   2. dispatch(commandBuffer) binds the pipeline, dispatches 16x16 workgroups,
//      each thread iterates over all trades and writes its pixel's
//      R/G/B/A heatmap value to the storage image.
//   3. Caller retrieves the ImTextureID via textureId() and shows the image
//      via ImGui::Image().
//
// Layout (matches shaders/heatmap.comp):
//   set 0, binding 0 — input SSBO (TradeInput[])
//   set 0, binding 1 — output storage image (R8G8B8A8_UNORM, 256x256)
//   set 0, binding 2 — config UBO (price_low, price_high, image_w, image_h, n)
struct TradeInput {
    float price;     // normalized 0..1
    float time;      // normalized 0..1
    float volume;    // lots
    uint32_t side;   // 0 = buy, 1 = sell
};

struct HeatmapConfig {
    float price_low = 0.0f;
    float price_high = 1.0f;
    uint32_t image_width = 256;
    uint32_t image_height = 256;
};

class HeatmapCompute {
public:
    HeatmapCompute() = default;
    ~HeatmapCompute();

    // One-time setup. Allocates SSBO, image, descriptor set, pipeline.
    // Registers the output image with ImGui_ImplVulkan and stashes the ImTextureID.
    // Returns nullptr on success, error string on failure.
    [[nodiscard]] std::optional<std::string>
    initialize(VkDevice device,
              VkPhysicalDevice physicalDevice,
              VkCommandPool commandPool,
              VkQueue computeQueue,
              uint32_t computeQueueFamily,
              const HeatmapConfig& cfg = {});

    // Upload input trades to the SSBO. Old trades are overwritten on the next
    // dispatch (the shader reads cfg.num_trades, not the SSBO size).
    void updateTrades(const TradeInput* trades, uint32_t count);

    // Record compute commands into the given command buffer.
    // The buffer must be in RECORDING state with VK_COMMAND_BUFFER_LEVEL_PRIMARY.
    // Must be called inside a render pass OR with VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
    // (we don't need a render pass for compute — caller manages encoder state).
    void dispatch(VkCommandBuffer cmd);

    // ImTextureID ready for ImGui::Image(). Valid until shutdown().
    ImTextureID textureId() const noexcept { return m_imguiTextureId; }
    // Free everything. Safe to call multiple times.
    void shutdown();

    // Cheap accessor — useful for debug overlays.
    uint32_t dispatchedTradeCount() const noexcept { return m_lastDispatchedCount; }

private:
    std::optional<std::string>
    createInputBuffer();
    std::optional<std::string>
    createOutputImage();
    std::optional<std::string>
    buildDescriptorSetLayout();
    std::optional<std::string>
    buildComputePipeline();
    std::optional<std::string>
    buildDescriptorSet();
    std::optional<std::string>
    createConfigBuffer();

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    uint32_t m_computeQueueFamily = 0;

    HeatmapConfig m_cfg{};
    uint32_t m_lastDispatchedCount = 0;

    // Input SSBO (host-visible, device-local preferred, but host-visible is
    // simpler for our small input stream).
    VkBuffer m_inputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_inputMemory = VK_NULL_HANDLE;
    VkDeviceSize m_inputCapacity = 16 * 1024;  // up to 16k trades per frame

    // Output storage image (R8G8B8A8_UNORM, 256x256).
    VkImage m_outputImage = VK_NULL_HANDLE;
    VkDeviceMemory m_outputMemory = VK_NULL_HANDLE;
    VkImageView m_outputView = VK_NULL_HANDLE;
    VkSampler m_outputSampler = VK_NULL_HANDLE;
    ImTextureID m_imguiTextureId = 0;

    // Config UBO — mirrors HeatmapConfig.
    VkBuffer m_configBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_configMemory = VK_NULL_HANDLE;

    // Descriptor set pool + layout + set.
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

    // Compute pipeline (shader module + pipeline layout + pipeline).
    VkShaderModule m_shaderModule = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

}  // namespace btquant::renderer

#endif
