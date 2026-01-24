/**
 * @file MarketMicrostructureRenderer.cpp
 * @brief Market Microstructure Renderer Implementation (C++23/26)
 *
 * Zero-latency trading visualization with modern C++ features:
 * - [[nodiscard]], [[likely]]/[[unlikely]] attributes
 * - Designated initializers and structured bindings
 * - std::span for safe buffer access
 *
 * @author Market Microstructure Renderer Team
 * @version 3.0.0 (C++23/26)
 */

#include "../../include/components/MarketMicrostructureRenderer.h"
#include "../../include/components/VulkanSynchronization.h"
#include "../../include/hotspine_data_bridge.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/trading/HotspineData.h"
#include "../../include/vulkan_base_types.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>

namespace BTQuant::RenderEngine {

// ============================================
// C++26 Error Types for Expected Returns
// ============================================

enum class RendererError {
  NotInitialized,
  NullVulkanCore,
  PipelineCreationFailed,
  BufferAllocationFailed,
  TooManyClusters
};

[[nodiscard]] constexpr auto to_string(RendererError error) noexcept
    -> std::string_view {
  switch (error) {
  case RendererError::NotInitialized:
    return "Renderer not initialized";
  case RendererError::NullVulkanCore:
    return "VulkanCore is null";
  case RendererError::PipelineCreationFailed:
    return "Pipeline creation failed";
  case RendererError::BufferAllocationFailed:
    return "Buffer allocation failed";
  case RendererError::TooManyClusters:
    return "Too many clusters provided";
  }
  return "Unknown error";
}

// ============================================
// MarketMicrostructureRenderer Implementation
// ============================================

MarketMicrostructureRenderer::MarketMicrostructureRenderer(
    VulkanCore *vulkanCore, std::shared_ptr<HotSpineDataBridge> hotspineBridge,
    std::shared_ptr<MarketDataProcessor> marketDataProcessor,
    const RendererConfig &config)
    : vulkanCore_(vulkanCore), hotspineBridge_(std::move(hotspineBridge)),
      marketDataProcessor_(std::move(marketDataProcessor)), config_(config),
      lastFrameTime_(std::chrono::high_resolution_clock::now()),
      initialized_(false) {}

MarketMicrostructureRenderer::~MarketMicrostructureRenderer() { cleanup(); }

void MarketMicrostructureRenderer::initialize() {
  if (initialized_) [[unlikely]] {
    return;
  }

  if (!vulkanCore_) [[unlikely]] {
    throw std::runtime_error("VulkanCore is null - cannot initialize renderer");
  }

  try {
    createComputePipelines();
    createGraphicsPipelines();
    createDescriptorSets();
    createStorageBuffers();
    createTextureResources();

    // Initialize ring buffer manager for data updates
    constexpr uint32_t DOUBLE_BUFFER = 2;
    constexpr VkDeviceSize BUFFER_SIZE = 1024 * 1024; // 1MB per buffer

    ringBufferManager_ = std::make_unique<RingBufferManager>(
        vulkanCore_->get_device(), DOUBLE_BUFFER, BUFFER_SIZE);

    initialized_ = true;

    std::cout << "[MarketMicrostructureRenderer] Initialized successfully\n";
  } catch (const std::exception &e) {
    std::cerr << "[MarketMicrostructureRenderer] Initialization failed: "
              << e.what() << "\n";
    cleanup();
    throw;
  }
}

void MarketMicrostructureRenderer::cleanup() {
  if (!vulkanCore_) [[unlikely]] {
    return;
  }

  auto device = vulkanCore_->get_device();
  vkDeviceWaitIdle(device);

  // Helper lambda for cleaning up Vulkan handles
  auto destroyIfValid = [device]<typename T>(T &handle, auto destroyFn) {
    if (handle != VK_NULL_HANDLE) {
      destroyFn(device, handle, nullptr);
      handle = VK_NULL_HANDLE;
    }
  };

  // Clean up LOB heatmap resources
  destroyIfValid(lobHeatmapPipeline_, vkDestroyPipeline);
  destroyIfValid(lobHeatmapPipelineLayout_, vkDestroyPipelineLayout);
  destroyIfValid(lobHeatmapDescriptorSetLayout_, vkDestroyDescriptorSetLayout);
  destroyIfValid(lobHeatmapImage_, vkDestroyImage);
  destroyIfValid(lobHeatmapImageView_, vkDestroyImageView);
  destroyIfValid(lobHeatmapImageMemory_, vkFreeMemory);
  destroyIfValid(lobHeatmapSampler_, vkDestroySampler);

  // Clean up footprint resources
  destroyIfValid(footprintPipeline_, vkDestroyPipeline);
  destroyIfValid(footprintPipelineLayout_, vkDestroyPipelineLayout);
  destroyIfValid(footprintDescriptorSetLayout_, vkDestroyDescriptorSetLayout);

  // Clean up TPO profile resources
  destroyIfValid(tpoProfilePipeline_, vkDestroyPipeline);
  destroyIfValid(tpoProfilePipelineLayout_, vkDestroyPipelineLayout);
  destroyIfValid(tpoProfileDescriptorSetLayout_, vkDestroyDescriptorSetLayout);

  // Deallocate buffer allocations through memory manager
  auto &memManager = vulkanCore_->get_memory_manager();

  auto deallocateIfValid = [&memManager](BufferAllocation &alloc) {
    if (alloc.buffer != VK_NULL_HANDLE) {
      memManager.deallocate_buffer(alloc);
      alloc = {};
    }
  };

  deallocateIfValid(lobHeatmapSSBO_);
  deallocateIfValid(lobHeatmapUBO_);
  deallocateIfValid(footprintSSBO_);
  deallocateIfValid(footprintUBO_);
  deallocateIfValid(tpoProfileSSBO_);
  deallocateIfValid(tpoProfileUBO_);
  deallocateIfValid(tpoProfileHistogram_);

  ringBufferManager_.reset();
  initialized_ = false;
}

void MarketMicrostructureRenderer::render(
    VkCommandBuffer cmdBuffer, [[maybe_unused]] uint32_t currentFrame,
    [[maybe_unused]] VulkanSyncContext &syncContext,
    [[maybe_unused]] TimelineSemaphore &timelineSemaphore) {
  if (!initialized_) [[unlikely]] {
    return;
  }

  auto frameStartTime = std::chrono::high_resolution_clock::now();

  // Update statistics atomically
  {
    std::lock_guard lock(statsMutex_);
    stats_.framesRendered++;
  }

  // Update storage buffers with latest data
  updateStorageBuffers();

  // Execute compute shaders
  executeLOBHeatmapCompute(cmdBuffer);
  executeTPOProfileCompute(cmdBuffer);

  // Render visualizations
  renderHeatmapTexture(cmdBuffer);
  renderFootprintChart(cmdBuffer);
  renderTPOProfile(cmdBuffer);

  // Calculate frame timing
  auto frameEndTime = std::chrono::high_resolution_clock::now();
  auto frameDuration =
      std::chrono::duration<float, std::milli>(frameEndTime - frameStartTime);

  lastFrameTime_ = frameEndTime;

  // Update statistics with exponential moving average
  {
    std::lock_guard lock(statsMutex_);
    constexpr float SMOOTHING_FACTOR = 0.1f;
    stats_.averageFrameTimeMs = std::lerp(
        stats_.averageFrameTimeMs, frameDuration.count(), SMOOTHING_FACTOR);
    stats_.lastUpdateTimeNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            frameEndTime.time_since_epoch())
            .count();
  }
}

void MarketMicrostructureRenderer::updateLOBData(
    const HotspineOrderBookSnapshot &snapshot) {
  if (!initialized_) [[unlikely]] {
    return;
  }

  std::lock_guard lock(dataMutex_);

  // Store snapshot for processing during render
  currentOrderBookData_.clear();
  currentOrderBookData_.push_back(snapshot);
  currentHeatmapTimeIndex_ = snapshot.currentTimeIndex;

  {
    std::lock_guard statsLock(statsMutex_);
    stats_.lobUpdates++;
  }
}

void MarketMicrostructureRenderer::updateTradeData(
    std::span<const HotspineTradeTick> trades) {
  if (!initialized_) [[unlikely]] {
    return;
  }

  std::lock_guard lock(dataMutex_);

  currentTradeData_.assign(trades.begin(), trades.end());
  tpoProfileNeedsReset_ = true;

  {
    std::lock_guard statsLock(statsMutex_);
    stats_.tradeUpdates++;
  }
}

void MarketMicrostructureRenderer::updateFootprintClusters(
    std::span<const CandleCluster> clusters) {
  if (!initialized_) [[unlikely]] {
    return;
  }

  if (clusters.size() > config_.footprintChart.maxClusters) [[unlikely]] {
    std::println(std::cerr,
                 "[MarketMicrostructureRenderer] Too many clusters: {} > {}",
                 clusters.size(), config_.footprintChart.maxClusters);
    return;
  }

  std::lock_guard lock(dataMutex_);

  currentFootprintClusters_.assign(clusters.begin(), clusters.end());
  currentClusterCount_ = static_cast<uint32_t>(clusters.size());

  {
    std::lock_guard statsLock(statsMutex_);
    stats_.footprintCellsRendered = currentClusterCount_;
  }
}

void MarketMicrostructureRenderer::updateConfig(const RendererConfig &config) {
  std::lock_guard lock(statsMutex_);

  bool needsRecreation =
      (config.lobHeatmap.width != config_.lobHeatmap.width ||
       config.lobHeatmap.height != config_.lobHeatmap.height);

  config_ = config;

  if (needsRecreation && initialized_) [[unlikely]] {
    std::println(std::cerr,
                 "[MarketMicrostructureRenderer] Config change requires "
                 "recreation - not implemented");
  }
}

[[nodiscard]] RendererStats MarketMicrostructureRenderer::getStats() const {
  std::lock_guard lock(statsMutex_);
  return stats_;
}

void MarketMicrostructureRenderer::resetStats() {
  std::lock_guard lock(statsMutex_);
  stats_ = RendererStats{};
}

// ============================================
// Private: Resource Creation
// ============================================

void MarketMicrostructureRenderer::createComputePipelines() {
  // Placeholder - real implementation would load SPIR-V and create pipelines
}

void MarketMicrostructureRenderer::createGraphicsPipelines() {
  // Placeholder - real implementation would create graphics pipelines
}

void MarketMicrostructureRenderer::createDescriptorSets() {
  // Placeholder - real implementation would create descriptor sets
}

void MarketMicrostructureRenderer::createStorageBuffers() {
  if (!vulkanCore_) [[unlikely]] {
    return;
  }

  auto &memManager = vulkanCore_->get_memory_manager();

  // Allocate LOB heatmap buffers
  auto lobDataSize =
      config_.lobHeatmap.width * config_.lobHeatmap.height * sizeof(float) * 4;
  lobHeatmapSSBO_ = memManager.allocate_storage_buffer(lobDataSize);
  lobHeatmapUBO_ = memManager.allocate_uniform_buffer(256);

  // Allocate footprint chart buffers
  auto footprintDataSize =
      config_.footprintChart.maxClusters * sizeof(CandleCluster);
  footprintSSBO_ = memManager.allocate_storage_buffer(footprintDataSize);
  footprintUBO_ = memManager.allocate_uniform_buffer(256);

  // Allocate TPO profile buffers
  auto tpoHistogramSize = config_.tpoProfile.bucketCount * sizeof(uint32_t);
  constexpr size_t TPO_TRADE_BUFFER_SIZE = 64 * 1024;
  tpoProfileSSBO_ = memManager.allocate_storage_buffer(TPO_TRADE_BUFFER_SIZE);
  tpoProfileUBO_ = memManager.allocate_uniform_buffer(256);
  tpoProfileHistogram_ = memManager.allocate_storage_buffer(tpoHistogramSize);
}

void MarketMicrostructureRenderer::createTextureResources() {
  // Placeholder for texture resource creation
}

// ============================================
// Private: Compute Execution
// ============================================

void MarketMicrostructureRenderer::executeLOBHeatmapCompute(
    VkCommandBuffer cmdBuffer) {
  if (lobHeatmapPipeline_ == VK_NULL_HANDLE) [[unlikely]] {
    return;
  }

  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    lobHeatmapPipeline_);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          lobHeatmapPipelineLayout_, 0, 1,
                          &lobHeatmapDescriptorSet_, 0, nullptr);

  constexpr uint32_t WORKGROUP_SIZE = 16;
  auto workGroupsX =
      (config_.lobHeatmap.width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  auto workGroupsY =
      (config_.lobHeatmap.height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
  vkCmdDispatch(cmdBuffer, workGroupsX, workGroupsY, 1);
}

void MarketMicrostructureRenderer::executeTPOProfileCompute(
    VkCommandBuffer cmdBuffer) {
  if (tpoProfilePipeline_ == VK_NULL_HANDLE) [[unlikely]] {
    return;
  }

  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    tpoProfilePipeline_);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          tpoProfilePipelineLayout_, 0, 1,
                          &tpoProfileDescriptorSet_, 0, nullptr);

  constexpr uint32_t TPO_WORKGROUP_SIZE = 64;
  auto workGroups = (config_.tpoProfile.bucketCount + TPO_WORKGROUP_SIZE - 1) /
                    TPO_WORKGROUP_SIZE;
  vkCmdDispatch(cmdBuffer, workGroups, 1, 1);
}

// ============================================
// Private: Rendering
// ============================================

void MarketMicrostructureRenderer::renderFootprintChart(
    VkCommandBuffer cmdBuffer) {
  if (footprintPipeline_ == VK_NULL_HANDLE || currentClusterCount_ == 0)
      [[unlikely]] {
    return;
  }

  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    footprintPipeline_);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          footprintPipelineLayout_, 0, 1,
                          &footprintDescriptorSet_, 0, nullptr);

  constexpr uint32_t VERTICES_PER_QUAD = 6;
  vkCmdDraw(cmdBuffer, VERTICES_PER_QUAD, currentClusterCount_, 0, 0);
}

void MarketMicrostructureRenderer::renderHeatmapTexture(
    [[maybe_unused]] VkCommandBuffer cmdBuffer) {
  // Placeholder for heatmap texture rendering
}

void MarketMicrostructureRenderer::renderTPOProfile(
    [[maybe_unused]] VkCommandBuffer cmdBuffer) {
  // Placeholder for TPO profile rendering
}

// ============================================
// Private: Buffer Updates
// ============================================

void MarketMicrostructureRenderer::updateUniformBuffers(
    [[maybe_unused]] uint32_t currentFrame) {
  // Update uniform buffers with current frame data
}

void MarketMicrostructureRenderer::updateStorageBuffers() {
  std::lock_guard lock(dataMutex_);

  // Update footprint SSBO using std::ranges::copy if possible
  if (!currentFootprintClusters_.empty() && footprintSSBO_.mapped_ptr) {
    auto dataSpan = std::as_bytes(std::span(currentFootprintClusters_));
    std::memcpy(footprintSSBO_.mapped_ptr, dataSpan.data(), dataSpan.size());
  }

  // Update trade data SSBO
  if (!currentTradeData_.empty() && tpoProfileSSBO_.mapped_ptr) {
    auto dataSpan = std::as_bytes(std::span(currentTradeData_));
    std::memcpy(tpoProfileSSBO_.mapped_ptr, dataSpan.data(), dataSpan.size());
  }
}

// Performance recording handled in render() method
void MarketMicrostructureRenderer::recordFrameStats() {}

} // namespace BTQuant::RenderEngine