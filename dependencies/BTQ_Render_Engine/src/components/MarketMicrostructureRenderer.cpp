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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>

#include "../../../ccapi/example/src/market_data_collector/market_data_types.h"
#include "../../include/analytics/cluster_engine.hpp"
#include "../../include/components/VulkanSynchronization.h"
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/vulkan_base_types.hpp"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"

namespace BTQuant::RenderEngine {

// to_string moved to header
// ============================================
// Helper functions for shader loading
// ============================================

namespace {
[[nodiscard]] std::expected<std::vector<uint32_t>, RendererError> load_spirv_binary(
    const std::string& path) noexcept {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) [[unlikely]] {
    return std::unexpected(RendererError::ShaderLoadFailed);
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
  file.close();

  return buffer;
}

[[nodiscard]] std::expected<VkShaderModule, RendererError> create_shader_module(
    VkDevice device, std::span<const uint32_t> code) noexcept {
  VkShaderModuleCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = code.size() * sizeof(uint32_t),
                                      .pCode = code.data()};

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
      [[unlikely]] {
    return std::unexpected(RendererError::PipelineCreationFailed);
  }

  return shaderModule;
}
}  // namespace

// ============================================
// MarketMicrostructureRenderer Implementation
// ============================================

MarketMicrostructureRenderer::MarketMicrostructureRenderer(
    BTQuant::VulkanCore* vulkanCore,
    std::shared_ptr<BTQuant::RenderEngine::MarketDataProcessor> marketDataProcessor,
    const RendererConfig& config)
    : vulkanCore_(vulkanCore),
      marketDataProcessor_(std::move(marketDataProcessor)),
      config_(config),
      lastFrameTime_(std::chrono::high_resolution_clock::now()),
      initialized_(false) {
  // Initialize the cluster engine with a default tick size
  // In a real implementation, this would be based on the symbol's tick size
  cluster_engine_ = std::make_unique<Analytics::ClusterEngine>(0.01);  // Default tick size of 0.01
}

MarketMicrostructureRenderer::~MarketMicrostructureRenderer() { cleanup(); }

std::expected<void, RendererError> MarketMicrostructureRenderer::initialize() {
  if (initialized_) [[unlikely]] {
    return {};
  }

  if (!vulkanCore_) [[unlikely]] {
    return std::unexpected(RendererError::NullVulkanCore);
  }

  if (auto res = createDescriptorSets(); !res) {
    return std::unexpected(res.error());
  }
  if (auto res = createComputePipelines(); !res) {
    return std::unexpected(res.error());
  }
  if (auto res = createGraphicsPipelines(); !res) {
    return std::unexpected(res.error());
  }
  if (auto res = createStorageBuffers(); !res) {
    return std::unexpected(res.error());
  }
  if (auto res = createTextureResources(); !res) {
    return std::unexpected(res.error());
  }

  // Initialize ring buffer manager for data updates
  constexpr uint32_t DOUBLE_BUFFER = 2;
  constexpr VkDeviceSize BUFFER_SIZE = 4 * 1024 * 1024;  // 4MB per buffer

  ringBufferManager_ =
      std::make_unique<RingBufferManager>(vulkanCore_->get_device(), DOUBLE_BUFFER, BUFFER_SIZE);

  initialized_ = true;

  std::println(
      "[MarketMicrostructureRenderer] Initialized successfully with "
      "modern C++26 standard");
  return {};
}

void MarketMicrostructureRenderer::cleanup() noexcept {
  if (!vulkanCore_) [[unlikely]] {
    return;
  }

  auto device = vulkanCore_->get_device();
  vkDeviceWaitIdle(device);

  // Helper lambda for cleaning up Vulkan handles
  auto destroyIfValid = [device]<typename T>(T& handle, auto destroyFn) {
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
  auto& memManager = vulkanCore_->get_memory_manager();

  auto deallocateIfValid = [&memManager](BufferAllocation& alloc) {
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

[[nodiscard]] std::expected<void, RendererError> MarketMicrostructureRenderer::prepare() {
  if (!initialized_) [[unlikely]] {
    return std::unexpected(RendererError::NotInitialized);
  }

  // Update storage buffers with latest data
  updateStorageBuffers();

  {
    std::lock_guard lock(statsMutex_);
    stats_.framesRendered++;
  }

  return {};
}

void MarketMicrostructureRenderer::executeCompute(VkCommandBuffer cmdBuffer) {
  if (!initialized_) [[unlikely]] {
    return;
  }

  // Execute compute shaders
  executeLOBHeatmapCompute(cmdBuffer);
  executeTPOProfileCompute(cmdBuffer);
}

void MarketMicrostructureRenderer::executeGraphics(VkCommandBuffer cmdBuffer) {
  if (!initialized_) [[unlikely]] {
    return;
  }

  // Render visualizations
  renderHeatmapTexture(cmdBuffer);
  renderFootprintChart(cmdBuffer);
  renderTPOProfile(cmdBuffer);

  // Calculate frame timing
  auto frameEndTime = std::chrono::high_resolution_clock::now();
  auto frameDuration = std::chrono::duration<float, std::milli>(frameEndTime - lastFrameTime_);

  lastFrameTime_ = frameEndTime;

  // Update statistics with exponential moving average
  {
    std::lock_guard lock(statsMutex_);
    constexpr float SMOOTHING_FACTOR = 0.1f;
    stats_.averageFrameTimeMs = stats_.averageFrameTimeMs * (1.0f - SMOOTHING_FACTOR) +
                                frameDuration.count() * SMOOTHING_FACTOR;
    stats_.lastUpdateTimeNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(frameEndTime.time_since_epoch())
            .count();
  }
}

void MarketMicrostructureRenderer::updateLOBData(const OrderbookData& orderbookData) {
  if (!initialized_) [[unlikely]] {
    return;
  }

  std::lock_guard lock(dataMutex_);

  // Convert OrderbookData to GPU-friendly format
  // Calculate total levels
  uint32_t totalLevels = static_cast<uint32_t>(orderbookData.bids.size() + orderbookData.asks.size());
  
  if (totalLevels > 0) {
    // Create GPU-friendly buffer
    size_t headerSize = sizeof(uint32_t) * 4; // currentTimeIndex, priceLevelsCount, basePrice, priceRange placeholder
    size_t levelSize = totalLevels * (sizeof(float) * 1 + sizeof(uint32_t) * 3); // price, askQuantity, bidQuantity, numOrders
    size_t totalSize = headerSize + levelSize;
    
    lobSnapshotBuffer_.resize(totalSize);
    
    // Fill the buffer with converted data
    uint8_t* bufferPtr = lobSnapshotBuffer_.data();
    
    // Write header data
    uint32_t currentTimeIndex = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    std::memcpy(bufferPtr, &currentTimeIndex, sizeof(currentTimeIndex));
    bufferPtr += sizeof(currentTimeIndex);
    
    std::memcpy(bufferPtr, &totalLevels, sizeof(totalLevels));
    bufferPtr += sizeof(totalLevels);
    
    // Calculate base price and price range
    float basePrice = 0.0f;
    float priceRange = 1.0f;
    if (!orderbookData.bids.empty() || !orderbookData.asks.empty()) {
        double minPrice = std::numeric_limits<double>::max();
        double maxPrice = std::numeric_limits<double>::lowest();
        
        for (const auto& bid : orderbookData.bids) {
            minPrice = std::min(minPrice, bid.price);
            maxPrice = std::max(maxPrice, bid.price);
        }
        for (const auto& ask : orderbookData.asks) {
            minPrice = std::min(minPrice, ask.price);
            maxPrice = std::max(maxPrice, ask.price);
        }
        
        basePrice = static_cast<float>(minPrice);
        priceRange = static_cast<float>(maxPrice - minPrice);
        if (priceRange <= 0.0f) priceRange = 1.0f; // Prevent division by zero
    }
    
    std::memcpy(bufferPtr, &basePrice, sizeof(basePrice));
    bufferPtr += sizeof(basePrice);
    
    std::memcpy(bufferPtr, &priceRange, sizeof(priceRange));
    bufferPtr += sizeof(priceRange);
    
    // Write level data
    for (const auto& bid : orderbookData.bids) {
        float price = static_cast<float>(bid.price);
        uint32_t bidQty = static_cast<uint32_t>(bid.size);
        uint32_t askQty = 0u; // No ask quantity for bid
        uint32_t numOrders = 1u; // Placeholder
        
        std::memcpy(bufferPtr, &price, sizeof(price));
        bufferPtr += sizeof(price);
        std::memcpy(bufferPtr, &askQty, sizeof(askQty)); // Ask quantity (0 for bids)
        bufferPtr += sizeof(askQty);
        std::memcpy(bufferPtr, &bidQty, sizeof(bidQty)); // Bid quantity
        bufferPtr += sizeof(bidQty);
        std::memcpy(bufferPtr, &numOrders, sizeof(numOrders));
        bufferPtr += sizeof(numOrders);
    }
    
    for (const auto& ask : orderbookData.asks) {
        float price = static_cast<float>(ask.price);
        uint32_t askQty = static_cast<uint32_t>(ask.size);
        uint32_t bidQty = 0u; // No bid quantity for ask
        uint32_t numOrders = 1u; // Placeholder
        
        std::memcpy(bufferPtr, &price, sizeof(price));
        bufferPtr += sizeof(price);
        std::memcpy(bufferPtr, &askQty, sizeof(askQty)); // Ask quantity
        bufferPtr += sizeof(askQty);
        std::memcpy(bufferPtr, &bidQty, sizeof(bidQty)); // Bid quantity (0 for asks)
        bufferPtr += sizeof(bidQty);
        std::memcpy(bufferPtr, &numOrders, sizeof(numOrders));
        bufferPtr += sizeof(numOrders);
    }
  }

  currentHeatmapTimeIndex_ = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());

  {
    std::lock_guard statsLock(statsMutex_);
    stats_.lobUpdates++;
  }
}

void MarketMicrostructureRenderer::updateTradeData(std::span<const TradeData> trades) {
  if (!initialized_) [[unlikely]] {
    return;
  }

  std::lock_guard lock(dataMutex_);
  currentTradeData_.clear();
  currentTradeData_.reserve(trades.size());
  
  // Convert TradeData to internal format
  for (const auto& trade : trades) {
    currentTradeData_.push_back(trade);
  }

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
    std::cerr << "[MarketMicrostructureRenderer] Too many clusters: " << clusters.size() << " > "
              << config_.footprintChart.maxClusters << "\n";
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

void MarketMicrostructureRenderer::notifyPriceAggregationChanged() {
  // This method should trigger a recalculation of clusters with the new price aggregation settings
  // In a complete implementation, this would notify the data pipeline to regenerate clusters
  // For now, we'll just clear the current clusters to force a refresh when new data comes in
  std::lock_guard lock(dataMutex_);
  currentFootprintClusters_.clear();
  currentClusterCount_ = 0;
}

void MarketMicrostructureRenderer::updateConfig(const RendererConfig& config) {
  std::lock_guard lock(statsMutex_);

  bool needsRecreation = (config.lobHeatmap.width != config_.lobHeatmap.width ||
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
// Symbol Management & Data Subscription
// ============================================

void MarketMicrostructureRenderer::setSymbol(uint32_t symbol_id) {
  if (symbol_id == current_symbol_id_) return;

  // Unsubscribe previous
  if (subscription_id_ > 0 && marketDataProcessor_) {
    marketDataProcessor_->unsubscribe(subscription_id_);
    subscription_id_ = 0;
  }
  if (subscription_id_lob_ > 0 && marketDataProcessor_) {
    marketDataProcessor_->unsubscribe(subscription_id_lob_);
    subscription_id_lob_ = 0;
  }

  current_symbol_id_ = symbol_id;

  // Clear buffers
  {
    std::lock_guard lock(dataMutex_);
    currentTradeData_.clear();
    currentFootprintClusters_.clear();
    lobSnapshotBuffer_.clear();
    currentClusterCount_ = 0;
  }
  resetStats();

  if (!marketDataProcessor_) return;

  // Subscribe to new symbol
  // Subscribe to TRADES
  subscription_id_ = marketDataProcessor_->subscribe(
      symbol_id, NotificationType::TRADE,
      [this](uint32_t sym, NotificationType type) { this->onMarketDataUpdate(sym, type); });

  // Subscribe to ORDERBOOK
  subscription_id_lob_ = marketDataProcessor_->subscribe(
      symbol_id, NotificationType::ORDERBOOK,
      [this](uint32_t sym, NotificationType type) { this->onMarketDataUpdate(sym, type); });

  // Also fetch initial state (Snapshot)
  auto ob = marketDataProcessor_->getOrderbookData(symbol_id);
  if (ob) {
    // Manually trigger update
    onMarketDataUpdate(symbol_id, NotificationType::ORDERBOOK);
  }
}

void MarketMicrostructureRenderer::onMarketDataUpdate(uint32_t symbol_id, NotificationType type) {
  if (symbol_id != current_symbol_id_ || !marketDataProcessor_) return;

  if (type == NotificationType::ORDERBOOK) {
    auto bookOpt = marketDataProcessor_->getOrderbookData(symbol_id);
    if (!bookOpt) return;

    const auto& book = *bookOpt;

    // Convert OrderbookData to GPU-compatible format
    // We need to merge bids and asks into price levels (assuming single price
    // represents row) Or separate? LOB Heatmap usually assumes Price -> BidVol,
    // AskVol
    std::map<double, std::pair<double, double>> levels;

    for (const auto& level : book.bids) levels[level.price].first += level.size;
    for (const auto& level : book.asks) levels[level.price].second += level.size;

    size_t numLevels = levels.size();
    size_t bufferSize =
        sizeof(uint32_t) * 4 + numLevels * (sizeof(float) * 1 + sizeof(uint32_t) * 3);
    std::vector<uint8_t> buffer(bufferSize);

    uint8_t* bufferPtr = buffer.data();
    
    // Write header data
    uint32_t currentTimeIndex = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    std::memcpy(bufferPtr, &currentTimeIndex, sizeof(currentTimeIndex));
    bufferPtr += sizeof(currentTimeIndex);
    
    uint32_t priceLevelsCount = static_cast<uint32_t>(numLevels);
    std::memcpy(bufferPtr, &priceLevelsCount, sizeof(priceLevelsCount));
    bufferPtr += sizeof(priceLevelsCount);
    
    // Calculate base price and price range
    float basePrice = 0.0f;
    float priceRange = 1.0f;
    if (!levels.empty()) {
        double minPrice = levels.begin()->first;
        double maxPrice = levels.rbegin()->first;
        basePrice = static_cast<float>(minPrice);
        priceRange = static_cast<float>(maxPrice - minPrice);
        if (priceRange <= 0.0f) priceRange = 1.0f; // Prevent division by zero
    }
    
    std::memcpy(bufferPtr, &basePrice, sizeof(basePrice));
    bufferPtr += sizeof(basePrice);
    
    std::memcpy(bufferPtr, &priceRange, sizeof(priceRange));
    bufferPtr += sizeof(priceRange);
    
    // Write level data
    uint32_t idx = 0;
    for (const auto& [price, volumes] : levels) {
        float fPrice = static_cast<float>(price);
        uint32_t askQty = static_cast<uint32_t>(volumes.second);
        uint32_t bidQty = static_cast<uint32_t>(volumes.first);
        uint32_t numOrders = 1u; // Placeholder
        
        std::memcpy(bufferPtr, &fPrice, sizeof(fPrice));
        bufferPtr += sizeof(fPrice);
        std::memcpy(bufferPtr, &askQty, sizeof(askQty)); // Ask quantity
        bufferPtr += sizeof(askQty);
        std::memcpy(bufferPtr, &bidQty, sizeof(bidQty)); // Bid quantity
        bufferPtr += sizeof(bidQty);
        std::memcpy(bufferPtr, &numOrders, sizeof(numOrders));
        bufferPtr += sizeof(numOrders);
    }
    
    // Create OrderbookData from the levels map
    OrderbookData orderbookData;
    orderbookData.symbol = "SYMBOL"; // Placeholder
    orderbookData.symbol_id = 0; // Placeholder
    orderbookData.timestamp = book.timestamp;
    
    // Convert levels back to bids and asks
    for (const auto& [price, volumes] : levels) {
        PriceLevel level;
        level.price = price;
        level.size = volumes.first + volumes.second; // Combined size

        if (volumes.first > 0) { // Has bid volume
            orderbookData.bids.push_back(level);
        } else if (volumes.second > 0) { // Has ask volume
            orderbookData.asks.push_back(level);
        }
    }

    updateLOBData(orderbookData);
  } else if (type == NotificationType::TRADE) {
    // Fetch recent trades for this symbol
    // Processor doesn't give us "just the new trade" in callback easily without
    // custom struct, but we can poll 'recent_trades' from analytics. Ideally we
    // should process the specific trade from the update if passed, but
    // subscription relies on callback signature. We will fetch latest 100
    // trades to ensure we have data.
    auto analytics = marketDataProcessor_->getSymbolAnalytics(symbol_id);

    // Convert to TradeData
    std::vector<TradeData> trades;
    trades.reserve(analytics.recent_trades.size());

    for (const auto& t : analytics.recent_trades) {
      TradeData trade;
      trade.symbol = t.symbol; // Assuming symbol exists in original struct
      trade.symbol_id = t.symbol_id;
      trade.timestamp = t.timestamp;
      trade.price = t.price;
      trade.size = t.size;
      trade.is_buy = t.is_buy;
      trades.push_back(trade);

      // Process the trade with the cluster engine
      // Convert to MarketData::Trade format for the cluster engine
      MarketData::Trade trade_data;
      trade_data.price = t.price;
      trade_data.quantity = t.size;
      trade_data.timestamp_us = t.timestamp;
      trade_data.is_buyer_maker = t.is_buy;  // Assuming buyer maker convention

      // Calculate time bucket based on timestamp (30-minute intervals as an example)
      constexpr int64_t INTERVAL_US = 30LL * 60 * 1000000;  // 30 minutes in microseconds
      int64_t elapsed =
          t.timestamp - analytics.last_update_time;  // Using last_update_time as reference
      int time_bucket = static_cast<int>(elapsed / INTERVAL_US);
      if (time_bucket < 0) time_bucket = 0;  // Ensure non-negative bucket index

      // Process the trade with the cluster engine using the selected time aggregation type
      if (cluster_engine_) {
        cluster_engine_->processTradeWithTimeAggregation(
            trade_data, time_aggregation_type_, volume_based_n_contracts_, tick_based_n_ticks_);
      }
    }

    updateTradeData(trades);

    // Also update Footprint Clusters?
    // If we don't have a cluster logic here, we rely on someone else calling
    // `updateFootprintClusters`. NOTE: Current footprint impl might need
    // external driving. We will leave it empty for now, assuming external or
    // future task implementation (Task 7).
  }
}

[[nodiscard]] std::expected<void, RendererError>
MarketMicrostructureRenderer::createComputePipelines() {
  if (!vulkanCore_) [[unlikely]] {
    return std::unexpected(RendererError::NullVulkanCore);
  }

  auto device = vulkanCore_->get_device();

  try {
    // 1. Create LOB Heatmap Pipeline
    auto lobHeatmapCode = load_spirv_binary("shaders/spirv/lob_heatmap.spv");
    if (!lobHeatmapCode)
      return std::unexpected(RendererError::ShaderLoadFailed);  // Handle error appropriately

    auto lobModuleRes = create_shader_module(device, *lobHeatmapCode);
    if (!lobModuleRes) return std::unexpected(lobModuleRes.error());
    VkShaderModule lobHeatmapModule = *lobModuleRes;

    VkPipelineShaderStageCreateInfo lobHeatmapStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = lobHeatmapModule,
        .pName = "main"};

    std::vector<VkDescriptorSetLayout> lobLayouts = {lobHeatmapDescriptorSetLayout_};
    VkPipelineLayoutCreateInfo lobLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(lobLayouts.size()),
        .pSetLayouts = lobLayouts.data()};

    if (vkCreatePipelineLayout(device, &lobLayoutInfo, nullptr, &lobHeatmapPipelineLayout_) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create LOB Heatmap pipeline layout");
    }

    VkComputePipelineCreateInfo lobPipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = lobHeatmapStage,
        .layout = lobHeatmapPipelineLayout_};

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &lobPipelineInfo, nullptr,
                                 &lobHeatmapPipeline_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create LOB Heatmap compute pipeline");
    }

    // 2. Create TPO Profile Pipeline
    auto tpoProfileCode = load_spirv_binary("shaders/spirv/tpo_profile.spv");
    if (!tpoProfileCode) return std::unexpected(RendererError::ShaderLoadFailed);

    auto tpoModuleRes = create_shader_module(device, *tpoProfileCode);
    if (!tpoModuleRes) return std::unexpected(tpoModuleRes.error());
    VkShaderModule tpoProfileModule = *tpoModuleRes;

    VkPipelineShaderStageCreateInfo tpoProfileStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = tpoProfileModule,
        .pName = "main"};

    std::vector<VkDescriptorSetLayout> tpoLayouts = {tpoProfileDescriptorSetLayout_};
    VkPipelineLayoutCreateInfo tpoLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(tpoLayouts.size()),
        .pSetLayouts = tpoLayouts.data()};

    if (vkCreatePipelineLayout(device, &tpoLayoutInfo, nullptr, &tpoProfilePipelineLayout_) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create TPO Profile pipeline layout");
    }

    VkComputePipelineCreateInfo tpoPipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = tpoProfileStage,
        .layout = tpoProfilePipelineLayout_};

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &tpoPipelineInfo, nullptr,
                                 &tpoProfilePipeline_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create TPO Profile compute pipeline");
    }

    vkDestroyShaderModule(device, lobHeatmapModule, nullptr);
    vkDestroyShaderModule(device, tpoProfileModule, nullptr);

    return {};
  } catch (const std::exception& e) {
    std::cerr << "[MarketMicrostructureRenderer] Compute pipeline creation failed: " << e.what()
              << "\n";
    return std::unexpected(RendererError::PipelineCreationFailed);
  }
}

[[nodiscard]] std::expected<void, RendererError>
MarketMicrostructureRenderer::createGraphicsPipelines() {
  if (!vulkanCore_) [[unlikely]] {
    return std::unexpected(RendererError::NullVulkanCore);
  }

  auto device = vulkanCore_->get_device();

  try {
    // 1. Load Footprint Shaders
    auto vertCode = load_spirv_binary("shaders/spirv/footprint_vert.spv");
    auto fragCode = load_spirv_binary("shaders/spirv/footprint_frag.spv");

    if (!vertCode || !fragCode) return std::unexpected(RendererError::ShaderLoadFailed);
    auto vertRes = create_shader_module(device, *vertCode);
    auto fragRes = create_shader_module(device, *fragCode);
    if (!vertRes || !fragRes) return std::unexpected(RendererError::ShaderCompilationFailed);

    VkShaderModule vertModule = *vertRes;
    VkShaderModule fragModule = *fragRes;

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vertModule,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragModule,
         .pName = "main"}};

    // 2. Pipeline Layout
    std::vector<VkDescriptorSetLayout> layouts = {
        footprintDescriptorSetLayout_  // Set 0
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()};

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &footprintPipelineLayout_) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create footprint pipeline layout");
    }

    // 3. Pipeline State
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE};

    VkViewport viewport{.x = 0.0f,
                        .y = 0.0f,
                        .width = (float)vulkanCore_->get_swapchain_extent().width,
                        .height = (float)vulkanCore_->get_swapchain_extent().height,
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f};

    VkRect2D scissor{.offset = {0, 0}, .extent = vulkanCore_->get_swapchain_extent()};

    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

    VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f};

    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE};

    VkPipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};

    VkPipelineColorBlendStateCreateInfo colorBlending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}};

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .layout = footprintPipelineLayout_,
        .renderPass = vulkanCore_->get_render_pass(),
        .subpass = 0};

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                  &footprintPipeline_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create footprint graphics pipeline");
    }

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);

    return {};
  } catch (const std::exception& e) {
    std::cerr << "[MarketMicrostructureRenderer] Pipeline creation failed: " << e.what() << "\n";
    return std::unexpected(RendererError::PipelineCreationFailed);
  }
}

[[nodiscard]] std::expected<void, RendererError>
MarketMicrostructureRenderer::createDescriptorSets() {
  if (!vulkanCore_) [[unlikely]] {
    return std::unexpected(RendererError::NullVulkanCore);
  }

  auto device = vulkanCore_->get_device();
  auto descriptorPool = vulkanCore_->get_descriptor_pool();

  // 1. LOB Heatmap Layout & Sets
  std::vector<VkDescriptorSetLayoutBinding> lobBindings = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};

  VkDescriptorSetLayoutCreateInfo lobLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(lobBindings.size()),
      .pBindings = lobBindings.data()};

  if (vkCreateDescriptorSetLayout(device, &lobLayoutInfo, nullptr,
                                  &lobHeatmapDescriptorSetLayout_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create LOB Heatmap descriptor set layout");
  }

  VkDescriptorSetAllocateInfo lobAllocInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                           .descriptorPool = descriptorPool,
                                           .descriptorSetCount = 1,
                                           .pSetLayouts = &lobHeatmapDescriptorSetLayout_};
  vkAllocateDescriptorSets(device, &lobAllocInfo, &lobHeatmapDescriptorSet_);

  // 2. Footprint Layout & Sets (Simplifying to single set for now if possible,
  // or multiple)
  std::vector<VkDescriptorSetLayoutBinding> footprintBindings = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};

  VkDescriptorSetLayoutCreateInfo footprintLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(footprintBindings.size()),
      .pBindings = footprintBindings.data()};

  if (vkCreateDescriptorSetLayout(device, &footprintLayoutInfo, nullptr,
                                  &footprintDescriptorSetLayout_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Footprint descriptor set layout");
  }

  VkDescriptorSetAllocateInfo footprintAllocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &footprintDescriptorSetLayout_};
  vkAllocateDescriptorSets(device, &footprintAllocInfo, &footprintDescriptorSet_);

  // 3. TPO Profile Layout & Sets
  std::vector<VkDescriptorSetLayoutBinding> tpoBindings = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};

  VkDescriptorSetLayoutCreateInfo tpoLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(tpoBindings.size()),
      .pBindings = tpoBindings.data()};

  if (vkCreateDescriptorSetLayout(device, &tpoLayoutInfo, nullptr,
                                  &tpoProfileDescriptorSetLayout_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create TPO Profile descriptor set layout");
  }

  VkDescriptorSetAllocateInfo tpoAllocInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                           .descriptorPool = descriptorPool,
                                           .descriptorSetCount = 1,
                                           .pSetLayouts = &tpoProfileDescriptorSetLayout_};
  vkAllocateDescriptorSets(device, &tpoAllocInfo, &tpoProfileDescriptorSet_);

  // 4. Update Descriptor Sets
  std::vector<VkWriteDescriptorSet> writes;

  // LOB Heatmap Updates
  VkDescriptorBufferInfo lobBufferInfo{lobHeatmapSSBO_.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorImageInfo lobImageInfo{lobHeatmapSampler_, lobHeatmapImageView_,
                                     VK_IMAGE_LAYOUT_GENERAL};

  writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = lobHeatmapDescriptorSet_,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &lobBufferInfo});

  writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = lobHeatmapDescriptorSet_,
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .pImageInfo = &lobImageInfo});

  // Footprint Updates
  VkDescriptorBufferInfo footprintSSBOInfo{footprintSSBO_.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo footprintUBOInfo{footprintUBO_.buffer, 0, VK_WHOLE_SIZE};

  writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = footprintDescriptorSet_,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &footprintSSBOInfo});

  writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = footprintDescriptorSet_,
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &footprintUBOInfo});

  // TPO Profile Updates
  VkDescriptorBufferInfo tpoInInfo{tpoProfileSSBO_.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo tpoOutInfo{tpoProfileHistogram_.buffer, 0, VK_WHOLE_SIZE};

  writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = tpoProfileDescriptorSet_,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &tpoInInfo});

  writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = tpoProfileDescriptorSet_,
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &tpoOutInfo});

  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  return {};
}

[[nodiscard]] std::expected<void, RendererError>
MarketMicrostructureRenderer::createStorageBuffers() {
  if (!vulkanCore_) [[unlikely]] {
    return std::unexpected(RendererError::NullVulkanCore);
  }

  auto& memManager = vulkanCore_->get_memory_manager();

  // Allocate LOB heatmap buffers
  auto lobDataSize = config_.lobHeatmap.width * config_.lobHeatmap.height * sizeof(float) * 4;
  lobHeatmapSSBO_ = memManager.allocate_storage_buffer(lobDataSize);
  lobHeatmapUBO_ = memManager.allocate_uniform_buffer(256);

  // Allocate footprint chart buffers
  auto footprintDataSize = config_.footprintChart.maxClusters * sizeof(CandleCluster);
  footprintSSBO_ = memManager.allocate_storage_buffer(footprintDataSize);
  footprintUBO_ = memManager.allocate_uniform_buffer(256);

  // Allocate TPO profile buffers
  auto tpoHistogramSize = config_.tpoProfile.bucketCount * sizeof(uint32_t);
  constexpr size_t TPO_TRADE_BUFFER_SIZE = 64 * 1024;
  tpoProfileSSBO_ = memManager.allocate_storage_buffer(TPO_TRADE_BUFFER_SIZE);
  tpoProfileUBO_ = memManager.allocate_uniform_buffer(256);
  tpoProfileHistogram_ = memManager.allocate_storage_buffer(tpoHistogramSize);

  return {};
}

[[nodiscard]] std::expected<void, RendererError>
MarketMicrostructureRenderer::createTextureResources() {
  if (!vulkanCore_) [[unlikely]] {
    return std::unexpected(RendererError::NullVulkanCore);
  }

  auto device = vulkanCore_->get_device();
  auto physicalDevice = vulkanCore_->get_physical_device();

  // 1. Create LOB Heatmap Texture
  VkImageCreateInfo imageInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D,
                              .format = VK_FORMAT_R8G8B8A8_UNORM,
                              .extent = {.width = config_.lobHeatmap.width,
                                         .height = config_.lobHeatmap.height,
                                         .depth = 1},
                              .mipLevels = 1,
                              .arrayLayers = 1,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                              .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

  if (vkCreateImage(device, &imageInfo, nullptr, &lobHeatmapImage_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create LOB Heatmap image");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, lobHeatmapImage_, &memRequirements);

  VkMemoryAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = vulkanCore_->find_memory_type(memRequirements.memoryTypeBits,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

  if (vkAllocateMemory(device, &allocInfo, nullptr, &lobHeatmapImageMemory_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate LOB Heatmap image memory");
  }

  vkBindImageMemory(device, lobHeatmapImage_, lobHeatmapImageMemory_, 0);

  // 2. Create Image View
  VkImageViewCreateInfo viewInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .image = lobHeatmapImage_,
                                 .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM,
                                 .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1}};

  if (vkCreateImageView(device, &viewInfo, nullptr, &lobHeatmapImageView_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create LOB Heatmap image view");
  }

  // 3. Create Sampler
  VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                  .magFilter = VK_FILTER_LINEAR,
                                  .minFilter = VK_FILTER_LINEAR,
                                  .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                  .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                                  .unnormalizedCoordinates = VK_FALSE};

  if (vkCreateSampler(device, &samplerInfo, nullptr, &lobHeatmapSampler_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create LOB Heatmap sampler");
  }

  // 4. Create Static Quad Vertex Buffer for Footprint
  static const float quadVertices[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                       1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  auto& memManager = vulkanCore_->get_memory_manager();
  footprintUBO_ = memManager.allocate_vertex_buffer(sizeof(quadVertices));
  if (footprintUBO_.mapped_ptr) {
    std::memcpy(footprintUBO_.mapped_ptr, quadVertices, sizeof(quadVertices));
  }

  return {};
}

// ============================================
// Private: Compute Execution
// ============================================

void MarketMicrostructureRenderer::executeLOBHeatmapCompute(VkCommandBuffer cmdBuffer) {
  if (lobHeatmapPipeline_ == VK_NULL_HANDLE) [[unlikely]] {
    return;
  }

  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, lobHeatmapPipeline_);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, lobHeatmapPipelineLayout_, 0,
                          1, &lobHeatmapDescriptorSet_, 0, nullptr);

  constexpr uint32_t WORKGROUP_SIZE_Y = 64;
  // Optimized dispatch: only update the single current time column
  auto workGroupsY = (config_.lobHeatmap.height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
  vkCmdDispatch(cmdBuffer, 1, workGroupsY, 1);
}

void MarketMicrostructureRenderer::executeTPOProfileCompute(VkCommandBuffer cmdBuffer) {
  if (tpoProfilePipeline_ == VK_NULL_HANDLE) [[unlikely]] {
    return;
  }

  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, tpoProfilePipeline_);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, tpoProfilePipelineLayout_, 0,
                          1, &tpoProfileDescriptorSet_, 0, nullptr);

  constexpr uint32_t TPO_WORKGROUP_SIZE = 64;
  auto workGroups = (config_.tpoProfile.bucketCount + TPO_WORKGROUP_SIZE - 1) / TPO_WORKGROUP_SIZE;
  vkCmdDispatch(cmdBuffer, workGroups, 1, 1);
}

// ============================================
// Private: Rendering
// ============================================

void MarketMicrostructureRenderer::renderFootprintChart(VkCommandBuffer cmdBuffer) {
  if (footprintPipeline_ == VK_NULL_HANDLE || currentClusterCount_ == 0) [[unlikely]] {
    return;
  }

  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, footprintPipeline_);

  // Bind Descriptor Sets
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, footprintPipelineLayout_, 0,
                          1, &footprintDescriptorSet_, 0, nullptr);

  // Bind Vertex Buffer (Static Quad)
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &footprintUBO_.buffer, offsets);

  constexpr uint32_t VERTICES_PER_QUAD = 6;
  vkCmdDraw(cmdBuffer, VERTICES_PER_QUAD, currentClusterCount_, 0, 0);
}

void MarketMicrostructureRenderer::renderHeatmapTexture(VkCommandBuffer cmdBuffer) {
  // Heatmap is drawn as a fullscreen quad or panel quad
  // We'll use a specialized graphics pipeline that reads from the compute
  // output image.
  // For now, assume it's part of the main UI overlay if not bound elsewhere.
}

void MarketMicrostructureRenderer::renderTPOProfile(VkCommandBuffer cmdBuffer) {
  if (tpoProfilePipeline_ == VK_NULL_HANDLE) [[unlikely]] {
    return;
  }

  // Draw TPO Histogram
}

void MarketMicrostructureRenderer::updateUniformBuffers(uint32_t currentFrame) {
  if (!initialized_) return;

  // Placeholder for updating view matrices and configuration
}

void MarketMicrostructureRenderer::updateStorageBuffers() {
  std::lock_guard lock(dataMutex_);

  // 1. Update Footprint SSBO
  if (!currentFootprintClusters_.empty() && footprintSSBO_.mapped_ptr) {
    auto dataSpan = std::as_bytes(std::span(currentFootprintClusters_));
    std::memcpy(footprintSSBO_.mapped_ptr, dataSpan.data(), dataSpan.size());
  }

  // 2. Update TPO Profile (Trade Data) SSBO
  // The shader expects a struct { uint tickCount, uint startTimeLow, ...
  // TradeTick ticks[] }
  if (!currentTradeData_.empty() && tpoProfileSSBO_.mapped_ptr) {
    struct TPOInputHeader {
      uint32_t tickCount;
      uint32_t startTimeL, startTimeH;
      uint32_t endTimeL, endTimeH;
      float minPrice, maxPrice;
    };

    TPOInputHeader header{
        .tickCount = static_cast<uint32_t>(currentTradeData_.size()),
    };

    std::memcpy(tpoProfileSSBO_.mapped_ptr, &header, sizeof(header));
    std::memcpy(static_cast<char*>(tpoProfileSSBO_.mapped_ptr) + sizeof(header),
                currentTradeData_.data(), currentTradeData_.size() * sizeof(TradeData));
  }

  // 3. Update LOB Heatmap Snapshot SSBO
  if (!lobSnapshotBuffer_.empty() && lobHeatmapSSBO_.mapped_ptr) {
    std::memcpy(lobHeatmapSSBO_.mapped_ptr, lobSnapshotBuffer_.data(), lobSnapshotBuffer_.size());
  }
}

// Performance recording handled in render() method
void MarketMicrostructureRenderer::recordFrameStats() {}

void* MarketMicrostructureRenderer::getHeatmapTextureID() {
  if (!initialized_ || lobHeatmapImageView_ == VK_NULL_HANDLE) [[unlikely]] {
    return nullptr;
  }

  if (heatmapTextureID_ == nullptr) {
    heatmapDescriptorSet_ = ImGui_ImplVulkan_AddTexture(lobHeatmapSampler_, lobHeatmapImageView_,
                                                        VK_IMAGE_LAYOUT_GENERAL);
    heatmapTextureID_ = (void*)heatmapDescriptorSet_;
    std::println("[MarketMicrostructureRenderer] Heatmap Texture registered with ImGui");
  }

  return heatmapTextureID_;
}

std::vector<std::vector<Analytics::ClusterCell>> MarketMicrostructureRenderer::getClusterCells()
    const {
  if (!cluster_engine_) {
    return {};  // Return empty vector if cluster engine is not initialized
  }

  // Access the cluster canvas from the cluster engine using the getter method
  return cluster_engine_->getClusterCanvas();
}

void MarketMicrostructureRenderer::set_on_cluster_engine_trade_callback(std::function<void()> callback) {
  on_cluster_engine_trade_callback_ = std::move(callback);
  
  // Set up the callback in the cluster engine to notify when a trade is processed
  if (cluster_engine_ && on_cluster_engine_trade_callback_) {
    cluster_engine_->set_on_trade_processed_callback(on_cluster_engine_trade_callback_);
  }
}

}  // namespace BTQuant::RenderEngine
