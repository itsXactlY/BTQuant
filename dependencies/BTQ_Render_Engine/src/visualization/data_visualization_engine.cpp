#include "data_visualization_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace BTQuant {
namespace RenderEngine {

DataVisualizationEngine::DataVisualizationEngine(
    VkDevice device, VkPhysicalDevice physical_device)
    : device_(device), physical_device_(physical_device),
      command_pool_(VK_NULL_HANDLE), staging_buffer_(VK_NULL_HANDLE),
      staging_memory_(VK_NULL_HANDLE), grid_buffer_(VK_NULL_HANDLE),
      grid_memory_(VK_NULL_HANDLE), heatmap_buffer_(VK_NULL_HANDLE),
      heatmap_memory_(VK_NULL_HANDLE), chart_buffer_(VK_NULL_HANDLE),
      chart_memory_(VK_NULL_HANDLE), orderbook_buffer_(VK_NULL_HANDLE),
      orderbook_memory_(VK_NULL_HANDLE), max_symbols_(1000),
      max_chart_points_(10000), max_orderbook_levels_(40) {
  initialized_ = initializeBuffers();
  if (initialized_) {
    std::cout << "[DataVisualizationEngine] Initialized with support for "
              << max_symbols_ << " symbols" << std::endl;
  } else {
    std::cerr << "[DataVisualizationEngine] ERROR: Failed to initialize "
                 "DataVisualizationEngine buffers"
              << std::endl;
  }
}

DataVisualizationEngine::~DataVisualizationEngine() { cleanup(); }

bool DataVisualizationEngine::initializeBuffers() {
  // Create command pool for data transfer operations
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = 0; // Assume graphics queue family

  if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) !=
      VK_SUCCESS) {
    std::cerr << "[DataVisualizationEngine] Failed to create command pool"
              << std::endl;
    return false;
  }

  // Calculate buffer sizes
  size_t grid_size = max_symbols_ * sizeof(GridDataGPU);
  size_t heatmap_size = max_symbols_ * sizeof(HeatmapDataGPU);
  size_t chart_size = max_symbols_ * max_chart_points_ * sizeof(ChartPointGPU);
  size_t orderbook_size =
      max_symbols_ * max_orderbook_levels_ * sizeof(OrderbookLevelGPU);
  size_t staging_size =
      std::max({grid_size, heatmap_size, chart_size, orderbook_size});

  // Create staging buffer for CPU->GPU transfers
  if (!createBuffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    staging_buffer_, staging_memory_)) {
    std::cerr << "[DataVisualizationEngine] Failed to create staging buffer"
              << std::endl;
    return false;
  }

  // Create GPU buffers
  if (!createBuffer(
          grid_size,
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, grid_buffer_, grid_memory_)) {
    std::cerr << "[DataVisualizationEngine] Failed to create grid buffer"
              << std::endl;
    return false;
  }

  if (!createBuffer(heatmap_size,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, heatmap_buffer_,
                    heatmap_memory_)) {
    std::cerr << "[DataVisualizationEngine] Failed to create heatmap buffer"
              << std::endl;
    return false;
  }

  if (!createBuffer(
          chart_size,
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, chart_buffer_, chart_memory_)) {
    std::cerr << "[DataVisualizationEngine] Failed to create chart buffer"
              << std::endl;
    return false;
  }

  if (!createBuffer(orderbook_size,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, orderbook_buffer_,
                    orderbook_memory_)) {
    std::cerr << "[DataVisualizationEngine] Failed to create orderbook buffer"
              << std::endl;
    return false;
  }

  std::cout << "[DataVisualizationEngine] Created GPU buffers:" << std::endl;
  std::cout << "  Grid: " << (grid_size / 1024) << " KB" << std::endl;
  std::cout << "  Heatmap: " << (heatmap_size / 1024) << " KB" << std::endl;
  std::cout << "  Chart: " << (chart_size / 1024) << " KB" << std::endl;
  std::cout << "  Orderbook: " << (orderbook_size / 1024) << " KB" << std::endl;
  std::cout << "  Staging: " << (staging_size / 1024) << " KB" << std::endl;

  return true;
}

void DataVisualizationEngine::updateGridData(
    const std::vector<SymbolData> &symbols) {
  if (symbols.empty() || symbols.size() > max_symbols_) {
    return;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Prepare grid data for GPU
  std::vector<GridDataGPU> grid_data;
  grid_data.reserve(symbols.size());

  for (const auto &symbol : symbols) {
    GridDataGPU gpu_data{};

    // Copy basic data
    gpu_data.symbol_id = symbol.symbol_id;
    gpu_data.last_price = static_cast<float>(symbol.last_price);
    gpu_data.price_change = static_cast<float>(symbol.price_change);
    gpu_data.price_change_percent =
        static_cast<float>(symbol.price_change_percent);
    gpu_data.volume_24h = static_cast<float>(symbol.volume_24h);
    gpu_data.bid_price = static_cast<float>(symbol.bid_price);
    gpu_data.ask_price = static_cast<float>(symbol.ask_price);
    gpu_data.spread = static_cast<float>(symbol.spread);
    gpu_data.momentum = static_cast<float>(symbol.momentum);

    // Calculate color based on price change
    gpu_data.color = calculatePriceChangeColor(symbol.price_change_percent);

    // Calculate intensity based on volume and momentum
    float volume_intensity =
        std::min(1.0f, static_cast<float>(symbol.volume_24h) / 1000000.0f);
    float momentum_intensity =
        std::min(1.0f, std::abs(static_cast<float>(symbol.momentum)) / 10.0f);
    gpu_data.intensity = std::max(volume_intensity, momentum_intensity);

    // Status flags
    gpu_data.flags = 0;
    if (symbol.last_update_time > 0) {
      auto now =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();
      if (now - symbol.last_update_time < 5000000) { // Updated within 5 seconds
        gpu_data.flags |= SYMBOL_FLAG_ACTIVE;
      }
    }

    if (std::abs(symbol.momentum) > 2.0) { // High momentum
      gpu_data.flags |= SYMBOL_FLAG_HIGH_MOMENTUM;
    }

    if (symbol.volume_24h > 1000000.0) { // High volume
      gpu_data.flags |= SYMBOL_FLAG_HIGH_VOLUME;
    }

    grid_data.push_back(gpu_data);
  }

  // Transfer to GPU
  transferDataToGPU(grid_data.data(), grid_data.size() * sizeof(GridDataGPU),
                    grid_buffer_);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Update performance metrics
  std::lock_guard lock(perf_mutex_);
  performance_metrics_.grid_update_count++;
  performance_metrics_.grid_update_latency_us = duration.count();
  performance_metrics_.symbols_processed = symbols.size();

  std::cout << "[DataVisualizationEngine] Updated grid data for "
            << symbols.size() << " symbols in " << duration.count() << " µs"
            << std::endl;
}

void DataVisualizationEngine::updateHeatmapData(
    const std::vector<SymbolData> &symbols) {
  if (symbols.empty() || symbols.size() > max_symbols_) {
    return;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Prepare heatmap data for GPU
  std::vector<HeatmapDataGPU> heatmap_data;
  heatmap_data.reserve(symbols.size());

  // Calculate momentum statistics for normalization
  std::vector<float> momentums;
  for (const auto &symbol : symbols) {
    momentums.push_back(static_cast<float>(symbol.momentum));
  }

  float momentum_mean = 0.0f;
  float momentum_std = 0.0f;
  if (!momentums.empty()) {
    momentum_mean = std::accumulate(momentums.begin(), momentums.end(), 0.0f) /
                    momentums.size();

    float variance = 0.0f;
    for (float m : momentums) {
      variance += (m - momentum_mean) * (m - momentum_mean);
    }
    momentum_std = std::sqrt(variance / momentums.size());
  }

  for (size_t i = 0; i < symbols.size(); ++i) {
    const auto &symbol = symbols[i];
    HeatmapDataGPU gpu_data{};

    gpu_data.symbol_id = symbol.symbol_id;
    gpu_data.momentum = static_cast<float>(symbol.momentum);
    gpu_data.volume = static_cast<float>(symbol.volume_24h);
    gpu_data.volatility =
        static_cast<float>(symbol.momentum); // Use momentum as volatility proxy

    // Normalize momentum for heatmap intensity
    float normalized_momentum = 0.5f; // Default neutral
    if (momentum_std > 0) {
      normalized_momentum =
          0.5f + ((gpu_data.momentum - momentum_mean) / (momentum_std * 3.0f));
      normalized_momentum = std::max(0.0f, std::min(1.0f, normalized_momentum));
    }
    gpu_data.intensity = normalized_momentum;

    // Calculate heatmap color
    gpu_data.color = calculateHeatmapColor(normalized_momentum);

    // Grid position (arrange in a grid layout)
    int grid_width = static_cast<int>(std::ceil(std::sqrt(symbols.size())));
    gpu_data.grid_x = static_cast<float>(i % grid_width);
    gpu_data.grid_y = static_cast<float>(i / grid_width);

    heatmap_data.push_back(gpu_data);
  }

  // Transfer to GPU
  transferDataToGPU(heatmap_data.data(),
                    heatmap_data.size() * sizeof(HeatmapDataGPU),
                    heatmap_buffer_);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Update performance metrics
  std::lock_guard lock(perf_mutex_);
  performance_metrics_.heatmap_update_count++;
  performance_metrics_.heatmap_update_latency_us = duration.count();

  std::cout << "[DataVisualizationEngine] Updated heatmap data for "
            << symbols.size() << " symbols in " << duration.count() << " µs"
            << std::endl;
}

void DataVisualizationEngine::updateChartData(
    uint32_t symbol_id, const std::vector<ChartPoint> &points) {
  if (points.empty() || points.size() > max_chart_points_) {
    return;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Prepare chart data for GPU
  std::vector<ChartPointGPU> chart_data;
  chart_data.reserve(points.size());

  // Find price range for normalization
  float min_price = std::numeric_limits<float>::max();
  float max_price = std::numeric_limits<float>::lowest();
  for (const auto &point : points) {
    min_price = std::min(min_price, static_cast<float>(point.price));
    max_price = std::max(max_price, static_cast<float>(point.price));
  }

  float price_range = max_price - min_price;
  if (price_range == 0.0f)
    price_range = 1.0f;

  for (size_t i = 0; i < points.size(); ++i) {
    const auto &point = points[i];
    ChartPointGPU gpu_point{};

    gpu_point.timestamp = point.timestamp_us;
    gpu_point.price = static_cast<float>(point.price);
    gpu_point.volume = static_cast<float>(point.volume);

    // Normalized coordinates for rendering
    gpu_point.x = static_cast<float>(i) / static_cast<float>(points.size() - 1);
    gpu_point.y = (gpu_point.price - min_price) / price_range;

    // Color based on price movement
    if (i > 0) {
      float prev_price = static_cast<float>(points[i - 1].price);
      float change_percent =
          ((gpu_point.price - prev_price) / prev_price) * 100.0f;
      gpu_point.color = calculatePriceChangeColor(change_percent);
    } else {
      gpu_point.color = {0.5f, 0.5f, 0.5f,
                         1.0f}; // Neutral color for first point
    }

    chart_data.push_back(gpu_point);
  }

  // Transfer to GPU (offset by mapped index for multiple charts)
  size_t mapped_idx = getSymbolIndex(symbol_id);
  if (mapped_idx >= max_symbols_)
    return;

  size_t offset = mapped_idx * max_chart_points_ * sizeof(ChartPointGPU);
  transferDataToGPU(chart_data.data(),
                    chart_data.size() * sizeof(ChartPointGPU), chart_buffer_,
                    offset);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Update performance metrics
  std::lock_guard lock(perf_mutex_);
  performance_metrics_.chart_update_count++;
  performance_metrics_.chart_update_latency_us = duration.count();

  std::cout << "[DataVisualizationEngine] Updated chart data for symbol "
            << symbol_id << " with " << points.size() << " points in "
            << duration.count() << " µs" << std::endl;
}

void DataVisualizationEngine::updateOrderbookData(
    uint32_t symbol_id, const std::vector<PriceLevel> &bids,
    const std::vector<PriceLevel> &asks) {
  if ((bids.size() + asks.size()) > max_orderbook_levels_) {
    return;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Prepare orderbook data for GPU
  std::vector<OrderbookLevelGPU> orderbook_data;
  orderbook_data.reserve(bids.size() + asks.size());

  // Find max size for normalization
  float max_size = 0.0f;
  for (const auto &bid : bids) {
    max_size = std::max(max_size, static_cast<float>(bid.size));
  }
  for (const auto &ask : asks) {
    max_size = std::max(max_size, static_cast<float>(ask.size));
  }

  if (max_size == 0.0f)
    max_size = 1.0f;

  // Add bid levels
  for (size_t i = 0; i < bids.size(); ++i) {
    OrderbookLevelGPU gpu_level{};
    gpu_level.price = static_cast<float>(bids[i].price);
    gpu_level.size = static_cast<float>(bids[i].size);
    gpu_level.normalized_size = gpu_level.size / max_size;
    gpu_level.is_bid = 1;
    gpu_level.level_index = static_cast<uint32_t>(i);
    gpu_level.color = {0.0f, 0.8f, 0.0f, 0.8f}; // Green for bids

    orderbook_data.push_back(gpu_level);
  }

  // Add ask levels
  for (size_t i = 0; i < asks.size(); ++i) {
    OrderbookLevelGPU gpu_level{};
    gpu_level.price = static_cast<float>(asks[i].price);
    gpu_level.size = static_cast<float>(asks[i].size);
    gpu_level.normalized_size = gpu_level.size / max_size;
    gpu_level.is_bid = 0;
    gpu_level.level_index = static_cast<uint32_t>(i);
    gpu_level.color = {0.8f, 0.0f, 0.0f, 0.8f}; // Red for asks

    orderbook_data.push_back(gpu_level);
  }

  // Transfer to GPU (offset by mapped index for multiple orderbooks)
  size_t mapped_idx = getSymbolIndex(symbol_id);
  if (mapped_idx >= max_symbols_)
    return;

  size_t offset =
      mapped_idx * max_orderbook_levels_ * sizeof(OrderbookLevelGPU);
  transferDataToGPU(orderbook_data.data(),
                    orderbook_data.size() * sizeof(OrderbookLevelGPU),
                    orderbook_buffer_, offset);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Update performance metrics
  std::lock_guard lock(perf_mutex_);
  performance_metrics_.orderbook_update_count++;
  performance_metrics_.orderbook_update_latency_us = duration.count();

  std::cout << "[DataVisualizationEngine] Updated orderbook data for symbol "
            << symbol_id << " with " << (bids.size() + asks.size())
            << " levels in " << duration.count() << " µs" << std::endl;
}

VisualizationPerformanceMetrics
DataVisualizationEngine::getPerformanceMetrics() const {
  std::lock_guard lock(perf_mutex_);
  return performance_metrics_;
}

VkBuffer DataVisualizationEngine::getGridBuffer() const { return grid_buffer_; }

VkBuffer DataVisualizationEngine::getHeatmapBuffer() const {
  return heatmap_buffer_;
}

VkBuffer DataVisualizationEngine::getChartBuffer() const {
  return chart_buffer_;
}

VkBuffer DataVisualizationEngine::getOrderbookBuffer() const {
  return orderbook_buffer_;
}

bool DataVisualizationEngine::createBuffer(VkDeviceSize size,
                                           VkBufferUsageFlags usage,
                                           VkMemoryPropertyFlags properties,
                                           VkBuffer &buffer,
                                           VkDeviceMemory &memory) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex =
      findMemoryType(mem_requirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device_, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
    vkDestroyBuffer(device_, buffer, nullptr);
    return false;
  }

  vkBindBufferMemory(device_, buffer, memory, 0);
  return true;
}

uint32_t
DataVisualizationEngine::findMemoryType(uint32_t type_filter,
                                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type");
}

void DataVisualizationEngine::transferDataToGPU(const void *data, size_t size,
                                                VkBuffer dst_buffer,
                                                size_t offset) {
  if (!initialized_ || dst_buffer == VK_NULL_HANDLE ||
      staging_memory_ == VK_NULL_HANDLE) {
    return;
  }
  // Map staging buffer and copy data
  void *mapped_data;
  vkMapMemory(device_, staging_memory_, 0, size, 0, &mapped_data);
  memcpy(mapped_data, data, size);
  vkUnmapMemory(device_, staging_memory_);

  // Create command buffer for transfer
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool_;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer, &begin_info);

  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = offset;
  copy_region.size = size;
  vkCmdCopyBuffer(command_buffer, staging_buffer_, dst_buffer, 1, &copy_region);

  vkEndCommandBuffer(command_buffer);

  // Submit command buffer (synchronous for simplicity)
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  // Note: This assumes queue index 0 is graphics queue
  VkQueue graphics_queue;
  vkGetDeviceQueue(device_, 0, 0, &graphics_queue);

  vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue);

  vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}

ColorRGBA
DataVisualizationEngine::calculatePriceChangeColor(float change_percent) {
  ColorRGBA color;

  if (change_percent > 0) {
    // Green for positive changes
    float intensity = std::min(1.0f, std::abs(change_percent) / 10.0f);
    color.r = 0.0f;
    color.g = 0.5f + (intensity * 0.5f);
    color.b = 0.0f;
    color.a = 0.8f + (intensity * 0.2f);
  } else if (change_percent < 0) {
    // Red for negative changes
    float intensity = std::min(1.0f, std::abs(change_percent) / 10.0f);
    color.r = 0.5f + (intensity * 0.5f);
    color.g = 0.0f;
    color.b = 0.0f;
    color.a = 0.8f + (intensity * 0.2f);
  } else {
    // Gray for no change
    color.r = 0.5f;
    color.g = 0.5f;
    color.b = 0.5f;
    color.a = 0.8f;
  }

  return color;
}

ColorRGBA DataVisualizationEngine::calculateHeatmapColor(float intensity) {
  ColorRGBA color;

  // Use a blue-white-red heatmap
  if (intensity < 0.5f) {
    // Blue to white (cold to neutral)
    float t = intensity * 2.0f;
    color.r = t;
    color.g = t;
    color.b = 1.0f;
  } else {
    // White to red (neutral to hot)
    float t = (intensity - 0.5f) * 2.0f;
    color.r = 1.0f;
    color.g = 1.0f - t;
    color.b = 1.0f - t;
  }

  color.a = 0.8f;
  return color;
}

size_t DataVisualizationEngine::getSymbolIndex(uint32_t symbol_id) {
  std::lock_guard<std::mutex> lock(mapping_mutex_);

  auto it = symbol_id_to_index_.find(symbol_id);
  if (it != symbol_id_to_index_.end()) {
    return it->second;
  }

  // Assign next available slot
  if (next_symbol_index_ < max_symbols_) {
    size_t idx = next_symbol_index_++;
    symbol_id_to_index_[symbol_id] = idx;
    return idx;
  }

  // Out of slots - wrap around or return invalid?
  // For now, return max_symbols_ to indicate error
  return max_symbols_;
}

void DataVisualizationEngine::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);

    if (staging_buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, staging_buffer_, nullptr);
      staging_buffer_ = VK_NULL_HANDLE;
    }
    if (staging_memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, staging_memory_, nullptr);
      staging_memory_ = VK_NULL_HANDLE;
    }

    if (grid_buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, grid_buffer_, nullptr);
      grid_buffer_ = VK_NULL_HANDLE;
    }
    if (grid_memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, grid_memory_, nullptr);
      grid_memory_ = VK_NULL_HANDLE;
    }

    if (heatmap_buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, heatmap_buffer_, nullptr);
      heatmap_buffer_ = VK_NULL_HANDLE;
    }
    if (heatmap_memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, heatmap_memory_, nullptr);
      heatmap_memory_ = VK_NULL_HANDLE;
    }

    if (chart_buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, chart_buffer_, nullptr);
      chart_buffer_ = VK_NULL_HANDLE;
    }
    if (chart_memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, chart_memory_, nullptr);
      chart_memory_ = VK_NULL_HANDLE;
    }

    if (orderbook_buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, orderbook_buffer_, nullptr);
      orderbook_buffer_ = VK_NULL_HANDLE;
    }
    if (orderbook_memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, orderbook_memory_, nullptr);
      orderbook_memory_ = VK_NULL_HANDLE;
    }

    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, command_pool_, nullptr);
      command_pool_ = VK_NULL_HANDLE;
    }
  }
}

} // namespace RenderEngine
} // namespace BTQuant