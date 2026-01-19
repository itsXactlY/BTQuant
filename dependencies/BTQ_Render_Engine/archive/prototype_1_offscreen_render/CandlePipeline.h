#pragma once

#include "vulkan_base_types.hpp"

namespace BTQuant {

struct CandleData {
  float x; // Index or Time
  float open;
  float high;
  float low;
  float close;
  uint32_t color; // Packed ARGB/ABGR
};

class CandlePipeline {
public:
  struct PushConstants {
    glm::mat4 projection;
    glm::vec2 chart_min;
    glm::vec2 chart_max;
    float candle_width;
    float padding;
  };

  CandlePipeline(VulkanCore *core, VkRenderPass renderPass);
  ~CandlePipeline();

  void Render(VkCommandBuffer cmd, const std::vector<CandleData> &candles,
              const PushConstants &pc);

private:
  void create_pipeline(VkRenderPass renderPass);
  void update_buffer(const std::vector<CandleData> &candles);

  VulkanCore *core_ = nullptr;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout ds_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

  BufferAllocation storage_buffer_;
  size_t current_buffer_size_ = 0;
};

} // namespace BTQuant
