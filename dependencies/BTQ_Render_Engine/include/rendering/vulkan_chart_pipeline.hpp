#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

// Unsere Memory-Strukturen aus dem letzten Schritt
#include "candlestick_instancing.hpp"

namespace BTQuant {
namespace Rendering {

class VulkanChartPipeline {
 public:
  VulkanChartPipeline(VkDevice device, VkPhysicalDevice physical_device, VkRenderPass render_pass);
  ~VulkanChartPipeline();

  // Verhindere Kopien (RAII)
  VulkanChartPipeline(const VulkanChartPipeline&) = delete;
  VulkanChartPipeline& operator=(const VulkanChartPipeline&) = delete;

  // Bindet die Pipeline im ImGui-Callback
  void bind_and_draw(VkCommandBuffer cb, const ChartPushConstants& push_constants,
                     VkBuffer instance_buffer, uint32_t instance_count);

  VkPipeline get_pipeline() const { return graphics_pipeline_; }
  VkPipelineLayout get_pipeline_layout() const { return pipeline_layout_; }

 private:
  VkDevice device_;
  VkRenderPass render_pass_;

  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline graphics_pipeline_ = VK_NULL_HANDLE;

  // Hilfsfunktionen
  VkShaderModule create_shader_module(const std::vector<char>& code);
  std::vector<char> read_file(const std::string& filename);
  void build_pipeline();
};

}  // namespace Rendering
}  // namespace BTQuant