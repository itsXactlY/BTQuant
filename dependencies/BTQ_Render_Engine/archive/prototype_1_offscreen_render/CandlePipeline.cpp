#include "CandlePipeline.h"
#include <cstring>

namespace BTQuant {

CandlePipeline::CandlePipeline(VulkanCore *core, VkRenderPass renderPass)
    : core_(core) {
  create_pipeline(renderPass);
}

CandlePipeline::~CandlePipeline() {
  VkDevice device = core_->get_device();
  if (pipeline_ != VK_NULL_HANDLE)
    vkDestroyPipeline(device, pipeline_, nullptr);
  if (layout_ != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(device, layout_, nullptr);
  if (ds_layout_ != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(device, ds_layout_, nullptr);
  if (storage_buffer_.buffer != VK_NULL_HANDLE)
    core_->get_memory_manager().deallocate_buffer(storage_buffer_);
}

void CandlePipeline::create_pipeline(VkRenderPass renderPass) {
  VkDevice device = core_->get_device();

  // 1. Descriptor Set Layout
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo dsInfo = {};
  dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsInfo.bindingCount = 1;
  dsInfo.pBindings = &binding;
  vkCreateDescriptorSetLayout(device, &dsInfo, nullptr, &ds_layout_);

  // 2. Pipeline Layout
  VkPushConstantRange pushRange{};
  pushRange.offset = 0;
  pushRange.size = sizeof(PushConstants);
  pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkPipelineLayoutCreateInfo plInfo = {};
  plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plInfo.setLayoutCount = 1;
  plInfo.pSetLayouts = &ds_layout_;
  plInfo.pushConstantRangeCount = 1;
  plInfo.pPushConstantRanges = &pushRange;
  vkCreatePipelineLayout(device, &plInfo, nullptr, &layout_);

  // 3. Pipeline
  // Instanced rendering uses gl_VertexIndex to generate quad on the fly or
  // empty VO
  std::vector<VkVertexInputBindingDescription> bindings = {};
  std::vector<VkVertexInputAttributeDescription> attributes = {};

  pipeline_ = core_->create_graphics_pipeline(
      "shaders/candle_instanced.vert.spv", "shaders/ui_fragment.frag.spv",
      bindings, attributes, layout_, renderPass);

  // 4. Descriptor Set Allocation
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = core_->get_descriptor_pool();
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &ds_layout_;
  vkAllocateDescriptorSets(device, &allocInfo, &descriptor_set_);
}

void CandlePipeline::update_buffer(const std::vector<CandleData> &candles) {
  size_t requiredSize = candles.size() * sizeof(CandleData);
  if (requiredSize == 0)
    return;

  if (requiredSize > current_buffer_size_) {
    if (storage_buffer_.buffer != VK_NULL_HANDLE) {
      core_->get_memory_manager().deallocate_buffer(storage_buffer_);
    }
    storage_buffer_ =
        core_->get_memory_manager().allocate_storage_buffer(requiredSize);
    current_buffer_size_ = requiredSize;

    // Update descriptor set
    VkDescriptorBufferInfo bufferInfo{
        storage_buffer_.buffer, storage_buffer_.offset, storage_buffer_.size};
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(core_->get_device(), 1, &write, 0, nullptr);
  }

  if (storage_buffer_.mapped_ptr) {
    std::memcpy(storage_buffer_.mapped_ptr, candles.data(), requiredSize);
  }
}

void CandlePipeline::Render(VkCommandBuffer cmd,
                            const std::vector<CandleData> &candles,
                            const PushConstants &pc) {
  if (candles.empty() || pipeline_ == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE)
    return;

  update_buffer(candles);

  if (storage_buffer_.buffer == VK_NULL_HANDLE)
    return;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(PushConstants), &pc);

  // 12 vertices per instance: 6 for body quad, 6 for wick quad
  vkCmdDraw(cmd, 12, static_cast<uint32_t>(candles.size()), 0, 0);
}

} // namespace BTQuant
