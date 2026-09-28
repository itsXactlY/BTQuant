#ifndef BTQUANT_RENDER_PIPELINE_HPP
#define BTQUANT_RENDER_PIPELINE_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

namespace btquant::vulkan {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct UniformBufferObject {
    alignas(16) float model[4][4];
    alignas(16) float view[4][4];
    alignas(16) float proj[4][4];
};

class RenderPipeline {
public:
    RenderPipeline();
    ~RenderPipeline();

    bool initialize(VkDevice device, VkRenderPass renderPass, VkExtent2D extent);
    void shutdown(VkDevice device);

    void beginRenderPass(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, VkExtent2D extent);
    void endRenderPass(VkCommandBuffer commandBuffer);
    
    // Update uniform buffer
    void updateUniformBuffer(uint32_t currentImage, const UniformBufferObject& ubo);

    // Getters
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }
    VkDescriptorSet getDescriptorSet(size_t index) const { return m_descriptorSets[index]; }

private:
    bool m_initialized = false;
    VkDevice m_device = VK_NULL_HANDLE;
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    VkExtent2D m_extent{};
    
    // Pipeline components
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
    
    // Uniform buffers
    std::vector<Buffer> m_uniformBuffers;
    
    // Descriptor sets
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_descriptorSets;
    
    // Private helper methods
    bool createDescriptorSetLayout();
    bool createGraphicsPipeline();
    bool createUniformBuffers();
    bool createDescriptorPool();
    bool createDescriptorSets();
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                      VkMemoryPropertyFlags properties, Buffer& buffer);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

} // namespace btquant::vulkan

#endif