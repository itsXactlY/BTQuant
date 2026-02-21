#include "rendering/vulkan_chart_pipeline.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace BTQuant {
namespace Rendering {

VulkanChartPipeline::VulkanChartPipeline(VkDevice device, VkPhysicalDevice physical_device,
                                         VkRenderPass render_pass)
    : device_(device), render_pass_(render_pass) {
  (void)physical_device;  // Für spätere VRAM-Allokationen reserviert
  build_pipeline();
}

VulkanChartPipeline::~VulkanChartPipeline() {
  if (device_) {
    if (graphics_pipeline_) vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
    if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
  }
}

std::vector<char> VulkanChartPipeline::read_file(const std::string& filename) {
  namespace fs = std::filesystem;

  // Search order: CWD, parent of CWD (build/ case), exe dir parent
  std::vector<fs::path> search_dirs = {
      fs::current_path(),
      fs::current_path().parent_path(),
  };

  // Also try relative to executable location
  try {
    auto exe_path = fs::read_symlink("/proc/self/exe").parent_path();
    search_dirs.push_back(exe_path);
    search_dirs.push_back(exe_path.parent_path());
  } catch (...) {
  }

  for (const auto& dir : search_dirs) {
    fs::path full_path = dir / filename;
    std::ifstream file(full_path, std::ios::ate | std::ios::binary);
    if (file.is_open()) {
      size_t fileSize = (size_t)file.tellg();
      std::vector<char> buffer(fileSize);
      file.seekg(0);
      file.read(buffer.data(), fileSize);
      file.close();
      std::cerr << "[VulkanChartPipeline] Loaded shader: " << full_path << "\n";
      return buffer;
    }
  }

  throw std::runtime_error("Failed to open shader file: " + filename +
                           " (searched from CWD: " + fs::current_path().string() + ")");
}

VkShaderModule VulkanChartPipeline::create_shader_module(const std::vector<char>& code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module!");
  }
  return shaderModule;
}

void VulkanChartPipeline::build_pipeline() {
  // 1. Lade die Shader aus dem build/shaders Verzeichnis
  auto vertShaderCode = read_file("shaders/spirv/candlestick.vert.spv");
  auto fragShaderCode = read_file("shaders/spirv/candlestick.frag.spv");

  VkShaderModule vertShaderModule = create_shader_module(vertShaderCode);
  VkShaderModule fragShaderModule = create_shader_module(fragShaderCode);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

  // 2. Vertex Input Setup (Instancing!)
  VkVertexInputBindingDescription bindingDescription{};
  bindingDescription.binding = 0;
  bindingDescription.stride = sizeof(CandlestickInstance);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;  // WICHTIG: GPU instancing!

  std::array<VkVertexInputAttributeDescription, 8> attributeDescriptions{};
  // Location 0: Open
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(CandlestickInstance, open);
  // Location 1: High
  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(CandlestickInstance, high);
  // Location 2: Low
  attributeDescriptions[2].binding = 0;
  attributeDescriptions[2].location = 2;
  attributeDescriptions[2].format = VK_FORMAT_R32_SFLOAT;
  attributeDescriptions[2].offset = offsetof(CandlestickInstance, low);
  // Location 3: Close
  attributeDescriptions[3].binding = 0;
  attributeDescriptions[3].location = 3;
  attributeDescriptions[3].format = VK_FORMAT_R32_SFLOAT;
  attributeDescriptions[3].offset = offsetof(CandlestickInstance, close);
  // Location 4: Time
  attributeDescriptions[4].binding = 0;
  attributeDescriptions[4].location = 4;
  attributeDescriptions[4].format = VK_FORMAT_R32_SFLOAT;
  attributeDescriptions[4].offset = offsetof(CandlestickInstance, time);
  // Location 5: Width
  attributeDescriptions[5].binding = 0;
  attributeDescriptions[5].location = 5;
  attributeDescriptions[5].format = VK_FORMAT_R32_SFLOAT;
  attributeDescriptions[5].offset = offsetof(CandlestickInstance, width);
  // Location 6: Color
  attributeDescriptions[6].binding = 0;
  attributeDescriptions[6].location = 6;
  attributeDescriptions[6].format = VK_FORMAT_R32_UINT;
  attributeDescriptions[6].offset = offsetof(CandlestickInstance, color);
  // Location 7: Flags
  attributeDescriptions[7].binding = 0;
  attributeDescriptions[7].location = 7;
  attributeDescriptions[7].format = VK_FORMAT_R32_UINT;
  attributeDescriptions[7].offset = offsetof(CandlestickInstance, flags);

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

  // 3. Topology (Triangle List für unsere selbst generierten Quads)
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  // 4. Viewport & Scissor (wird dynamisch in bind_and_draw gesetzt)
  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  // 5. Rasterizer (Kein Culling, da wir 2D machen)
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

  // 6. Multisampling (Aus)
  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // 7. Color Blending (Standard Alpha Blending)
  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_TRUE;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;

  // 8. Dynamic States (Viewport, Scissor)
  std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  // 9. Push Constants für unsere Kamera-Matrix
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(ChartPushConstants);

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 0;  // Später für SSBO Descriptor Sets
  pipelineLayoutInfo.pSetLayouts = nullptr;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

  if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipeline_layout_) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout!");
  }

  // 10. Die finale Grafik-Pipeline
  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = pipeline_layout_;
  pipelineInfo.renderPass = render_pass_;
  pipelineInfo.subpass = 0;

  if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                &graphics_pipeline_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create graphics pipeline!");
  }

  // Cleanup Shader Module
  vkDestroyShaderModule(device_, fragShaderModule, nullptr);
  vkDestroyShaderModule(device_, vertShaderModule, nullptr);
}

void VulkanChartPipeline::bind_and_draw(VkCommandBuffer cb,
                                        const ChartPushConstants& push_constants,
                                        VkBuffer instance_buffer, uint32_t instance_count) {
  if (instance_count == 0) return;

  // Binde die Pipeline
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

  // Push Constants (Kamera) senden
  vkCmdPushConstants(cb, pipeline_layout_,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(ChartPushConstants), &push_constants);

  // Instance Buffer binden
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(cb, 0, 1, &instance_buffer, offsets);

  // Magie: 6 Vertices pro Kerze (Quad) mal X Instanzen
  vkCmdDraw(cb, 6, instance_count, 0, 0);
}

}  // namespace Rendering
}  // namespace BTQuant