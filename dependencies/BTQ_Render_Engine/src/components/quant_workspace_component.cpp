#include "../../include/components/quant_workspace_component.hpp"
#include "implot_internal.h"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : UIComponent({0, 0}, {0, 0}), bridge_(bridge), processor_(processor) {

  // Ensure ImPlot context is created (must be called once)
  static bool implot_init = false;
  if (!implot_init) {
    ImPlot::CreateContext();
    implot_init = true;
  }

  // Create chart manager and indicator renderer
  chart_manager_ = std::make_unique<ChartManager>(bridge, processor);
  indicator_renderer_ = std::make_unique<IndicatorRenderer>(nullptr, processor);

  std::cout << "[QuantWorkspaceComponent] Enhanced version initialized"
            << std::endl;
}

void QuantWorkspaceComponent::initialize_vulkan_resources(VulkanCore *core) {
  core_ = core;
  indicator_renderer_->initialize_vulkan_resources();

  if (!viz_engine_)
    return;

  // 1. Create Descriptor Set Layout for the Candle Storage Buffer
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &binding;

  if (vkCreateDescriptorSetLayout(core->get_device(), &layoutInfo, nullptr,
                                  &candle_descriptor_set_layout_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create candle descriptor set layout!");
  }

  // 2. Create Pipeline Layout with Push Constants
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(CandlePushConstants);

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &candle_descriptor_set_layout_;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

  if (vkCreatePipelineLayout(core->get_device(), &pipelineLayoutInfo, nullptr,
                             &candle_pipeline_layout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create candle pipeline layout!");
  }

  // 3. Create Graphics Pipeline
  // candle_instanced.vert/frag don't use vertex buffers (use gl_VertexIndex)
  candle_pipeline_ = core->create_graphics_pipeline(
      "shaders/candle_instanced.vert.spv", "shaders/candle_instanced.frag.spv",
      {}, {}, // No vertex bindings/attributes
      candle_pipeline_layout_, core->get_render_pass());

  // 4. Allocate Descriptor Set
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = core->get_descriptor_pool();
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &candle_descriptor_set_layout_;

  if (vkAllocateDescriptorSets(core->get_device(), &allocInfo,
                               &candle_descriptor_set_) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate candle descriptor sets!");
  }

  // 5. Update Descriptor Set with Viz Engine Buffer
  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = viz_engine_->getChartBuffer();
  bufferInfo.offset = 0;
  bufferInfo.range = VK_WHOLE_SIZE;

  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = candle_descriptor_set_;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(core->get_device(), 1, &descriptorWrite, 0, nullptr);

  // 6. Create Descriptor Set Layout for WAE Compute
  std::vector<VkDescriptorSetLayoutBinding> waeBindings(2);
  waeBindings[0].binding = 0;
  waeBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  waeBindings[0].descriptorCount = 1;
  waeBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  waeBindings[1].binding = 1;
  waeBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  waeBindings[1].descriptorCount = 1;
  waeBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo waeLayoutInfo{};
  waeLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  waeLayoutInfo.bindingCount = static_cast<uint32_t>(waeBindings.size());
  waeLayoutInfo.pBindings = waeBindings.data();

  if (vkCreateDescriptorSetLayout(core->get_device(), &waeLayoutInfo, nullptr,
                                  &wae_descriptor_set_layout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create wae descriptor set layout!");
  }

  // 7. Create WAE Pipeline Layout with Push Constants
  VkPushConstantRange waePushConstantRange{};
  waePushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  waePushConstantRange.offset = 0;
  waePushConstantRange.size = sizeof(RenderEngine::WAEPushConstants);

  VkPipelineLayoutCreateInfo waePipelineLayoutInfo{};
  waePipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  waePipelineLayoutInfo.setLayoutCount = 1;
  waePipelineLayoutInfo.pSetLayouts = &wae_descriptor_set_layout_;
  waePipelineLayoutInfo.pushConstantRangeCount = 1;
  waePipelineLayoutInfo.pPushConstantRanges = &waePushConstantRange;

  if (vkCreatePipelineLayout(core->get_device(), &waePipelineLayoutInfo,
                             nullptr, &wae_pipeline_layout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create wae pipeline layout!");
  }

  // 8. Create WAE Compute Pipeline
  wae_pipeline_ = core->create_compute_pipeline("shaders/wae.comp.spv",
                                                wae_pipeline_layout_);

  // 9. Allocate WAE Descriptor Set
  VkDescriptorSetAllocateInfo waeAllocInfo{};
  waeAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  waeAllocInfo.descriptorPool = core->get_descriptor_pool();
  waeAllocInfo.descriptorSetCount = 1;
  waeAllocInfo.pSetLayouts = &wae_descriptor_set_layout_;

  if (vkAllocateDescriptorSets(core->get_device(), &waeAllocInfo,
                               &wae_descriptor_set_) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate wae descriptor sets!");
  }

  // 10. Update WAE Descriptor Set
  std::vector<VkDescriptorBufferInfo> waeBufferInfos(2);
  waeBufferInfos[0].buffer = viz_engine_->getChartBuffer();
  waeBufferInfos[0].offset = 0;
  waeBufferInfos[0].range = VK_WHOLE_SIZE;

  waeBufferInfos[1].buffer = viz_engine_->getIndicatorBuffer();
  waeBufferInfos[1].offset = 0;
  waeBufferInfos[1].range = VK_WHOLE_SIZE;

  std::vector<VkWriteDescriptorSet> waeDescriptorWrites(2);
  waeDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  waeDescriptorWrites[0].dstSet = wae_descriptor_set_;
  waeDescriptorWrites[0].dstBinding = 0;
  waeDescriptorWrites[0].descriptorCount = 1;
  waeDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  waeDescriptorWrites[0].pBufferInfo = &waeBufferInfos[0];

  waeDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  waeDescriptorWrites[1].dstSet = wae_descriptor_set_;
  waeDescriptorWrites[1].dstBinding = 1;
  waeDescriptorWrites[1].descriptorCount = 1;
  waeDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  waeDescriptorWrites[1].pBufferInfo = &waeBufferInfos[1];

  vkUpdateDescriptorSets(core->get_device(),
                         static_cast<uint32_t>(waeDescriptorWrites.size()),
                         waeDescriptorWrites.data(), 0, nullptr);

  // 11. Create Descriptor Set Layout for WAE Graphics (Visualizing Results)
  VkDescriptorSetLayoutBinding waeGraphicsBinding{};
  waeGraphicsBinding.binding = 0;
  waeGraphicsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  waeGraphicsBinding.descriptorCount = 1;
  waeGraphicsBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo waeGraphicsLayoutInfo{};
  waeGraphicsLayoutInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  waeGraphicsLayoutInfo.bindingCount = 1;
  waeGraphicsLayoutInfo.pBindings = &waeGraphicsBinding;

  if (vkCreateDescriptorSetLayout(
          core->get_device(), &waeGraphicsLayoutInfo, nullptr,
          &wae_graphics_descriptor_set_layout_) != VK_SUCCESS) {
    throw std::runtime_error(
        "failed to create wae graphics descriptor set layout!");
  }

  // 12. Create WAE Graphics Pipeline Layout
  VkPushConstantRange waeGraphicsPushRange{};
  waeGraphicsPushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  waeGraphicsPushRange.offset = 0;
  waeGraphicsPushRange.size = sizeof(CandlePushConstants);

  VkPipelineLayoutCreateInfo waeGraphicsPipelineLayoutInfo{};
  waeGraphicsPipelineLayoutInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  waeGraphicsPipelineLayoutInfo.setLayoutCount = 1;
  waeGraphicsPipelineLayoutInfo.pSetLayouts =
      &wae_graphics_descriptor_set_layout_;
  waeGraphicsPipelineLayoutInfo.pushConstantRangeCount = 1;
  waeGraphicsPipelineLayoutInfo.pPushConstantRanges = &waeGraphicsPushRange;

  if (vkCreatePipelineLayout(core->get_device(), &waeGraphicsPipelineLayoutInfo,
                             nullptr,
                             &wae_graphics_pipeline_layout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create wae graphics pipeline layout!");
  }

  // 13. Create WAE Graphics Pipeline
  wae_graphics_pipeline_ = core->create_graphics_pipeline(
      "shaders/wae_instanced.vert.spv", "shaders/wae_instanced.frag.spv", {},
      {}, // No vertex attributes
      wae_graphics_pipeline_layout_, core->get_render_pass());

  // 14. Allocate and Update WAE Graphics Descriptor Set
  VkDescriptorSetAllocateInfo waeGraphicsAllocInfo{};
  waeGraphicsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  waeGraphicsAllocInfo.descriptorPool = core->get_descriptor_pool();
  waeGraphicsAllocInfo.descriptorSetCount = 1;
  waeGraphicsAllocInfo.pSetLayouts = &wae_graphics_descriptor_set_layout_;

  if (vkAllocateDescriptorSets(core->get_device(), &waeGraphicsAllocInfo,
                               &wae_graphics_descriptor_set_) != VK_SUCCESS) {
    throw std::runtime_error(
        "failed to allocate wae graphics descriptor sets!");
  }

  VkDescriptorBufferInfo waeGraphicsBufferInfo{};
  waeGraphicsBufferInfo.buffer = viz_engine_->getIndicatorBuffer();
  waeGraphicsBufferInfo.offset = 0;
  waeGraphicsBufferInfo.range = VK_WHOLE_SIZE;

  VkWriteDescriptorSet waeGraphicsWrite{};
  waeGraphicsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  waeGraphicsWrite.dstSet = wae_graphics_descriptor_set_;
  waeGraphicsWrite.dstBinding = 0;
  waeGraphicsWrite.descriptorCount = 1;
  waeGraphicsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  waeGraphicsWrite.pBufferInfo = &waeGraphicsBufferInfo;

  vkUpdateDescriptorSets(core->get_device(), 1, &waeGraphicsWrite, 0, nullptr);

  std::cout
      << "[QuantWorkspaceComponent] High-performance candle, WAE compute, and "
         "WAE graphics pipelines created."
      << std::endl;
}

void QuantWorkspaceComponent::update(float dt) { chart_manager_->update(); }

void QuantWorkspaceComponent::render_gui() {
  auto &instruments = bridge_->GetAllInstruments();
  std::lock_guard<std::mutex> lock(bridge_->GetMapMutex());

  // Render chart controls
  if (show_chart_controls_) {
    render_chart_controls();
  }

  // Render indicator selector
  if (show_indicator_selector_) {
    render_indicator_selector();
  }

  // Render all visible charts
  for (const auto &chart : chart_manager_->get_visible_charts()) {
    auto it = instruments.find(chart.symbol);
    if (it != instruments.end()) {
      bool open = true;
      ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

      if (ImGui::Begin((chart.symbol + " - " +
                        std::to_string(static_cast<int>(chart.timeframe)))
                           .c_str(),
                       &open)) {
        render_instrument_chart(chart.chart_id, chart.symbol, *it->second,
                                chart.timeframe);
      }
      ImGui::End();

      if (!open) {
        chart_manager_->destroy_chart(chart.chart_id);
      }
    }
  }
}

void QuantWorkspaceComponent::render_chart_controls() {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Chart Controls", &show_chart_controls_)) {
    // Timeframe selector
    const char *timeframes[] = {"1 Minute", "5 Minutes", "15 Minutes",
                                "1 Hour",   "4 Hours",   "1 Day"};
    int selected = static_cast<int>(selected_timeframe_);
    if (ImGui::Combo("Timeframe", &selected, timeframes,
                     IM_ARRAYSIZE(timeframes))) {
      selected_timeframe_ = static_cast<RenderEngine::TimeFrame>(selected);
    }

    ImGui::Separator();

    // Create chart button
    if (ImGui::Button("Create New Chart")) {
      // For now, create chart for first available symbol
      auto &instruments = bridge_->GetAllInstruments();
      if (!instruments.empty()) {
        chart_manager_->create_chart(instruments.begin()->first,
                                     selected_timeframe_);
      }
    }

    ImGui::Separator();

    // Chart list
    ImGui::Text("Active Charts: %zu", chart_manager_->get_charts().size());
    for (const auto &[id, chart] : chart_manager_->get_charts()) {
      std::string chart_label =
          chart.symbol + " (" +
          std::to_string(static_cast<int>(chart.timeframe)) + ")";
      if (ImGui::Checkbox(chart_label.c_str(),
                          &const_cast<ChartInstance &>(chart).visible)) {
        chart_manager_->toggle_chart_visibility(id);
      }
    }
  }
  ImGui::End();
}

void QuantWorkspaceComponent::render_indicator_selector() {
  ImGui::SetNextWindowPos(ImVec2(270, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Indicator Selector", &show_indicator_selector_)) {
    // Default indicator configuration for all charts
    static IndicatorConfig global_config;

    ImGui::Text("Global Indicators");
    ImGui::Separator();

    ImGui::Checkbox("Show SMA 10", &global_config.show_sma_10);
    ImGui::Checkbox("Show SMA 20", &global_config.show_sma_20);
    ImGui::Checkbox("Show SMA 50", &global_config.show_sma_50);
    ImGui::Checkbox("Show EMA 10", &global_config.show_ema_10);
    ImGui::Checkbox("Show EMA 20", &global_config.show_ema_20);
    ImGui::Checkbox("Show EMA 50", &global_config.show_ema_50);
    ImGui::Checkbox("Show RSI", &global_config.show_rsi);
    ImGui::Checkbox("Show MACD", &global_config.show_macd);
    ImGui::Checkbox("Show Bollinger Bands", &global_config.show_bollinger);
    ImGui::Checkbox("Show Stochastic", &global_config.show_stochastic);
    ImGui::Checkbox("Show Waddah Attar Explosion",
                    &global_config.show_waddah_explosion);

    // Apply to all charts
    if (ImGui::Button("Apply to All")) {
      for (const auto &chart : chart_manager_->get_charts()) {
        indicator_configs_[chart.first] = global_config;
      }
    }
  }
  ImGui::End();
}

void QuantWorkspaceComponent::render_instrument_chart(
    uint32_t chart_id, const std::string &symbol, const InstrumentStore &inst,
    RenderEngine::TimeFrame timeframe) {
  std::lock_guard<std::mutex> inst_lock(inst.data_mutex);

  if (inst.timestamps.empty()) {
    ImGui::Text("Initializing Stream for %s...", symbol.c_str());
    return;
  }

  // Get indicator configuration for this chart
  auto &indicator_config = indicator_configs_[chart_id];

  // Prepare indicator parameters
  std::vector<IndicatorParams> indicators;

  if (indicator_config.show_sma_10) {
    indicators.push_back({IndicatorType::SMA_10,
                          10,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.94f, 1.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_sma_20) {
    indicators.push_back({IndicatorType::SMA_20,
                          20,
                          0,
                          0,
                          2.0,
                          {1.0f, 0.0f, 0.3f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_sma_50) {
    indicators.push_back({IndicatorType::SMA_50,
                          50,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.5f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_10) {
    indicators.push_back({IndicatorType::EMA_10,
                          10,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.5f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_20) {
    indicators.push_back({IndicatorType::EMA_20,
                          20,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.0f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_50) {
    indicators.push_back({IndicatorType::EMA_50,
                          50,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.5f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_rsi) {
    indicators.push_back({IndicatorType::RSI_14,
                          14,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.8f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_macd) {
    indicators.push_back({IndicatorType::MACD,
                          12,
                          26,
                          9,
                          2.0,
                          {0.8f, 0.4f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_bollinger) {
    indicators.push_back({IndicatorType::BOLLINGER_MID,
                          20,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.6f, 0.6f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_stochastic) {
    indicators.push_back({IndicatorType::STOCHASTIC_K,
                          14,
                          3,
                          0,
                          2.0,
                          {0.6f, 0.0f, 0.6f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_waddah_explosion) {
    indicators.push_back({IndicatorType::WADDAH_ATTAR_EXPLOSION,
                          20,
                          40,
                          150,
                          2.0,
                          {1.0f, 1.0f, 0.0f, 1.0f},
                          1.5f,
                          true});
  }

  // Cyber-Cyan: #00F0FF (0xFFFFF000), Neon-Red: #FF0033 (0xFF3300FF)
  ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::GetColorU32(ImVec4(
                                             0.0f, 0.94f, 1.0f, 1.0f))); // Cyan

  if (ImPlot::BeginPlot(symbol.c_str(), ImVec2(-1, -1),
                        ImPlotFlags_CanvasOnly | ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxis(ImAxis_Y2, "Volume",
                      ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_NoGridLines |
                          ImPlotAxisFlags_NoTickLabels);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y2, 0,
                                       1000000); // For Volume alignment

    const double *dates = inst.timestamps.data();
    const double *opens = inst.opens.data();
    const double *closes = inst.closes.data();
    const double *lows = inst.lows.data();
    const double *highs = inst.highs.data();
    int count = (int)inst.timestamps.size();

    // Plot 1: Candlesticks (Manual high-perf implementation)
    if (ImPlot::BeginItem("OHLC")) {
      ImDrawList *draw_list = ImPlot::GetPlotDrawList();
      double width = 0.25;
      if (count > 1) {
        width = (dates[1] - dates[0]) * 0.25;
      }

      for (int i = 0; i < count; ++i) {
        ImVec2 open_pos = ImPlot::PlotToPixels(dates[i] - width, opens[i]);
        ImVec2 close_pos = ImPlot::PlotToPixels(dates[i] + width, closes[i]);
        ImVec2 low_pos = ImPlot::PlotToPixels(dates[i], lows[i]);
        ImVec2 high_pos = ImPlot::PlotToPixels(dates[i], highs[i]);

        // Neon Red for down, Cyber Cyan for up
        ImU32 color = (opens[i] > closes[i])
                          ? ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.2f, 1.0f))
                          : // Neon Red
                          ImGui::GetColorU32(
                              ImVec4(0.0f, 0.94f, 1.0f, 1.0f)); // Cyber Cyan

        draw_list->AddLine(low_pos, high_pos, color);
        draw_list->AddRectFilled(open_pos, close_pos, color);

        ImPlot::FitPoint(ImPlotPoint(dates[i], lows[i]));
        ImPlot::FitPoint(ImPlotPoint(dates[i], highs[i]));
      }
      ImPlot::EndItem();
    }

    // Plot 2: Volume Profile (PlotBarsH on Y-axis)
    if (!inst.m_vol_profile.empty()) {
      std::vector<double> vp_prices;
      std::vector<double> vp_volumes;
      for (auto const &[price, vol] : inst.m_vol_profile) {
        vp_prices.push_back(price);
        vp_volumes.push_back(vol);
      }

      ImPlot::SetAxis(ImAxis_Y1); // Align to Price Axis
      ImPlot::SetNextFillStyle(
          ImVec4(0.0f, 0.94f, 1.0f, 0.3f)); // Transparent Cyan
      ImPlot::PlotBars("VolProfile", vp_prices.data(), vp_volumes.data(),
                       (int)vp_prices.size(), 0.5, ImPlotBarsFlags_Horizontal);
    }

    // Plot indicators
    indicator_renderer_->render_indicators(inst.symbol_id, timeframe,
                                           indicators);

    // Capture viewport state for GPU synchronization
    auto *chart_ptr =
        const_cast<ChartInstance *>(chart_manager_->get_chart(chart_id));
    if (chart_ptr) {
      ImPlotRect limits = ImPlot::GetPlotLimits();
      chart_ptr->x_min = limits.X.Min;
      chart_ptr->x_max = limits.X.Max;
      chart_ptr->y_min = limits.Y.Min;
      chart_ptr->y_max = limits.Y.Max;
      chart_ptr->plot_pos = ImPlot::GetPlotPos();
      chart_ptr->plot_size = ImPlot::GetPlotSize();
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor();
}

void QuantWorkspaceComponent::render(VkCommandBuffer cmd) {
  if (candle_pipeline_ == VK_NULL_HANDLE || !viz_engine_ || !core_)
    return;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, candle_pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          candle_pipeline_layout_, 0, 1,
                          &candle_descriptor_set_, 0, nullptr);

  auto extent = core_->get_swapchain_extent();
  glm::mat4 projection =
      glm::ortho(0.0f, static_cast<float>(extent.width),
                 static_cast<float>(extent.height), 0.0f, -1.0f, 1.0f);

  for (const auto &chart : chart_manager_->get_visible_charts()) {
    uint32_t symbol_id = bridge_->GetSymbolId(chart.symbol);
    auto candles = processor_->getCandles(symbol_id, chart.timeframe);
    auto current_candle =
        processor_->getCurrentCandle(symbol_id, chart.timeframe);
    if (current_candle) {
      candles.push_back(*current_candle);
    }

    if (candles.empty())
      continue;

    // Update GPU buffer with relative timestamps to preserve precision
    // We use the first candle or x_min as base
    double base_x = chart.x_min;
    viz_engine_->updateChartData(chart.chart_id, candles, base_x);

    uint32_t chart_index =
        static_cast<uint32_t>(viz_engine_->getSymbolIndex(chart.chart_id));

    // --- Compute Phase: Waddah Attar Explosion ---
    if (indicator_configs_[chart.chart_id].show_waddah_explosion &&
        wae_pipeline_ != VK_NULL_HANDLE) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, wae_pipeline_);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              wae_pipeline_layout_, 0, 1, &wae_descriptor_set_,
                              0, nullptr);

      RenderEngine::WAEPushConstants wae_pc{};
      wae_pc.count = static_cast<uint32_t>(candles.size());
      wae_pc.sens = 150;   // Default sensitivity
      wae_pc.fast = 20;    // Default fast EMA
      wae_pc.slow = 40;    // Default slow EMA
      wae_pc.channel = 20; // Default Bollinger
      wae_pc.mult = 2.0f;  // Multiplier
      wae_pc.chart_offset = chart_index * 10000;

      vkCmdPushConstants(cmd, wae_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(RenderEngine::WAEPushConstants), &wae_pc);

      uint32_t groupCount = (wae_pc.count + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);

      // Barrier: Compute Write -> Vertex Read
      VkBufferMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = viz_engine_->getIndicatorBuffer();
      barrier.offset = wae_pc.chart_offset * sizeof(RenderEngine::WAEDataGPU);
      barrier.size = wae_pc.count * sizeof(RenderEngine::WAEDataGPU);

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr,
                           1, &barrier, 0, nullptr);
    }

    // --- Graphics Phase: Candles ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, candle_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            candle_pipeline_layout_, 0, 1,
                            &candle_descriptor_set_, 0, nullptr);

    // Setup Push Constants
    CandlePushConstants pc{};
    pc.projection = projection;
    pc.chart_min = glm::vec2(0.0f, chart.y_min); // chart_min.x is now relative
    pc.chart_max = glm::vec2(chart.x_max - chart.x_min, chart.y_max);

    // Convert ImVec2 to glm::vec2
    pc.viewport_offset = glm::vec2(chart.plot_pos.x, chart.plot_pos.y);
    pc.viewport_size = glm::vec2(chart.plot_size.x, chart.plot_size.y);

    // Scale candle width based on zoom level (visible range)
    float range_x = static_cast<float>(chart.x_max - chart.x_min);
    pc.candle_width = (pc.viewport_size.x / range_x) * 0.8f;

    pc.chart_offset = chart_index * 10000;

    vkCmdPushConstants(cmd, candle_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(CandlePushConstants), &pc);

    // Draw! (12 vertices per candle)
    vkCmdDraw(cmd, 12, static_cast<uint32_t>(candles.size()), 0, 0);

    // --- Graphics Phase: WAE Indicators ---
    if (indicator_configs_[chart.chart_id].show_waddah_explosion &&
        wae_graphics_pipeline_ != VK_NULL_HANDLE) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        wae_graphics_pipeline_);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              wae_graphics_pipeline_layout_, 0, 1,
                              &wae_graphics_descriptor_set_, 0, nullptr);

      vkCmdPushConstants(cmd, wae_graphics_pipeline_layout_,
                         VK_SHADER_STAGE_VERTEX_BIT, 0,
                         sizeof(CandlePushConstants), &pc);

      // Draw WAE (12 vertices per indicator bar set)
      vkCmdDraw(cmd, 12, static_cast<uint32_t>(candles.size()), 0, 0);
    }
  }
}

void QuantWorkspaceComponent::clear_data() {
  chart_manager_.reset();
  indicator_renderer_.reset();
  indicator_configs_.clear();
}

} // namespace BTQuant
