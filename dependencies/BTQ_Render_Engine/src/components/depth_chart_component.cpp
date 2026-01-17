/**
 * BTQuant Advanced Vulkan Dashboard - Market Depth Chart Component
 *
 * Visualizes the cumulative bid/ask depth (liquidity) at each price level.
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>

namespace BTQuant {

MarketDepthChartComponent::MarketDepthChartComponent(const glm::vec2 &position,
                                                     const glm::vec2 &size)
    : UIComponent(position, size) {
  current_data_.bids.clear();
  current_data_.asks.clear();
}

void MarketDepthChartComponent::clear_data() {
  current_data_.bids.clear();
  current_data_.asks.clear();
  mark_dirty();
}

MarketDepthChartComponent::~MarketDepthChartComponent() {
  if (vulkan_core_) {
    vulkan_core_->get_memory_manager().deallocate_buffer(vertex_buffer_);
    if (pipeline_ != VK_NULL_HANDLE) {
      vkDestroyPipeline(vulkan_core_->get_device(), pipeline_, nullptr);
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(vulkan_core_->get_device(), pipeline_layout_,
                              nullptr);
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(vulkan_core_->get_device(),
                                   descriptor_set_layout_, nullptr);
    }
  }
}

void MarketDepthChartComponent::handle_trade(const RenderEngine::TradeData &) {
  // Depth chart doesn't typically react to individual trades directly,
  // but could show trade markers.
}

void MarketDepthChartComponent::handle_orderbook(
    const RenderEngine::OrderbookData &orderbook) {
  if (orderbook.symbol != target_symbol_)
    return;

  // Convert RenderEngine::OrderbookData to OrderBookData
  current_data_.bids.clear();
  for (const auto &b : orderbook.bids) {
    current_data_.bids.push_back(
        {.price = b.price, .size = b.size, .total_size = 0.0});
  }
  current_data_.asks.clear();
  for (const auto &a : orderbook.asks) {
    current_data_.asks.push_back(
        {.price = a.price, .size = a.size, .total_size = 0.0});
  }
  // Recalculate totals...
  mark_dirty();
}

void MarketDepthChartComponent::update(float) {
  if (is_dirty()) {
    rebuild_geometry();
    dirty_frames_--;
  }
}

void MarketDepthChartComponent::render(VkCommandBuffer) {
  if (!visible_ || vertex_count_ == 0)
    return;

  // Render logic...
  // For now, let's focus on the geometry and GUI.
}

void MarketDepthChartComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Market Depth Chart", &visible_)) {
    // Basic plot while Vulkan path is being integrated
    if (!current_data_.bids.empty() || !current_data_.asks.empty()) {
      std::vector<float> prices, depths;
      // TODO: Use ImGui::PlotHistogram or custom drawing for depth
      ImGui::Text("Bid/Ask Depth Visualization");
    } else {
      ImGui::Text("No depth data available");
    }
  }
  ImGui::End();
}

void MarketDepthChartComponent::handle_input(const InputEvent &) {
  // Tooltip logic based on mouse position
}

void MarketDepthChartComponent::initialize_vulkan_resources(
    VulkanCore *vulkan_core) {
  vulkan_core_ = vulkan_core;
  // Initialize pipeline and layout...
}

void MarketDepthChartComponent::rebuild_geometry() {
  if (current_data_.bids.empty() && current_data_.asks.empty())
    return;

  std::vector<CandlestickVertex>
      vertices; // Reusing vertex structure for consistency

  // Find price range etc.
  float min_p = current_data_.bids.back().price;
  float max_p = current_data_.asks.back().price;
  float p_range = max_p - min_p;

  double max_depth = 0;
  if (!current_data_.bids.empty())
    max_depth = std::max(max_depth, current_data_.bids.back().total_size);
  if (!current_data_.asks.empty())
    max_depth = std::max(max_depth, current_data_.asks.back().total_size);

  float y_base = position_.y + size_.y;

  // Render Bids (Cumulative)
  for (size_t i = 0; i < current_data_.bids.size(); ++i) {
    const auto &level = current_data_.bids[i];
    float x_rel = (level.price - min_p) / p_range;
    float x = position_.x + x_rel * size_.x;
    float depth_h = (level.total_size / max_depth) * size_.y;

    glm::vec4 color = theme_.price_up;
    color.a = 0.4f;

    // Triangle 1
    vertices.push_back({{x, y_base}, {0, 0}, color, 0, 0});
    vertices.push_back({{x, y_base - depth_h}, {0, 0}, color, 0, 0});
    // ... logic for polygon ...
  }

  vertex_count_ = static_cast<uint32_t>(vertices.size());
}

} // namespace BTQuant
