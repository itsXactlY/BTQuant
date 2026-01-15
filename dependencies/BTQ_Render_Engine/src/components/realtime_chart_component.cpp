/**
 * BTQuant Advanced Vulkan Dashboard - Realtime Chart Component Implementation
 *
 * High-performance real-time price chart component with GPU-accelerated
 * rendering, smooth line drawing, candlestick support, and advanced visual
 * effects.
 *
 * Features:
 * - GPU-accelerated line rendering with anti-aliasing
 * - Real-time data streaming with minimal latency
 * - Candlestick and line chart modes
 * - Auto-scaling and manual Y-axis control
 * - Smooth animations and transitions
 * - Volume overlay support
 * - Professional styling with gradients
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <cmath>

namespace BTQuant {

RealtimeChartComponent::RealtimeChartComponent(const glm::vec2 &position,
                                               const glm::vec2 &size)
    : UIComponent(position, size) {

  // Initialize with reasonable defaults
  time_window_ = 60.0f; // 1 minute default
  min_y_ = 0.0f;
  max_y_ = 100.0f;
  auto_scale_ = true;
  candlestick_mode_ = false;
  line_color_ = theme_.accent_primary;

  // Initialize indicators
  indicators_.push_back(std::make_unique<EMAIndicator>(9));
  indicators_.push_back(std::make_unique<EMAIndicator>(21));
  indicators_.push_back(std::make_unique<SMAIndicator>(50));
  indicators_.push_back(std::make_unique<RSIIndicator>(14));
  indicators_.push_back(std::make_unique<MACDIndicator>(12, 26, 9));
}

RealtimeChartComponent::~RealtimeChartComponent() {
  if (vulkan_core_) {
    VkDevice device = vulkan_core_->get_device();
    if (line_pipeline_ != VK_NULL_HANDLE)
      vkDestroyPipeline(device, line_pipeline_, nullptr);
    if (candlestick_pipeline_ != VK_NULL_HANDLE)
      vkDestroyPipeline(device, candlestick_pipeline_, nullptr);
    if (line_pipeline_layout_ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, line_pipeline_layout_, nullptr);
    if (ui_pipeline_layout_ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, ui_pipeline_layout_, nullptr);
    if (line_layout_ != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, line_layout_, nullptr);
    if (ui_layout_ != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, ui_layout_, nullptr);
  }
}

void RealtimeChartComponent::add_data_point(float timestamp, float value,
                                            float volume) {
  DataPoint point = {timestamp, value, volume};
  data_points_.push_back(point);

  // Remove old data points outside the time window
  float cutoff_time = timestamp - time_window_;
  while (!data_points_.empty() && data_points_[0].timestamp < cutoff_time) {
    data_points_.pop_front();
  }

  // Update Y range if auto-scaling is enabled
  if (auto_scale_) {
    update_y_range();
  }

  mark_dirty();
}

void RealtimeChartComponent::set_time_window(float seconds) {
  time_window_ = seconds;

  // Remove data points outside the new time window
  if (!data_points_.empty()) {
    float cutoff_time = data_points_.back().timestamp - time_window_;
    while (!data_points_.empty() &&
           data_points_.front().timestamp < cutoff_time) {
      data_points_.pop_front();
    }
  }

  mark_dirty();
}

void RealtimeChartComponent::set_y_range(float min_y, float max_y) {
  min_y_ = min_y;
  max_y_ = max_y;
  auto_scale_ = false;
  mark_dirty();
}

void RealtimeChartComponent::enable_candlestick_mode(bool enable) {
  if (candlestick_mode_ != enable) {
    candlestick_mode_ = enable;
    mark_dirty();
  }
}

void RealtimeChartComponent::update(float delta_time) {
  if (is_dirty()) {
    update_indicators();
    if (candlestick_mode_) {
      rebuild_candlestick_geometry();
      rebuild_indicator_geometry();
    } else {
      rebuild_line_geometry();
    }
    dirty_frames_--;
  }

  // Crosshair is rebuilt every frame if active
  rebuild_crosshair_geometry();

  // Update any animations or smooth transitions
  static float animation_time = 0.0f;
  animation_time += delta_time;

  // TODO: Implement smooth data transitions and animations
}

void RealtimeChartComponent::handle_trade(
    const RenderEngine::TradeData &trade) {
  if (dashboard_ && trade.symbol != dashboard_->get_active_symbol())
    return;

  float ts = static_cast<float>(trade.timestamp_us % 1000000000) / 1000.0f;
  add_data_point(ts, static_cast<float>(trade.price),
                 static_cast<float>(trade.size));

  // Aggregate into 5-second candles for the visual chart
  uint64_t candle_interval_us = 5 * 1000000;
  uint64_t bucket =
      trade.timestamp_us - (trade.timestamp_us % candle_interval_us);

  if (candles_.empty() || candles_.back().timestamp_us != bucket) {
    Candle new_candle;
    new_candle.timestamp_us = bucket;
    new_candle.open = static_cast<float>(trade.price);
    new_candle.high = static_cast<float>(trade.price);
    new_candle.low = static_cast<float>(trade.price);
    new_candle.close = static_cast<float>(trade.price);
    new_candle.volume = static_cast<float>(trade.size);
    candles_.push_back(new_candle);
  } else {
    Candle &latest = candles_.back();
    latest.high = std::max(latest.high, static_cast<float>(trade.price));
    latest.low = std::min(latest.low, static_cast<float>(trade.price));
    latest.close = static_cast<float>(trade.price);
    latest.volume += static_cast<float>(trade.size);
  }
}

void RealtimeChartComponent::handle_orderbook(
    const RenderEngine::OrderbookData &orderbook) {
  if (dashboard_ && orderbook.symbol != dashboard_->get_active_symbol())
    return;
  // Real-time chart usually tracks price from trades, but could also track
  // mid-price
}

void RealtimeChartComponent::render(VkCommandBuffer cmd) {
  if (!visible_)
    return;

  uint32_t frame_idx = vulkan_core_->get_current_frame_index();
  VkExtent2D extent = vulkan_core_->get_swapchain_extent();

  if (candlestick_mode_ == false) { // Line mode
    if (line_vertex_buffers_[frame_idx].buffer) {
      // Update Line UBO
      if (line_ubo_buffer_.mapped_ptr) {
        ChartUniformBuffer ubo{};
        ubo.projection = glm::ortho(0.0f, (float)extent.width, 0.0f,
                                    (float)extent.height, -1.0f, 1.0f);
        ubo.view = glm::mat4(1.0f);
        ubo.viewport_size = glm::vec2(extent.width, extent.height);
        ubo.chart_bounds_min = glm::vec2(0, min_y_);
        ubo.chart_bounds_max = glm::vec2(time_window_, max_y_);
        ubo.data_range = glm::vec2(min_y_, max_y_);
        ubo.line_thickness_scale = 1.0f;
        ubo.anti_alias_width = 1.0f;
        memcpy(line_ubo_buffer_.mapped_ptr, &ubo, sizeof(ubo));
      }

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline_);

      if (line_descriptor_set_ != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                line_pipeline_layout_, 0, 1,
                                &line_descriptor_set_, 0, nullptr);
      }

      VkBuffer buffers[] = {line_vertex_buffers_[frame_idx].buffer};
      VkDeviceSize offsets[] = {line_vertex_buffers_[frame_idx].offset};
      vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);

      uint32_t vertex_count = static_cast<uint32_t>(
          line_vertex_buffers_[frame_idx].size / sizeof(LineVertex));
      if (vertex_count > 0) {
        vkCmdDraw(cmd, vertex_count, 1, 0, 0);
      }
    }
  } else { // Candlestick mode
    if (candlestick_vertex_buffers_[frame_idx].buffer) {
      // Update UI UBO
      if (ui_ubo_buffer_.mapped_ptr) {
        UIUniformBuffer ubo{};
        // Swap 0.0f and extent.height to match Vulkan NDC Y direction
        ubo.projection = glm::ortho(0.0f, (float)extent.width, 0.0f,
                                    (float)extent.height, -1.0f, 1.0f);
        ubo.view = glm::mat4(1.0f);
        ubo.model = glm::mat4(1.0f);
        ubo.viewport_size = glm::vec2(extent.width, extent.height);
        ubo.dpi_scale = glm::vec2(1.0f);
        ubo.global_tint = glm::vec4(1.0f);
        memcpy(ui_ubo_buffer_.mapped_ptr, &ubo, sizeof(ubo));
      }

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        candlestick_pipeline_);

      struct {
        glm::vec2 offset;
        glm::vec2 scale;
      } ui_push;
      ui_push.offset = glm::vec2(0, 0);
      ui_push.scale = glm::vec2(1, 1);

      vkCmdPushConstants(cmd, ui_pipeline_layout_,
                         VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(ui_push), &ui_push);

      if (ui_descriptor_set_ != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                ui_pipeline_layout_, 0, 1, &ui_descriptor_set_,
                                0, nullptr);
      }

      VkBuffer v_buffers[] = {candlestick_vertex_buffers_[frame_idx].buffer};
      VkDeviceSize v_offsets[] = {
          candlestick_vertex_buffers_[frame_idx].offset};
      vkCmdBindVertexBuffers(cmd, 0, 1, v_buffers, v_offsets);

      if (candlestick_vertex_count_ > 0) {
        vkCmdDraw(cmd, candlestick_vertex_count_, 1, 0, 0);
      }
    }

    // Indicators
    if (indicator_vertex_buffers_[frame_idx].buffer &&
        indicator_vertex_count_ > 0) {
      VkBuffer ind_buffers[] = {indicator_vertex_buffers_[frame_idx].buffer};
      VkDeviceSize ind_offsets[] = {
          indicator_vertex_buffers_[frame_idx].offset};
      vkCmdBindVertexBuffers(cmd, 0, 1, ind_buffers, ind_offsets);
      vkCmdDraw(cmd, indicator_vertex_count_, 1, 0, 0);
    }
  }

  // Crosshair
  if (show_crosshair_ && crosshair_vertex_buffers_[frame_idx].buffer &&
      crosshair_vertex_count_ > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline_);
    VkBuffer cs_buffers[] = {crosshair_vertex_buffers_[frame_idx].buffer};
    VkDeviceSize cs_offsets[] = {crosshair_vertex_buffers_[frame_idx].offset};
    vkCmdBindVertexBuffers(cmd, 0, 1, cs_buffers, cs_offsets);
    vkCmdDraw(cmd, crosshair_vertex_count_, 1, 0, 0);
  }
}

void RealtimeChartComponent::render_gui() {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_FirstUseEver);

  if (!ImGui::Begin("Price Chart", &visible_, ImGuiWindowFlags_NoScrollbar)) {
    ImGui::End();
    ImGui::PopStyleVar();
    return;
  }

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  ImVec2 pos = ImGui::GetWindowPos();
  ImVec2 size = ImGui::GetWindowSize();

  // 0. In-Chart Toolbar (Top)
  ImGui::SetCursorPos(ImVec2(10, 5));
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
  ImGui::Text("%s", chart_symbol_.c_str());
  ImGui::PopStyleColor();

  ImGui::SameLine(100);
  const char *timeframes[] = {"1m", "5m", "15m", "1h", "4h", "1d"};
  if (ImGui::BeginCombo("##Timeframe", current_timeframe_.c_str(),
                        ImGuiComboFlags_None)) {
    for (int n = 0; n < IM_ARRAYSIZE(timeframes); n++) {
      bool is_selected = (current_timeframe_ == timeframes[n]);
      if (ImGui::Selectable(timeframes[n], is_selected)) {
        current_timeframe_ = timeframes[n];
        mark_dirty();
      }
      if (is_selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine(size.x - 220);
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.6f, 0.2f, 0.8f));
  if (ImGui::Button("BUY", ImVec2(60, 22))) {
    // TODO: Open quick order entry
  }
  ImGui::PopStyleColor();

  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 0.8f));
  if (ImGui::Button("SELL", ImVec2(60, 22))) {
    // TODO: Open quick order entry
  }
  ImGui::PopStyleColor();

  ImGui::SameLine();
  if (ImGui::Button("INDICATORS")) {
    // TODO: Open indicator settings
  }

  // 1. Draw Grid Lines
  float grid_color = ImGui::GetColorU32(ImGuiCol_Border, 0.3f);
  int horizontal_lines = 5;
  for (int i = 0; i <= horizontal_lines; ++i) {
    float y = pos.y + (size.y * 0.85f) * (float)i / (float)horizontal_lines;
    draw_list->AddLine(ImVec2(pos.x, y), ImVec2(pos.x + size.x - 60, y),
                       grid_color);
  }

  // 2. Y-Axis Price Labels (Right-aligned)
  for (int i = 0; i <= horizontal_lines; ++i) {
    float price =
        max_y_ - (max_y_ - min_y_) * (float)i / (float)horizontal_lines;
    char label[32];
    snprintf(label, sizeof(label), "%.2f", price);
    float y = pos.y + (size.y * 0.85f) * (float)i / (float)horizontal_lines;
    draw_list->AddText(ImVec2(pos.x + size.x - 55, y - 7),
                       ImGui::GetColorU32(ImGuiCol_Text), label);
  }

  // 3. Current Price Tag
  if (!candles_.empty()) {
    float last_price = candles_.back().close;
    float rel_y = (last_price - min_y_) / (max_y_ - min_y_);
    float y = pos.y + (size.y * 0.85f) * (1.0f - rel_y);

    draw_list->AddRectFilled(
        ImVec2(pos.x + size.x - 60, y - 10), ImVec2(pos.x + size.x, y + 10),
        ImGui::GetColorU32(ImVec4(0.2f, 0.6f, 1.0f, 0.8f)));
    char price_str[32];
    snprintf(price_str, sizeof(price_str), "%.2f", last_price);
    draw_list->AddText(ImVec2(pos.x + size.x - 55, y - 7),
                       ImGui::GetColorU32(ImVec4(1, 1, 1, 1)), price_str);
  }

  // 4. Volume Sub-panel Label/Marker
  draw_list->AddLine(ImVec2(pos.x, pos.y + size.y * 0.85f),
                     ImVec2(pos.x + size.x - 60, pos.y + size.y * 0.85f),
                     ImGui::GetColorU32(ImVec4(0.5f, 0.5f, 0.5f, 1.0f)));
  // 5. Candle Tooltip on Hover
  if (ImGui::IsWindowHovered()) {
    const auto &cs = dashboard_->get_crosshair_state();
    if (cs.active) {
      // Find closest candle to cs.timestamp_us
      const Candle *target = nullptr;
      for (const auto &c : candles_) {
        if (std::abs((int64_t)c.timestamp_us - (int64_t)cs.timestamp_us) <
            2500000) {
          target = &c;
          break;
        }
      }

      if (target) {
        ImGui::BeginTooltip();
        ImGui::Text("Time: %s", "00:00:00"); // TODO: Format timestamp
        ImGui::Separator();
        ImGui::Text("Open:  %.2f", target->open);
        ImGui::Text("High:  %.2f", target->high);
        ImGui::Text("Low:   %.2f", target->low);
        ImGui::Text("Close: %.2f", target->close);
        ImGui::Text("Vol:   %.2f", target->volume);
        ImGui::EndTooltip();
      }
    }
  }

  ImGui::End();
  ImGui::PopStyleVar();
}

void RealtimeChartComponent::handle_input(const InputEvent &event) {
  glm::vec2 mouse_pos = event.position;
  bool inside =
      mouse_pos.x >= position_.x && mouse_pos.x <= position_.x + size_.x &&
      mouse_pos.y >= position_.y && mouse_pos.y <= position_.y + size_.y;

  switch (event.type) {
  case InputEventType::MouseMove:
    if (inside) {
      float rel_x = (mouse_pos.x - position_.x) / size_.x;
      float rel_y = 1.0f - (mouse_pos.y - position_.y) / size_.y;

      if (!candles_.empty()) {
        double t_min = candles_.front().timestamp_us;
        double t_max = candles_.back().timestamp_us;
        double ts = t_min + rel_x * (t_max - t_min);
        double p = min_y_ + rel_y * (max_y_ - min_y_);

        CrosshairState state;
        state.active = true;
        state.timestamp_us = ts;
        state.price = p;
        state.screen_pos = mouse_pos;
        state.source = this;
        if (dashboard_)
          dashboard_->synchronize_crosshair(state);
      }
    }

    if (is_dragging_) {
      glm::vec2 delta = mouse_pos - last_mouse_pos_;
      // Pan logic - for now just affecting view offset
      view_offset_ -= delta.x * (time_window_ / size_.x);
      mark_dirty();
    }
    last_mouse_pos_ = mouse_pos;
    break;

  case InputEventType::Scroll:
    if (inside && event.scroll_delta.y != 0.0f) {
      float zoom_delta = 1.0f - event.scroll_delta.y * 0.1f;
      view_zoom_ *= zoom_delta;
      view_zoom_ = std::max(0.01f, std::min(10.0f, view_zoom_));
      mark_dirty();
    }
    break;

  case InputEventType::MouseButton:
    if (event.mouse_button == MouseButton::Left) {
      is_dragging_ = event.pressed;
    } else if (event.pressed && event.mouse_button == MouseButton::Middle) {
      view_zoom_ = 1.0f;
      view_offset_ = 0.0f;
      auto_scale_ = true;
      update_y_range();
      mark_dirty();
    }
    break;

  default:
    break;
  }
}

void RealtimeChartComponent::rebuild_line_geometry() {
  if (data_points_.size() < 2)
    return;

  std::vector<LineVertex> vertices;
  vertices.reserve((data_points_.size() - 1) *
                   6); // 2 triangles per line segment

  // Calculate time range for X mapping
  float time_min = data_points_.front().timestamp;
  float time_max = data_points_.back().timestamp;
  float time_range = time_max - time_min;

  if (time_range <= 0.0f)
    return;

  // Generate line segments with proper thickness and anti-aliasing
  for (size_t i = 0; i < data_points_.size() - 1; ++i) {
    const auto &p1 = data_points_[i];
    const auto &p2 = data_points_[i + 1];

    // Map data points to screen coordinates
    float x1 = position_.x + ((p1.timestamp - time_min) / time_range) * size_.x;
    float y1 = position_.y + size_.y -
               ((p1.value - min_y_) / (max_y_ - min_y_)) * size_.y;
    float x2 = position_.x + ((p2.timestamp - time_min) / time_range) * size_.x;
    float y2 = position_.y + size_.y -
               ((p2.value - min_y_) / (max_y_ - min_y_)) * size_.y;

    glm::vec2 pos1(x1, y1);
    glm::vec2 pos2(x2, y2);
    glm::vec2 direction = glm::normalize(pos2 - pos1);
    glm::vec2 normal(-direction.y, direction.x);

    float thickness = 2.0f;
    float half_thickness = thickness * 0.5f;

    // Create quad for line segment
    glm::vec2 v1 = pos1 - normal * half_thickness;
    glm::vec2 v2 = pos1 + normal * half_thickness;
    glm::vec2 v3 = pos2 + normal * half_thickness;
    glm::vec2 v4 = pos2 - normal * half_thickness;

    // Color interpolation based on price change
    glm::vec4 color = line_color_;
    if (i > 0) {
      float price_change = p2.value - p1.value;
      if (price_change > 0) {
        color = glm::mix(line_color_, theme_.price_up, 0.3f);
      } else if (price_change < 0) {
        color = glm::mix(line_color_, theme_.price_down, 0.3f);
      }
    }

    // First triangle
    vertices.push_back({v1, direction, thickness, color, 0.0f});
    vertices.push_back({v2, direction, thickness, color, 0.0f});
    vertices.push_back(
        {v3, direction, thickness, color, glm::length(pos2 - pos1)});

    // Second triangle
    vertices.push_back({v1, direction, thickness, color, 0.0f});
    vertices.push_back(
        {v3, direction, thickness, color, glm::length(pos2 - pos1)});
    vertices.push_back(
        {v4, direction, thickness, color, glm::length(pos2 - pos1)});
  }

  if (vertices.empty()) {
    return;
  }

  size_t buffer_size = vertices.size() * sizeof(LineVertex);

  // Reallocate buffer if necessary
  if (!line_vertex_buffers_[0].buffer ||
      line_vertex_buffers_[0].size < buffer_size) {
    for (int i = 0; i < 2; ++i) {
      if (line_vertex_buffers_[i].buffer) {
        vulkan_core_->get_memory_manager().deallocate_buffer(
            line_vertex_buffers_[i]);
      }
      line_vertex_buffers_[i] =
          vulkan_core_->get_memory_manager().allocate_vertex_buffer(
              buffer_size);
    }
  }

  // Copy data to GPU-mapped memory for both frames
  for (int i = 0; i < 2; ++i) {
    if (line_vertex_buffers_[i].mapped_ptr) {
      memcpy(line_vertex_buffers_[i].mapped_ptr, vertices.data(), buffer_size);
    }
  }
}

void RealtimeChartComponent::rebuild_candlestick_geometry() {
  if (candles_.empty())
    return;

  std::vector<CandlestickVertex> vertices;

  double latest_t = (double)candles_.back().timestamp_us;
  double base_range = (double)candles_.size() * 5000000.0;
  double t_max = latest_t + 5000000.0 + (double)view_offset_ * 1000000.0;
  double t_min = t_max - base_range * (double)view_zoom_;
  double t_range = t_max - t_min;

  // Account for the interval of the last candle
  float interval = 5000000.0f; // 5s in us
  t_range += interval;

  float candle_width = (size_.x / (float)candles_.size()) / (float)view_zoom_;
  float bar_width = candle_width * 0.8f;

  float max_volume = 0.0f;
  for (const auto &c : candles_)
    max_volume = std::max(max_volume, c.volume);
  if (max_volume <= 0)
    max_volume = 1.0f;

  // LOD 2: Adaptive Stride Aggregation
  int stride = 1;
  if (candle_width < 1.0f) {
    stride = (int)(1.0f / candle_width) + 1;
  }

  for (size_t i = 0; i < candles_.size(); i += stride) {
    const auto &c_base = candles_[i];

    // Aggregate High/Low over stride
    float high = c_base.high;
    float low = c_base.low;
    float open = c_base.open;
    float close = c_base.close;
    float volume = c_base.volume;

    for (int k = 1; k < stride && (i + k) < candles_.size(); ++k) {
      high = std::max(high, candles_[i + k].high);
      low = std::min(low, candles_[i + k].low);
      close = candles_[i + k].close;
      volume += candles_[i + k].volume;
    }

    // LOD 1: Time-based Culling
    if ((double)c_base.timestamp_us + (double)interval * stride < t_min ||
        (double)c_base.timestamp_us > t_max) {
      continue;
    }

    float x =
        position_.x +
        (float)(((double)c_base.timestamp_us - t_min) / t_range) * size_.x +
        candle_width * 0.5f;

    // Price mapping
    float y_open =
        position_.y + size_.y - ((open - min_y_) / (max_y_ - min_y_)) * size_.y;
    float y_high =
        position_.y + size_.y - ((high - min_y_) / (max_y_ - min_y_)) * size_.y;
    float y_low =
        position_.y + size_.y - ((low - min_y_) / (max_y_ - min_y_)) * size_.y;
    float y_close = position_.y + size_.y -
                    ((close - min_y_) / (max_y_ - min_y_)) * size_.y;

    bool is_bullish = close >= open;
    glm::vec4 color = is_bullish ? theme_.price_up : theme_.price_down;
    glm::vec4 border_color =
        is_bullish ? theme_.price_up_bright : theme_.price_down_bright;

    float body_top = std::min(y_open, y_close);
    float body_bottom = std::max(y_open, y_close);
    if (body_bottom - body_top < 1.0f)
      body_bottom = body_top + 1.0f;

    // Body (Type 0) - Enhanced with border and professional fill
    glm::vec4 body_fill_color = color;
    if (is_bullish) {
      body_fill_color.a = 0.4f; // More opaque for bullish
    } else {
      body_fill_color.a = 0.8f; // More solid for bearish
    }

    // 1. Candlestick Body Quad
    vertices.push_back(
        {{x - bar_width * 0.5f, body_top}, {0, 0}, body_fill_color, 1.0f, 0});
    vertices.push_back(
        {{x + bar_width * 0.5f, body_top}, {1, 0}, body_fill_color, 1.0f, 0});
    vertices.push_back({{x + bar_width * 0.5f, body_bottom},
                        {1, 1},
                        body_fill_color,
                        1.0f,
                        0});
    vertices.push_back(
        {{x - bar_width * 0.5f, body_top}, {0, 0}, body_fill_color, 1.0f, 0});
    vertices.push_back({{x + bar_width * 0.5f, body_bottom},
                        {1, 1},
                        body_fill_color,
                        1.0f,
                        0});
    vertices.push_back({{x - bar_width * 0.5f, body_bottom},
                        {0, 1},
                        body_fill_color,
                        1.0f,
                        0});

    // 2. Candlestick Bottom Border (optional for crispness)
    vertices.push_back(
        {{x - bar_width * 0.5f, body_bottom}, {0, 1}, border_color, 0.0f, 1});
    vertices.push_back(
        {{x + bar_width * 0.5f, body_bottom}, {1, 1}, border_color, 0.0f, 1});
    vertices.push_back({{x + bar_width * 0.5f, body_bottom + 1.0f},
                        {1, 1},
                        border_color,
                        0.0f,
                        1});

    // 3. Wicks (Type 1) - One pixel wide wicks with bright color
    vertices.push_back({{x - 0.5f, y_high}, {0.5f, 0}, border_color, 0.0f, 1});
    vertices.push_back({{x + 0.5f, y_high}, {0.5f, 0}, border_color, 0.0f, 1});
    vertices.push_back({{x + 0.5f, y_low}, {0.5f, 1}, border_color, 0.0f, 1});
    vertices.push_back({{x - 0.5f, y_high}, {0.5f, 0}, border_color, 0.0f, 1});
    vertices.push_back({{x + 0.5f, y_low}, {0.5f, 1}, border_color, 0.0f, 1});
    vertices.push_back({{x - 0.5f, y_low}, {0.5f, 1}, border_color, 0.0f, 1});

    // 4. Volume (Type 2) - Professional separate panel look at bottom
    // Volume panel takes bottom 15% height
    float volume_panel_h = size_.y * 0.15f;
    float vol_h = (volume / max_volume) *
                  (volume_panel_h * 0.8f); // 80% usage for head-room
    glm::vec4 vol_color = color;
    vol_color.a = 0.5f;

    float vol_base = position_.y + size_.y - 5.0f; // 5px padding from bottom
    vertices.push_back(
        {{x - bar_width * 0.5f, vol_base - vol_h}, {0, 0}, vol_color, 0.0f, 2});
    vertices.push_back(
        {{x + bar_width * 0.5f, vol_base - vol_h}, {1, 0}, vol_color, 0.0f, 2});
    vertices.push_back(
        {{x + bar_width * 0.5f, vol_base}, {1, 1}, vol_color * 0.5f, 0.0f, 2});
    vertices.push_back(
        {{x - bar_width * 0.5f, vol_base - vol_h}, {0, 0}, vol_color, 0.0f, 2});
    vertices.push_back(
        {{x + bar_width * 0.5f, vol_base}, {1, 1}, vol_color * 0.5f, 0.0f, 2});
    vertices.push_back(
        {{x - bar_width * 0.5f, vol_base}, {0, 1}, vol_color * 0.5f, 0.0f, 2});
  }

  size_t buffer_size = vertices.size() * sizeof(CandlestickVertex);

  // Allocate for both frames
  for (int i = 0; i < 2; ++i) {
    if (!candlestick_vertex_buffers_[i].buffer ||
        candlestick_vertex_buffers_[i].size < buffer_size) {
      if (candlestick_vertex_buffers_[i].buffer) {
        vulkan_core_->get_memory_manager().deallocate_buffer(
            candlestick_vertex_buffers_[i]);
      }
      candlestick_vertex_buffers_[i] =
          vulkan_core_->get_memory_manager().allocate_vertex_buffer(
              std::max(buffer_size, (size_t)1024));
    }
    if (candlestick_vertex_buffers_[i].mapped_ptr) {
      memcpy(candlestick_vertex_buffers_[i].mapped_ptr, vertices.data(),
             buffer_size);
    }
  }
  candlestick_vertex_count_ = static_cast<uint32_t>(vertices.size());
}

void RealtimeChartComponent::rebuild_crosshair_geometry() {
  if (!dashboard_)
    return;
  const auto &cs = dashboard_->get_crosshair_state();
  if (!cs.active)
    return;

  std::vector<CandlestickVertex> vertices;
  glm::vec4 color = {0.8f, 0.8f, 0.8f, 0.6f};

  // Calculate X based on timestamp if this component is time-aligned
  if (!candles_.empty()) {
    double latest_t = (double)candles_.back().timestamp_us;
    double base_range = (double)candles_.size() * 5000000.0;
    double t_max = latest_t + 5000000.0 + (double)view_offset_ * 1000000.0;
    double t_min = t_max - base_range * (double)view_zoom_;
    double t_range = t_max - t_min;

    if (cs.timestamp_us >= t_min && cs.timestamp_us <= t_max) {
      float rel_x = (float)((cs.timestamp_us - t_min) / (t_max - t_min));
      float x = position_.x + rel_x * size_.x;

      // Vertical line
      vertices.push_back({{x - 0.5f, position_.y}, {0.5f, 0}, color, 1.0f, 4});
      vertices.push_back({{x + 0.5f, position_.y}, {0.5f, 0}, color, 1.0f, 4});
      vertices.push_back(
          {{x + 0.5f, position_.y + size_.y}, {0.5f, 1}, color, 1.0f, 4});
      vertices.push_back({{x - 0.5f, position_.y}, {0.5f, 0}, color, 1.0f, 4});
      vertices.push_back(
          {{x + 0.5f, position_.y + size_.y}, {0.5f, 1}, color, 1.0f, 4});
      vertices.push_back(
          {{x - 0.5f, position_.y + size_.y}, {0.5f, 1}, color, 1.0f, 4});
    }
  }

  // Horizontal line based on price if within range
  if (cs.price >= (double)min_y_ && cs.price <= (double)max_y_) {
    float rel_y = (float)((cs.price - min_y_) / (max_y_ - min_y_));
    float y = position_.y + size_.y - rel_y * size_.y;

    vertices.push_back({{position_.x, y - 0.5f}, {0, 0.5f}, color, 1.0f, 4});
    vertices.push_back(
        {{position_.x + size_.x, y - 0.5f}, {1, 0.5f}, color, 1.0f, 4});
    vertices.push_back(
        {{position_.x + size_.x, y + 0.5f}, {1, 0.5f}, color, 1.0f, 4});
    vertices.push_back({{position_.x, y - 0.5f}, {0, 0.5f}, color, 1.0f, 4});
    vertices.push_back(
        {{position_.x + size_.x, y + 0.5f}, {1, 0.5f}, color, 1.0f, 4});
    vertices.push_back({{position_.x, y + 0.5f}, {0, 0.5f}, color, 1.0f, 4});
  }

  if (vertices.empty())
    return;

  size_t buffer_size = vertices.size() * sizeof(CandlestickVertex);
  if (!crosshair_vertex_buffers_[0].buffer ||
      crosshair_vertex_buffers_[0].size < buffer_size) {
    for (int i = 0; i < 2; ++i) {
      if (crosshair_vertex_buffers_[i].buffer) {
        vulkan_core_->get_memory_manager().deallocate_buffer(
            crosshair_vertex_buffers_[i]);
      }
      crosshair_vertex_buffers_[i] =
          vulkan_core_->get_memory_manager().allocate_vertex_buffer(
              std::max(buffer_size, (size_t)512));
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (crosshair_vertex_buffers_[i].mapped_ptr) {
      memcpy(crosshair_vertex_buffers_[i].mapped_ptr, vertices.data(),
             buffer_size);
    }
  }
  crosshair_vertex_count_ = static_cast<uint32_t>(vertices.size());
}

void RealtimeChartComponent::rebuild_indicator_geometry() {
  if (candles_.size() < 2)
    return;

  std::vector<CandlestickVertex> vertices;

  double latest_t = (double)candles_.back().timestamp_us;
  double base_range = (double)candles_.size() * 5000000.0;
  double t_max = latest_t + 5000000.0 + (double)view_offset_ * 1000000.0;
  double t_min = t_max - base_range * (double)view_zoom_;
  float t_range = t_max - t_min;
  float candle_width = (size_.x / (float)candles_.size()) / (float)view_zoom_;

  // Split vertical space: 60% main chart, 12.5% RSI, 12.5% MACD, 15%
  // Volume/Bottom
  float main_height = size_.y * 0.60f;
  float rsi_height = size_.y * 0.125f;
  float macd_height = size_.y * 0.125f;

  float rsi_y_base = position_.y + main_height;
  float macd_y_base = position_.y + main_height + rsi_height;

  auto draw_line_indicator = [&](TechnicalIndicator &ind, float y_base, float h,
                                 float min_v, float max_v, glm::vec4 color) {
    ind.reset();
    float prev_val = 0;
    bool prev_ready = false;

    for (size_t j = 0; j < candles_.size(); ++j) {
      ind.update(candles_[j].close);
      if (!ind.is_ready())
        continue;

      float current_val = ind.get_value();
      if (prev_ready) {
        // LOD 1: Time-based Culling
        if (!((double)candles_[j].timestamp_us + 5000000.0 < t_min ||
              (double)candles_[j - 1].timestamp_us > t_max)) {

          float x1 = position_.x +
                     (float)(((double)candles_[j - 1].timestamp_us - t_min) /
                             t_range) *
                         size_.x +
                     candle_width * 0.5f;
          float x2 =
              position_.x +
              (float)(((double)candles_[j].timestamp_us - t_min) / t_range) *
                  size_.x +
              candle_width * 0.5f;

          float y1 = y_base + h - ((prev_val - min_v) / (max_v - min_v)) * h;
          float y2 = y_base + h - ((current_val - min_v) / (max_v - min_v)) * h;

          vertices.push_back({{x1, y1 - 1.0f}, {0, 0}, color, 1.0f, 5});
          vertices.push_back({{x2, y2 - 1.0f}, {0, 0}, color, 1.0f, 5});
          vertices.push_back({{x2, y2 + 1.0f}, {0, 0}, color, 1.0f, 5});
          vertices.push_back({{x1, y1 - 1.0f}, {0, 0}, color, 1.0f, 5});
          vertices.push_back({{x2, y2 + 1.0f}, {0, 0}, color, 1.0f, 5});
          vertices.push_back({{x1, y1 + 1.0f}, {0, 0}, color, 1.0f, 5});
        }
      }
      prev_val = current_val;
      prev_ready = true;
    }
  };

  // 1. Overlay Indicators (on main chart)
  EMAIndicator ema9(9);
  draw_line_indicator(ema9, position_.y, main_height, min_y_, max_y_,
                      {0.0f, 0.66f, 1.0f, 0.8f}); // EMA 9 (Blue)
  EMAIndicator ema21(21);
  draw_line_indicator(ema21, position_.y, main_height, min_y_, max_y_,
                      {1.0f, 0.66f, 0.0f, 0.8f}); // EMA 21 (Orange)
  SMAIndicator sma50(50);
  draw_line_indicator(sma50, position_.y, main_height, min_y_, max_y_,
                      {1.0f, 0.0f, 1.0f, 0.8f}); // SMA 50 (Purple)

  // 2. Sub-chart Indicators
  RSIIndicator rsi14(14);
  draw_line_indicator(rsi14, rsi_y_base, rsi_height, 0.0f, 100.0f,
                      {0.0f, 1.0f, 1.0f, 0.9f}); // RSI (Cyan)

  MACDIndicator macd(12, 26, 9);
  // MACD is trickier because it has 3 lines. I'll just draw the MACD line for
  // now.
  draw_line_indicator(macd, macd_y_base, macd_height, -0.01f * max_y_,
                      0.01f * max_y_,
                      {1.0f, 1.0f, 0.0f, 0.9f}); // MACD (Yellow)

  if (vertices.empty())
    return;

  size_t buffer_size = vertices.size() * sizeof(CandlestickVertex);
  if (!indicator_vertex_buffers_[0].buffer ||
      indicator_vertex_buffers_[0].size < buffer_size) {
    for (int i = 0; i < 2; ++i) {
      if (indicator_vertex_buffers_[i].buffer) {
        vulkan_core_->get_memory_manager().deallocate_buffer(
            indicator_vertex_buffers_[i]);
      }
      indicator_vertex_buffers_[i] =
          vulkan_core_->get_memory_manager().allocate_vertex_buffer(
              std::max(buffer_size, (size_t)1024));
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (indicator_vertex_buffers_[i].mapped_ptr) {
      memcpy(indicator_vertex_buffers_[i].mapped_ptr, vertices.data(),
             buffer_size);
    }
  }
  indicator_vertex_count_ = static_cast<uint32_t>(vertices.size());
}

void RealtimeChartComponent::update_y_range() {
  float data_min = 1e10f;
  float data_max = -1e10f;

  if (candlestick_mode_ && !candles_.empty()) {
    for (const auto &c : candles_) {
      data_min = std::min(data_min, c.low);
      data_max = std::max(data_max, c.high);
    }
  } else if (!data_points_.empty()) {
    for (const auto &dp : data_points_) {
      data_min = std::min(data_min, dp.value);
      data_max = std::max(data_max, dp.value);
    }
  } else {
    return;
  }

  // Add some padding
  float range = data_max - data_min;
  float padding = range * 0.15f;

  if (range < 0.001f) {
    padding = std::abs(data_min) * 0.1f;
    if (padding < 0.001f)
      padding = 1.0f;
  }

  min_y_ = data_min - padding;
  max_y_ = data_max + padding;
}

void RealtimeChartComponent::initialize_vulkan_resources(
    VulkanCore *vulkan_core) {
  vulkan_core_ = vulkan_core;

  // 1. Create Descriptor Set Layouts (both have 1 UBO)
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &binding;

  VulkanErrorHandler::check_result(
      vkCreateDescriptorSetLayout(vulkan_core_->get_device(), &layout_info,
                                  nullptr, &line_layout_),
      "vkCreateDescriptorSetLayout (Line)");
  VulkanErrorHandler::check_result(
      vkCreateDescriptorSetLayout(vulkan_core_->get_device(), &layout_info,
                                  nullptr, &ui_layout_),
      "vkCreateDescriptorSetLayout (UI)");

  // 2. Create Pipeline Layouts
  VkPushConstantRange chart_push{};
  chart_push.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  chart_push.offset = 0;
  chart_push.size = 64; // Match Chart shader

  VkPipelineLayoutCreateInfo line_pl_info{};
  line_pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  line_pl_info.setLayoutCount = 1;
  line_pl_info.pSetLayouts = &line_layout_;
  line_pl_info.pushConstantRangeCount = 1;
  line_pl_info.pPushConstantRanges = &chart_push;

  VulkanErrorHandler::check_result(
      vkCreatePipelineLayout(vulkan_core_->get_device(), &line_pl_info, nullptr,
                             &line_pipeline_layout_),
      "vkCreatePipelineLayout (Line)");

  VkPushConstantRange ui_push{};
  ui_push.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  ui_push.offset = 0;
  ui_push.size = 64; // Match UI shader

  VkPipelineLayoutCreateInfo ui_pl_info{};
  ui_pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ui_pl_info.setLayoutCount = 1;
  ui_pl_info.pSetLayouts = &ui_layout_;
  ui_pl_info.pushConstantRangeCount = 1;
  ui_pl_info.pPushConstantRanges = &ui_push;

  VulkanErrorHandler::check_result(
      vkCreatePipelineLayout(vulkan_core_->get_device(), &ui_pl_info, nullptr,
                             &ui_pipeline_layout_),
      "vkCreatePipelineLayout (UI)");

  // 3. Create Line Pipeline
  VkVertexInputBindingDescription line_binding{};
  line_binding.binding = 0;
  line_binding.stride = sizeof(LineVertex);
  line_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::vector<VkVertexInputAttributeDescription> line_attrs(5);
  line_attrs[0].location = 0;
  line_attrs[0].binding = 0;
  line_attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
  line_attrs[0].offset = offsetof(LineVertex, position);
  line_attrs[1].location = 1;
  line_attrs[1].binding = 0;
  line_attrs[1].format = VK_FORMAT_R32G32_SFLOAT;
  line_attrs[1].offset = offsetof(LineVertex, direction);
  line_attrs[2].location = 2;
  line_attrs[2].binding = 0;
  line_attrs[2].format = VK_FORMAT_R32_SFLOAT;
  line_attrs[2].offset = offsetof(LineVertex, thickness);
  line_attrs[3].location = 3;
  line_attrs[3].binding = 0;
  line_attrs[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  line_attrs[3].offset = offsetof(LineVertex, color);
  line_attrs[4].location = 4;
  line_attrs[4].binding = 0;
  line_attrs[4].format = VK_FORMAT_R32_SFLOAT;
  line_attrs[4].offset = offsetof(LineVertex, distance);

  std::vector<VkVertexInputBindingDescription> line_bindings_vec = {
      line_binding};
  line_pipeline_ = vulkan_core_->create_graphics_pipeline(
      "shaders/chart_lines.vert.spv", "shaders/chart_lines.frag.spv",
      line_bindings_vec, line_attrs, line_pipeline_layout_);

  // 4. Create Candlestick Pipeline
  VkVertexInputBindingDescription candle_binding{};
  candle_binding.binding = 0;
  candle_binding.stride = sizeof(CandlestickVertex);
  candle_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::vector<VkVertexInputAttributeDescription> candle_attrs(5);
  candle_attrs[0].location = 0;
  candle_attrs[0].binding = 0;
  candle_attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
  candle_attrs[0].offset = offsetof(CandlestickVertex, position);
  candle_attrs[1].location = 1;
  candle_attrs[1].binding = 0;
  candle_attrs[1].format = VK_FORMAT_R32G32_SFLOAT;
  candle_attrs[1].offset = offsetof(CandlestickVertex, size);
  candle_attrs[2].location = 2;
  candle_attrs[2].binding = 0;
  candle_attrs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  candle_attrs[2].offset = offsetof(CandlestickVertex, color);
  candle_attrs[3].location = 3;
  candle_attrs[3].binding = 0;
  candle_attrs[3].format = VK_FORMAT_R32_SFLOAT;
  candle_attrs[3].offset = offsetof(CandlestickVertex, border_width);
  candle_attrs[4].location = 4;
  candle_attrs[4].binding = 0;
  candle_attrs[4].format = VK_FORMAT_R32_UINT;
  candle_attrs[4].offset = offsetof(CandlestickVertex, candle_type);

  std::vector<VkVertexInputBindingDescription> candle_bindings_vec = {
      candle_binding};
  candlestick_pipeline_ = vulkan_core_->create_graphics_pipeline(
      "shaders/ui_vertex.vert.spv", "shaders/ui_fragment.frag.spv",
      candle_bindings_vec, candle_attrs, ui_pipeline_layout_);

  // 6. Allocate and Update Descriptor Sets
  line_ubo_buffer_ = vulkan_core_->get_memory_manager().allocate_uniform_buffer(
      sizeof(ChartUniformBuffer));
  ui_ubo_buffer_ = vulkan_core_->get_memory_manager().allocate_uniform_buffer(
      sizeof(UIUniformBuffer));

  // Line Set
  VkDescriptorSetAllocateInfo line_alloc{};
  line_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  line_alloc.descriptorPool = vulkan_core_->get_descriptor_pool();
  line_alloc.descriptorSetCount = 1;
  line_alloc.pSetLayouts = &line_layout_;
  vkAllocateDescriptorSets(vulkan_core_->get_device(), &line_alloc,
                           &line_descriptor_set_);

  // UI Set
  VkDescriptorSetAllocateInfo ui_alloc{};
  ui_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ui_alloc.descriptorPool = vulkan_core_->get_descriptor_pool();
  ui_alloc.descriptorSetCount = 1;
  ui_alloc.pSetLayouts = &ui_layout_;
  vkAllocateDescriptorSets(vulkan_core_->get_device(), &ui_alloc,
                           &ui_descriptor_set_);

  // Update Line Set
  VkDescriptorBufferInfo line_ubo_info{};
  line_ubo_info.buffer = line_ubo_buffer_.buffer;
  line_ubo_info.offset = line_ubo_buffer_.offset;
  line_ubo_info.range = sizeof(ChartUniformBuffer);

  VkWriteDescriptorSet line_write{};
  line_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  line_write.dstSet = line_descriptor_set_;
  line_write.dstBinding = 0;
  line_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  line_write.descriptorCount = 1;
  line_write.pBufferInfo = &line_ubo_info;
  vkUpdateDescriptorSets(vulkan_core_->get_device(), 1, &line_write, 0,
                         nullptr);

  // Update UI Set
  VkDescriptorBufferInfo ui_ubo_info{};
  ui_ubo_info.buffer = ui_ubo_buffer_.buffer;
  ui_ubo_info.offset = ui_ubo_buffer_.offset;
  ui_ubo_info.range = sizeof(UIUniformBuffer);

  VkWriteDescriptorSet ui_write{};
  ui_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  ui_write.dstSet = ui_descriptor_set_;
  ui_write.dstBinding = 0;
  ui_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  ui_write.descriptorCount = 1;
  ui_write.pBufferInfo = &ui_ubo_info;
  vkUpdateDescriptorSets(vulkan_core_->get_device(), 1, &ui_write, 0, nullptr);

  fprintf(
      stderr,
      "[RealtimeChartComponent] Vulkan resources initialized successfully\n");
  mark_dirty();
}
void RealtimeChartComponent::update_indicators() {
  if (candles_.empty())
    return;
  for (auto &indicator : indicators_) {
    indicator->reset();
    for (size_t i = 0; i < candles_.size(); ++i) {
      indicator->update(candles_[i].close);
    }
  }
}

void RealtimeChartComponent::clear_data() {
  data_points_.clear();
  candles_.clear();
  mark_dirty();
}

} // namespace BTQuant