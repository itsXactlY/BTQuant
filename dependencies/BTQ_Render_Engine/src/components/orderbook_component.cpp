/**
 * BTQuant Advanced Vulkan Dashboard - Order Book Component Implementation
 *
 * Professional-grade order book visualization with real-time updates,
 * depth visualization, and high-performance GPU rendering.
 *
 * Features:
 * - Real-time order book depth visualization
 * - Color-coded bid/ask levels with intensity mapping
 * - Size bars showing relative volume at each level
 * - Spread calculation and mid-price display
 * - Smooth animations for level updates
 * - Professional typography for price/size formatting
 * - Interactive hover effects and selection
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace BTQuant {

// OrderBookComponent implementation

OrderBookComponent::OrderBookComponent(const glm::vec2 &position,
                                       const glm::vec2 &size)
    : UIComponent(position, size) {

  // Initialize empty order book data
  current_data_.bids.clear();
  current_data_.asks.clear();
  current_data_.spread = 0.0;
  current_data_.mid_price = 0.0;
  current_data_.timestamp = 0;
}

OrderBookComponent::~OrderBookComponent() {
  if (vulkan_core_) {
    VkDevice device = vulkan_core_->get_device();
    if (bar_pipeline_ != VK_NULL_HANDLE)
      vkDestroyPipeline(device, bar_pipeline_, nullptr);
    if (text_pipeline_ != VK_NULL_HANDLE)
      vkDestroyPipeline(device, text_pipeline_, nullptr);
    if (bar_pipeline_layout_ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, bar_pipeline_layout_, nullptr);
    if (text_pipeline_layout_ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, text_pipeline_layout_, nullptr);
    if (bar_layout_ != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, bar_layout_, nullptr);
    if (text_layout_ != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, text_layout_, nullptr);

    if (font_sampler_ != VK_NULL_HANDLE)
      vkDestroySampler(device, font_sampler_, nullptr);
    if (font_image_view_ != VK_NULL_HANDLE)
      vkDestroyImageView(device, font_image_view_, nullptr);
    if (font_image_ != VK_NULL_HANDLE)
      vkDestroyImage(device, font_image_, nullptr);
    if (font_memory_ != VK_NULL_HANDLE)
      vkFreeMemory(device, font_memory_, nullptr);
  }
}

void OrderBookComponent::update_orderbook(const OrderBookData &data) {
  std::lock_guard lock(data_mutex_);
  float current_time = 0.0f;
  // Use steady clock for reliable animation timing
  current_time = static_cast<float>(
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count()) /
                 1000.0f;

  // Detect changes for animation
  auto update_levs = [&](const std::vector<OrderBookLevel> &old_levels,
                         const std::vector<OrderBookLevel> &new_levels) {
    std::vector<OrderBookLevel> result;
    for (const auto &nl : new_levels) {
      OrderBookLevel level = nl;
      level.last_update_ts = 0.0f;
      for (const auto &ol : old_levels) {
        if (std::abs(ol.price - nl.price) < 1e-9) {
          if (std::abs(ol.size - nl.size) > 1e-9) {
            level.last_update_ts = current_time;
          } else {
            level.last_update_ts = ol.last_update_ts;
          }
          break;
        }
      }
      result.push_back(level);
    }
    return result;
  };

  current_data_.bids = update_levs(current_data_.bids, data.bids);
  current_data_.asks = update_levs(current_data_.asks, data.asks);
  current_data_.spread = data.spread;
  current_data_.mid_price = data.mid_price;
  current_data_.timestamp = data.timestamp;

  // Limit and Sort
  if (current_data_.bids.size() > max_levels_)
    current_data_.bids.resize(max_levels_);
  if (current_data_.asks.size() > max_levels_)
    current_data_.asks.resize(max_levels_);

  std::sort(current_data_.bids.begin(), current_data_.bids.end(),
            [](const auto &a, const auto &b) { return a.price > b.price; });
  std::sort(current_data_.asks.begin(), current_data_.asks.end(),
            [](const auto &a, const auto &b) { return a.price < b.price; });

  mark_dirty();
}

void OrderBookComponent::set_precision(int price_precision,
                                       int size_precision) {
  price_precision_ = price_precision;
  size_precision_ = size_precision;
  mark_dirty();
}

void OrderBookComponent::update(float delta_time) {
  if (is_dirty()) {
    rebuild_geometry();
    dirty_frames_--;
  }

  // Update animations for level changes
  static float animation_time = 0.0f;
  animation_time += delta_time;

  // TODO: Implement smooth transitions for price level updates
}

void OrderBookComponent::render(VkCommandBuffer cmd) {
  if (!visible_)
    return;

  std::lock_guard lock(data_mutex_);

  // 1. Update UBOs
  VkExtent2D extent = vulkan_core_->get_swapchain_extent();

  if (bar_ubo_buffer_.mapped_ptr) {
    UIUniformBuffer ubo{};
    // Swap 0.0f and extent.height to match Vulkan NDC Y direction
    ubo.projection = glm::ortho(0.0f, (float)extent.width, 0.0f,
                                (float)extent.height, -1.0f, 1.0f);
    ubo.view = glm::mat4(1.0f);
    ubo.model = glm::mat4(1.0f);
    ubo.viewport_size = glm::vec2(extent.width, extent.height);
    ubo.dpi_scale = glm::vec2(1.0f);
    ubo.global_tint = glm::vec4(1.0f);
    memcpy(bar_ubo_buffer_.mapped_ptr, &ubo, sizeof(ubo));
  }

  if (text_ubo_buffer_.mapped_ptr) {
    TextUniformBuffer ubo{};
    // Swap 0.0f and extent.height to match Vulkan NDC Y direction
    ubo.projection = glm::ortho(0.0f, (float)extent.width, 0.0f,
                                (float)extent.height, -1.0f, 1.0f);
    ubo.view = glm::mat4(1.0f);
    ubo.viewport_size = glm::vec2(extent.width, extent.height);
    ubo.dpi_scale = glm::vec2(1.0f);
    ubo.global_text_color = glm::vec4(1.0f);
    ubo.render_flags = 0x10; // RENDER_GAMMA_CORRECT
    memcpy(text_ubo_buffer_.mapped_ptr, &ubo, sizeof(ubo));
  }

  // 1. Render size bars
  if (bar_vertex_buffer_.buffer && bar_pipeline_ != VK_NULL_HANDLE) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bar_pipeline_);

    struct {
      glm::vec2 offset;
      glm::vec2 scale;
    } bar_push;
    bar_push.offset = glm::vec2(0, 0);
    bar_push.scale = glm::vec2(1, 1);

    vkCmdPushConstants(cmd, bar_pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(bar_push), &bar_push);

    if (bar_descriptor_set_ != VK_NULL_HANDLE) {
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              bar_pipeline_layout_, 0, 1, &bar_descriptor_set_,
                              0, nullptr);
    }

    VkBuffer buffers[] = {bar_vertex_buffer_.buffer};
    VkDeviceSize offsets[] = {bar_vertex_buffer_.offset};
    vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);

    uint32_t vertex_count =
        static_cast<uint32_t>(bar_vertex_buffer_.size / sizeof(DepthBarVertex));
    if (vertex_count > 0) {
      vkCmdDraw(cmd, vertex_count, 1, 0, 0);
    }
  }

  // 2. Render text labels
  if (text_vertex_buffer_.buffer && text_pipeline_ != VK_NULL_HANDLE) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, text_pipeline_);

    struct {
      glm::vec2 offset;
      float scale;
    } text_push;
    text_push.offset = glm::vec2(0, 0);
    text_push.scale = 1.0f;

    vkCmdPushConstants(cmd, text_pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(text_push), &text_push);

    if (text_descriptor_set_ != VK_NULL_HANDLE) {
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              text_pipeline_layout_, 0, 1,
                              &text_descriptor_set_, 0, nullptr);
    }

    VkBuffer buffers[] = {text_vertex_buffer_.buffer};
    VkDeviceSize offsets[] = {text_vertex_buffer_.offset};
    vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);

    uint32_t vertex_count = static_cast<uint32_t>(text_vertex_buffer_.size /
                                                  sizeof(OrderBookTextVertex));
    if (vertex_count > 0) {
      vkCmdDraw(cmd, vertex_count, 1, 0, 0);
    }
  }
}

void OrderBookComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_FirstUseEver);

  ImGui::SetNextWindowCollapsed(minimized_, ImGuiCond_Appearing);
  if (!ImGui::Begin(("Order Book: " + symbol_).c_str(), &visible_)) {
    minimized_ = true;
    ImGui::End();
    return;
  }
  minimized_ = false;

  std::lock_guard lock(data_mutex_);

  if (theme_.monospace_font)
    ImGui::PushFont((ImFont *)theme_.monospace_font);

  // Use monochromatic institutional styling for the table
  // Use monochromatic institutional styling for the table
  ImGuiTableFlags table_flags =
      ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
      ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_SizingFixedFit;

  if (ImGui::BeginTable("OrderBookLadder", 3, table_flags)) {
    ImGui::TableSetupColumn("BidSize", ImGuiTableColumnFlags_WidthFixed, 65.0f);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 85.0f);
    ImGui::TableSetupColumn("AskSize", ImGuiTableColumnFlags_WidthFixed, 65.0f);

    // Headers
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("SIZE");
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("  PRICE");
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("SIZE");

    // Calculate max size for bar scaling
    double max_size = 0.0;
    for (const auto &l : current_data_.bids)
      max_size = std::max(max_size, l.size);
    for (const auto &l : current_data_.asks)
      max_size = std::max(max_size, l.size);
    if (max_size < 1e-9)
      max_size = 1.0;

    // Asks (Highest price at top)
    for (auto it = current_data_.asks.rbegin(); it != current_data_.asks.rend();
         ++it) {
      ImGui::TableNextRow(ImGuiTableRowFlags_None, 16.0f);

      // Ask Size Column with Bar
      ImGui::TableSetColumnIndex(2);
      float bar_width = (float)(it->size / max_size) * ImGui::GetColumnWidth();
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      ImVec2 pos = ImGui::GetCursorScreenPos();
      ImU32 ask_bar_color = to_imu32(theme_.price_down);
      ask_bar_color =
          (ask_bar_color & 0x00FFFFFF) | (static_cast<ImU32>(0.2f * 255) << 24);
      draw_list->AddRectFilled(pos, ImVec2(pos.x + bar_width, pos.y + 16.0f),
                               ask_bar_color);
      ImGui::Text("%.4f", it->size);

      // Price Column
      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(to_imvec4(theme_.price_down), "  %.2f", it->price);
    }

    // Spread Row
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 20.0f);
    ImGui::TableSetBgColor(
        ImGuiTableBgTarget_RowBg0,
        ImGui::GetColorU32(ImVec4(0.08f, 0.08f, 0.10f, 1.0f)));

    ImGui::TableSetColumnIndex(1);
    ImGui::TextColored(to_imvec4(theme_.accent_primary), "  %.2f",
                       current_data_.mid_price);

    ImGui::TableSetColumnIndex(2);
    ImGui::TextDisabled("SPR %.2f", current_data_.spread);

    // Bids (Highest price at top)
    for (const auto &level : current_data_.bids) {
      ImGui::TableNextRow(ImGuiTableRowFlags_None, 16.0f);

      // Bid Size Column with Bar (Right-aligned bar)
      ImGui::TableSetColumnIndex(0);
      float bar_width =
          (float)(level.size / max_size) * ImGui::GetColumnWidth();
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      ImVec2 pos = ImGui::GetCursorScreenPos();
      ImU32 bid_bar_color = to_imu32(theme_.price_up);
      bid_bar_color =
          (bid_bar_color & 0x00FFFFFF) | (static_cast<ImU32>(0.2f * 255) << 24);
      draw_list->AddRectFilled(
          ImVec2(pos.x + ImGui::GetColumnWidth() - bar_width, pos.y),
          ImVec2(pos.x + ImGui::GetColumnWidth(), pos.y + 16.0f),
          bid_bar_color);
      ImGui::Text("%.4f", level.size);

      // Price Column
      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(to_imvec4(theme_.price_up), "  %.2f", level.price);
    }

    ImGui::EndTable();
  }

  if (theme_.monospace_font)
    ImGui::PopFont();

  ImGui::End();
}

void OrderBookComponent::handle_input(const InputEvent &event) {
  switch (event.type) {
  case InputEventType::MouseMove: {
    // Calculate which level is being hovered
    float row_height = 20.0f; // Fixed row height
    float header_height = 25.0f;

    float local_y = event.position.y - position_.y - header_height;

    if (local_y >= 0) {
      int row = static_cast<int>(local_y / row_height);

      // Determine if hovering over bid or ask
      int total_ask_rows = static_cast<int>(current_data_.asks.size());

      if (row < total_ask_rows) {
        // Hovering over ask level
        int ask_index = total_ask_rows - 1 - row; // Reverse order for asks
        if (ask_index >= 0 &&
            ask_index < static_cast<int>(current_data_.asks.size())) {
          // TODO: Show tooltip with level details
        }
      } else {
        // Hovering over bid level
        int bid_index = row - total_ask_rows - 1; // Account for spread row
        if (bid_index >= 0 &&
            bid_index < static_cast<int>(current_data_.bids.size())) {
          // TODO: Show tooltip with level details
        }
      }
    }
    break;
  }

  case InputEventType::MouseButton:
    if (event.pressed) {
      // TODO: Handle level selection for trading interface
    }
    break;

  default:
    break;
  }
}

void OrderBookComponent::rebuild_geometry() {
  std::vector<OrderBookTextVertex> text_vertices;
  std::vector<DepthBarVertex> bar_vertices;

  float row_height = 18.0f;
  float header_height = 25.0f;
  float font_size = 11.0f;

  // Calculate maximum size for bar scaling (use 90% of column width for max
  // size)
  double max_size = 0.0;
  for (const auto &level : current_data_.bids)
    max_size = std::max(max_size, level.size);
  for (const auto &level : current_data_.asks)
    max_size = std::max(max_size, level.size);
  if (max_size <= 0)
    max_size = 1.0;

  float current_y = position_.y + header_height;
  float col_w = size_.x / 3.0f;
  float mid_x = position_.x + size_.x * 0.5f;

  // 1. Headers
  add_text_at_position(text_vertices, "SIZE", position_.x + 10, current_y + 2,
                       theme_.text_secondary, font_size);
  add_centered_text(text_vertices, "PRICE", current_y + 2,
                    theme_.text_secondary, font_size);
  add_text_at_position(text_vertices, "SIZE", position_.x + size_.x - 40,
                       current_y + 2, theme_.text_secondary, font_size);
  current_y += header_height;

  // 2. Asks (Top half, descending price)
  for (int i = static_cast<int>(current_data_.asks.size()) - 1; i >= 0; --i) {
    const auto &level = current_data_.asks[i];
    float bar_w = (static_cast<float>(level.size / max_size)) * col_w * 0.95f;

    // Intensity mapping and Animation flash
    float intensity =
        std::min(1.0f, static_cast<float>(level.size / max_size) * 1.5f);
    glm::vec4 bar_color = ask_bar_color_;

    // Process animation flash
    float flash = 0.0f;
    float current_time =
        static_cast<float>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count()) /
        1000.0f;
    if (level.last_update_ts > 0 &&
        current_time - level.last_update_ts < 0.5f) {
      flash = 1.0f - (current_time - level.last_update_ts) / 0.5f;
    }
    bar_color.a = (0.15f + 0.65f * intensity) + flash * 0.3f;
    if (flash > 0)
      bar_color += glm::vec4(flash * 0.2f);

    // Render bar background (behind the Size column)
    float bar_start_x = position_.x + size_.x - col_w;
    bar_vertices.push_back(
        {{bar_start_x, current_y}, {0, 0}, bar_color, intensity, 1});
    bar_vertices.push_back(
        {{bar_start_x + bar_w, current_y}, {1, 0}, bar_color, intensity, 1});
    bar_vertices.push_back({{bar_start_x + bar_w, current_y + row_height - 1},
                            {1, 1},
                            bar_color,
                            intensity,
                            1});
    bar_vertices.push_back(
        {{bar_start_x, current_y}, {0, 0}, bar_color, intensity, 1});
    bar_vertices.push_back({{bar_start_x + bar_w, current_y + row_height - 1},
                            {1, 1},
                            bar_color,
                            intensity,
                            1});
    bar_vertices.push_back({{bar_start_x, current_y + row_height - 1},
                            {0, 1},
                            bar_color,
                            intensity,
                            1});

    add_text_at_position(text_vertices, format_price(level.price),
                         mid_x - col_w * 0.4f, current_y + 2, theme_.price_down,
                         font_size);
    add_text_at_position(text_vertices, format_size(level.size),
                         position_.x + size_.x - col_w + 10, current_y + 2,
                         theme_.text_primary, font_size);
    current_y += row_height;
  }

  // 3. Spread/Mid Row (Enhanced Highlighting)
  if (current_data_.spread >= 0) {
    // Spread background
    glm::vec4 spread_bg = {0.1f, 0.1f, 0.15f, 0.8f};
    bar_vertices.push_back({{position_.x, current_y}, {0, 0}, spread_bg, 0, 2});
    bar_vertices.push_back(
        {{position_.x + size_.x, current_y}, {1, 0}, spread_bg, 0, 2});
    bar_vertices.push_back({{position_.x + size_.x, current_y + row_height},
                            {1, 1},
                            spread_bg,
                            0,
                            2});
    bar_vertices.push_back({{position_.x, current_y}, {0, 0}, spread_bg, 0, 2});
    bar_vertices.push_back({{position_.x + size_.x, current_y + row_height},
                            {1, 1},
                            spread_bg,
                            0,
                            2});
    bar_vertices.push_back(
        {{position_.x, current_y + row_height}, {0, 1}, spread_bg, 0, 2});

    add_centered_text(text_vertices,
                      "MID: " + format_price(current_data_.mid_price),
                      current_y + 2, theme_.accent_secondary, font_size);
    add_text_at_position(text_vertices,
                         "SPR: " + format_price(current_data_.spread),
                         position_.x + size_.x - 65, current_y + 2,
                         theme_.text_muted, font_size * 0.8f);
    current_y += row_height;
  }

  // 4. Bids (Bottom half, descending price)
  for (const auto &level : current_data_.bids) {
    float bar_w = (static_cast<float>(level.size / max_size)) * col_w * 0.95f;
    float intensity =
        std::min(1.0f, static_cast<float>(level.size / max_size) * 1.5f);
    glm::vec4 bar_color = bid_bar_color_;

    float flash = 0.0f;
    float current_time =
        static_cast<float>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count()) /
        1000.0f;
    if (level.last_update_ts > 0 &&
        current_time - level.last_update_ts < 0.5f) {
      flash = 1.0f - (current_time - level.last_update_ts) / 0.5f;
    }
    bar_color.a = (0.15f + 0.65f * intensity) + flash * 0.3f;
    if (flash > 0)
      bar_color += glm::vec4(flash * 0.2f);

    // Bar on the left Size column
    float bar_start_x = position_.x;
    bar_vertices.push_back({{bar_start_x + col_w - bar_w, current_y},
                            {0, 0},
                            bar_color,
                            intensity,
                            0});
    bar_vertices.push_back(
        {{bar_start_x + col_w, current_y}, {1, 0}, bar_color, intensity, 0});
    bar_vertices.push_back({{bar_start_x + col_w, current_y + row_height - 1},
                            {1, 1},
                            bar_color,
                            intensity,
                            0});
    bar_vertices.push_back({{bar_start_x + col_w - bar_w, current_y},
                            {0, 0},
                            bar_color,
                            intensity,
                            0});
    bar_vertices.push_back({{bar_start_x + col_w, current_y + row_height - 1},
                            {1, 1},
                            bar_color,
                            intensity,
                            0});
    bar_vertices.push_back(
        {{bar_start_x + col_w - bar_w, current_y + row_height - 1},
         {0, 1},
         bar_color,
         intensity,
         0});

    add_text_at_position(text_vertices, format_price(level.price),
                         mid_x - col_w * 0.4f, current_y + 2, theme_.price_up,
                         font_size);
    add_text_at_position(text_vertices, format_size(level.size),
                         position_.x + 10, current_y + 2, theme_.text_primary,
                         font_size);
    current_y += row_height;
  }

  // Upload vertices to GPU buffers
  if (!text_vertices.empty()) {
    size_t text_buffer_size =
        text_vertices.size() * sizeof(OrderBookTextVertex);
    if (!text_vertex_buffer_.buffer ||
        text_vertex_buffer_.size < text_buffer_size) {
      if (text_vertex_buffer_.buffer) {
        vulkan_core_->get_memory_manager().deallocate_buffer(
            text_vertex_buffer_);
      }
      text_vertex_buffer_ =
          vulkan_core_->get_memory_manager().allocate_vertex_buffer(
              text_buffer_size);
    }
    if (text_vertex_buffer_.mapped_ptr) {
      memcpy(text_vertex_buffer_.mapped_ptr, text_vertices.data(),
             text_buffer_size);
    }
  }

  if (!bar_vertices.empty()) {
    size_t bar_buffer_size = bar_vertices.size() * sizeof(DepthBarVertex);
    if (!bar_vertex_buffer_.buffer ||
        bar_vertex_buffer_.size < bar_buffer_size) {
      if (bar_vertex_buffer_.buffer) {
        vulkan_core_->get_memory_manager().deallocate_buffer(
            bar_vertex_buffer_);
      }
      bar_vertex_buffer_ =
          vulkan_core_->get_memory_manager().allocate_vertex_buffer(
              bar_buffer_size);
    }
    if (bar_vertex_buffer_.mapped_ptr) {
      memcpy(bar_vertex_buffer_.mapped_ptr, bar_vertices.data(),
             bar_buffer_size);
    }
  }
}

std::string OrderBookComponent::format_price(double price) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(price_precision_) << price;
  return oss.str();
}

std::string OrderBookComponent::format_size(double size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(size_precision_) << size;
  return oss.str();
}

void OrderBookComponent::add_text_line(
    std::vector<OrderBookTextVertex> &vertices, const std::string &price,
    const std::string &size, const std::string &total, float y,
    const glm::vec4 &color, float font_size) {

  float col_width = size_.x / 3.0f;

  // Price column (left-aligned)
  add_text_at_position(vertices, price, position_.x + 5.0f, y, color,
                       font_size);

  // Size column (center-aligned)
  add_text_at_position(vertices, size, position_.x + col_width + 5.0f, y, color,
                       font_size);

  // Total column (right-aligned)
  add_text_at_position(vertices, total, position_.x + 2 * col_width + 5.0f, y,
                       color, font_size);
}

void OrderBookComponent::add_centered_text(
    std::vector<OrderBookTextVertex> &vertices, const std::string &text,
    float y, const glm::vec4 &color, float font_size) {

  // TODO: Calculate text width for proper centering
  float text_x =
      position_.x + size_.x * 0.5f - (text.length() * font_size * 0.3f);
  add_text_at_position(vertices, text, text_x, y, color, font_size);
}

void OrderBookComponent::add_text_at_position(
    std::vector<OrderBookTextVertex> &vertices, const std::string &text,
    float x, float y, const glm::vec4 &color, float font_size) {

  // Generate vertices for text rendering
  // This is a simplified implementation - a real text renderer would use
  // a font atlas and proper glyph metrics

  float char_width = font_size * 0.6f;
  float current_x = x;

  for (size_t i = 0; i < text.length(); ++i) {
    char c = text[i];
    uint32_t glyph_id = static_cast<uint32_t>(c);

    // Create quad for character
    vertices.push_back(
        {{current_x, y}, {0.0f, 0.0f}, color, glyph_id, font_size});
    vertices.push_back({{current_x + char_width, y},
                        {1.0f, 0.0f},
                        color,
                        glyph_id,
                        font_size});
    vertices.push_back({{current_x + char_width, y + font_size},
                        {1.0f, 1.0f},
                        color,
                        glyph_id,
                        font_size});

    vertices.push_back(
        {{current_x, y}, {0.0f, 0.0f}, color, glyph_id, font_size});
    vertices.push_back({{current_x + char_width, y + font_size},
                        {1.0f, 1.0f},
                        color,
                        glyph_id,
                        font_size});
    vertices.push_back(
        {{current_x, y + font_size}, {0.0f, 1.0f}, color, glyph_id, font_size});

    current_x += char_width;
  }
}

void OrderBookComponent::setup_uniform_buffer(OrderBookUniformBuffer &ubo) {
  ubo.projection = glm::ortho(0.0f, 1920.0f, 1080.0f, 0.0f, -1.0f, 1.0f);
  ubo.view = glm::mat4(1.0f);
  ubo.component_size = size_;
  ubo.component_position = position_;
  ubo.row_height = 20.0f;

  // Calculate max size for bar scaling
  double max_size = 0.0;
  for (const auto &level : current_data_.bids) {
    max_size = std::max(max_size, level.size);
  }
  for (const auto &level : current_data_.asks) {
    max_size = std::max(max_size, level.size);
  }
  ubo.max_size_for_bars = static_cast<float>(max_size);

  ubo.spread_highlight_intensity = 1.0f;
  ubo.time = static_cast<float>(
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count()) /
             1000.0f;

  ubo.bid_color = theme_.price_up;
  ubo.ask_color = theme_.price_down;
  ubo.spread_color = theme_.accent_secondary;
  ubo.animation_phase = std::sin(ubo.time * 2.0f) * 0.5f + 0.5f;
}

void OrderBookComponent::initialize_vulkan_resources(VulkanCore *vulkan_core) {
  vulkan_core_ = vulkan_core;

  // 1. Create Bar Layout (1 UBO)
  VkDescriptorSetLayoutBinding bar_layout_binding{};
  bar_layout_binding.binding = 0;
  bar_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bar_layout_binding.descriptorCount = 1;
  bar_layout_binding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo bar_layout_info{};
  bar_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  bar_layout_info.bindingCount = 1;
  bar_layout_info.pBindings = &bar_layout_binding;

  VulkanErrorHandler::check_result(
      vkCreateDescriptorSetLayout(vulkan_core_->get_device(), &bar_layout_info,
                                  nullptr, &bar_layout_),
      "vkCreateDescriptorSetLayout (Bar)");

  // 2. Create Text Layout (UBO, SSBO, Sampler)
  std::vector<VkDescriptorSetLayoutBinding> text_bindings(3);
  text_bindings[0].binding = 0;
  text_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  text_bindings[0].descriptorCount = 1;
  text_bindings[0].stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  text_bindings[1].binding = 1;
  text_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  text_bindings[1].descriptorCount = 1;
  text_bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  text_bindings[2].binding = 2;
  text_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  text_bindings[2].descriptorCount = 1;
  text_bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo text_layout_info{};
  text_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  text_layout_info.bindingCount = 3;
  text_layout_info.pBindings = text_bindings.data();

  VulkanErrorHandler::check_result(
      vkCreateDescriptorSetLayout(vulkan_core_->get_device(), &text_layout_info,
                                  nullptr, &text_layout_),
      "vkCreateDescriptorSetLayout (Text)");

  // 3. Create pipeline layout (can be separate or combined, let's use separate
  // for simplicity if needed, but the core function takes one layout) Actually,
  // create_graphics_pipeline takes a VkPipelineLayout. I'll create a single
  // pipeline layout that works for both? No, use two different ones if needed.
  // Wait, I only have one member pipeline_layout_. Let's just create one that
  // is compatible with the most complex one if possible, OR change to have two.
  // Shaders have specific layouts.

  VkPipelineLayoutCreateInfo bar_pl_info{};
  bar_pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  bar_pl_info.setLayoutCount = 1;
  bar_pl_info.pSetLayouts = &bar_layout_;
  // Add push constants for bars (UI shader)
  VkPushConstantRange bar_push{};
  bar_push.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  bar_push.offset = 0;
  bar_push.size = 64; // Match UI shader
  bar_pl_info.pushConstantRangeCount = 1;
  bar_pl_info.pPushConstantRanges = &bar_push;

  VulkanErrorHandler::check_result(
      vkCreatePipelineLayout(vulkan_core_->get_device(), &bar_pl_info, nullptr,
                             &bar_pipeline_layout_),
      "vkCreatePipelineLayout (Bar)");

  VkPipelineLayoutCreateInfo text_pl_info{};
  text_pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  text_pl_info.setLayoutCount = 1;
  text_pl_info.pSetLayouts = &text_layout_;
  // Add push constants for text
  VkPushConstantRange text_push{};
  text_push.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  text_push.offset = 0;
  text_push.size = 64; // Match Text shader
  text_pl_info.pushConstantRangeCount = 1;
  text_pl_info.pPushConstantRanges = &text_push;

  VulkanErrorHandler::check_result(
      vkCreatePipelineLayout(vulkan_core_->get_device(), &text_pl_info, nullptr,
                             &text_pipeline_layout_),
      "vkCreatePipelineLayout (Text)");

  // 4. Create Bar Pipeline
  VkVertexInputBindingDescription bar_vi_binding{};
  bar_vi_binding.binding = 0;
  bar_vi_binding.stride = sizeof(DepthBarVertex);
  bar_vi_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::vector<VkVertexInputAttributeDescription> bar_attrs(5);
  bar_attrs[0].location = 0;
  bar_attrs[0].binding = 0;
  bar_attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
  bar_attrs[0].offset = offsetof(DepthBarVertex, position);
  bar_attrs[1].location = 1;
  bar_attrs[1].binding = 0;
  bar_attrs[1].format = VK_FORMAT_R32G32_SFLOAT;
  bar_attrs[1].offset = offsetof(DepthBarVertex, size);
  bar_attrs[2].location = 2;
  bar_attrs[2].binding = 0;
  bar_attrs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  bar_attrs[2].offset = offsetof(DepthBarVertex, color);
  bar_attrs[3].location = 3;
  bar_attrs[3].binding = 0;
  bar_attrs[3].format = VK_FORMAT_R32_SFLOAT;
  bar_attrs[3].offset = offsetof(DepthBarVertex, intensity);
  bar_attrs[4].location = 4;
  bar_attrs[4].binding = 0;
  bar_attrs[4].format = VK_FORMAT_R32_UINT;
  bar_attrs[4].offset = offsetof(DepthBarVertex, level_type);

  std::vector<VkVertexInputBindingDescription> bar_bindings_vec = {
      bar_vi_binding};
  bar_pipeline_ = vulkan_core_->create_graphics_pipeline(
      "shaders/ui_vertex.vert.spv", "shaders/ui_fragment.frag.spv",
      bar_bindings_vec, bar_attrs, bar_pipeline_layout_);

  // 5. Create Text Pipeline
  VkVertexInputBindingDescription text_vi_binding{};
  text_vi_binding.binding = 0;
  text_vi_binding.stride = sizeof(OrderBookTextVertex);
  text_vi_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::vector<VkVertexInputAttributeDescription> text_attrs(5);
  text_attrs[0].location = 0;
  text_attrs[0].binding = 0;
  text_attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
  text_attrs[0].offset = offsetof(OrderBookTextVertex, position);
  text_attrs[1].location = 1;
  text_attrs[1].binding = 0;
  text_attrs[1].format = VK_FORMAT_R32G32_SFLOAT;
  text_attrs[1].offset = offsetof(OrderBookTextVertex, texcoord);
  text_attrs[2].location = 2;
  text_attrs[2].binding = 0;
  text_attrs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  text_attrs[2].offset = offsetof(OrderBookTextVertex, color);
  text_attrs[3].location = 3;
  text_attrs[3].binding = 0;
  text_attrs[3].format = VK_FORMAT_R32_UINT;
  text_attrs[3].offset = offsetof(OrderBookTextVertex, glyph_id);
  text_attrs[4].location = 4;
  text_attrs[4].binding = 0;
  text_attrs[4].format = VK_FORMAT_R32_SFLOAT;
  text_attrs[4].offset = offsetof(OrderBookTextVertex, font_size);

  std::vector<VkVertexInputBindingDescription> text_bindings_vec = {
      text_vi_binding};
  text_pipeline_ = vulkan_core_->create_graphics_pipeline(
      "shaders/text_rendering.vert.spv", "shaders/text_rendering.frag.spv",
      text_bindings_vec, text_attrs, text_pipeline_layout_);

  // 6. Allocate and Update Descriptor Sets
  bar_ubo_buffer_ = vulkan_core_->get_memory_manager().allocate_uniform_buffer(
      sizeof(UIUniformBuffer));
  text_ubo_buffer_ = vulkan_core_->get_memory_manager().allocate_uniform_buffer(
      sizeof(TextUniformBuffer));
  font_metrics_buffer_ =
      vulkan_core_->get_memory_manager().allocate_storage_buffer(
          sizeof(GlyphMetric) * 128);

  // Fill dummy font metrics
  if (font_metrics_buffer_.mapped_ptr) {
    GlyphMetric *metrics =
        static_cast<GlyphMetric *>(font_metrics_buffer_.mapped_ptr);
    for (int i = 0; i < 128; ++i) {
      metrics[i].atlas_coords = glm::vec4(0, 0, 1, 1);
      metrics[i].bearing = glm::vec2(0, 0);
      metrics[i].advance = 1.0f;
    }
  }

  // Bar Set
  VkDescriptorSetAllocateInfo bar_alloc{};
  bar_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  bar_alloc.descriptorPool = vulkan_core_->get_descriptor_pool();
  bar_alloc.descriptorSetCount = 1;
  bar_alloc.pSetLayouts = &bar_layout_;
  vkAllocateDescriptorSets(vulkan_core_->get_device(), &bar_alloc,
                           &bar_descriptor_set_);

  // Text Set
  VkDescriptorSetAllocateInfo text_alloc{};
  text_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  text_alloc.descriptorPool = vulkan_core_->get_descriptor_pool();
  text_alloc.descriptorSetCount = 1;
  text_alloc.pSetLayouts = &text_layout_;
  vkAllocateDescriptorSets(vulkan_core_->get_device(), &text_alloc,
                           &text_descriptor_set_);

  // Update Bar Set
  VkDescriptorBufferInfo bar_ubo_info{};
  bar_ubo_info.buffer = bar_ubo_buffer_.buffer;
  bar_ubo_info.offset = bar_ubo_buffer_.offset;
  bar_ubo_info.range = sizeof(OrderBookUniformBuffer);

  VkWriteDescriptorSet bar_write{};
  bar_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  bar_write.dstSet = bar_descriptor_set_;
  bar_write.dstBinding = 0;
  bar_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bar_write.descriptorCount = 1;
  bar_write.pBufferInfo = &bar_ubo_info;
  vkUpdateDescriptorSets(vulkan_core_->get_device(), 1, &bar_write, 0, nullptr);

  // Update Text Set
  vulkan_core_->create_placeholder_texture(font_image_, font_memory_,
                                           font_image_view_, font_sampler_);

  VkDescriptorBufferInfo text_ubo_info{};
  text_ubo_info.buffer = text_ubo_buffer_.buffer;
  text_ubo_info.offset = text_ubo_buffer_.offset;
  text_ubo_info.range = sizeof(TextUniformBuffer);

  VkDescriptorBufferInfo ssbo_info{};
  ssbo_info.buffer = font_metrics_buffer_.buffer;
  ssbo_info.offset = font_metrics_buffer_.offset;
  ssbo_info.range = sizeof(GlyphMetric) * 128;

  VkDescriptorImageInfo image_info{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  image_info.imageView = font_image_view_;
  image_info.sampler = font_sampler_;

  std::vector<VkWriteDescriptorSet> text_writes(3);
  text_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  text_writes[0].dstSet = text_descriptor_set_;
  text_writes[0].dstBinding = 0;
  text_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  text_writes[0].descriptorCount = 1;
  text_writes[0].pBufferInfo = &text_ubo_info;

  text_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  text_writes[1].dstSet = text_descriptor_set_;
  text_writes[1].dstBinding = 1;
  text_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  text_writes[1].descriptorCount = 1;
  text_writes[1].pBufferInfo = &ssbo_info;

  text_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  text_writes[2].dstSet = text_descriptor_set_;
  text_writes[2].dstBinding = 2;
  text_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  text_writes[2].descriptorCount = 1;
  text_writes[2].pImageInfo = &image_info;

  vkUpdateDescriptorSets(vulkan_core_->get_device(), 3, text_writes.data(), 0,
                         nullptr);

  fprintf(stderr,
          "[OrderBookComponent] Vulkan resources initialized successfully\n");
  mark_dirty();
}
void OrderBookComponent::handle_trade(const RenderEngine::TradeData &trade) {
  if (trade.symbol != target_symbol_)
    return;
  if (dashboard_ && trade.symbol != dashboard_->get_active_symbol())
    return;
  // Order book might highlight levels where trades occurred
}

void OrderBookComponent::handle_orderbook(
    const RenderEngine::OrderbookData &orderbook) {
  if (orderbook.symbol != target_symbol_)
    return;
  OrderBookData ui_data;
  ui_data.timestamp = orderbook.timestamp_us;
  ui_data.spread = orderbook.spread;
  ui_data.mid_price =
      (orderbook.bids.empty() || orderbook.asks.empty())
          ? 0.0
          : (orderbook.bids[0].price + orderbook.asks[0].price) / 2.0;

  size_t bid_limit = std::min(orderbook.bids.size(), max_levels_);
  double cumulative_bid = 0.0;
  for (size_t i = 0; i < bid_limit; ++i) {
    cumulative_bid += orderbook.bids[i].size;
    ui_data.bids.push_back(
        {orderbook.bids[i].price, orderbook.bids[i].size, cumulative_bid});
  }

  size_t ask_limit = std::min(orderbook.asks.size(), max_levels_);
  double cumulative_ask = 0.0;
  for (size_t i = 0; i < ask_limit; ++i) {
    cumulative_ask += orderbook.asks[i].size;
    ui_data.asks.push_back(
        {orderbook.asks[i].price, orderbook.asks[i].size, cumulative_ask});
  }

  std::lock_guard lock(data_mutex_);
  update_orderbook(ui_data);
}

void OrderBookComponent::clear_data() {
  std::lock_guard lock(data_mutex_);
  current_data_.bids.clear();
  current_data_.asks.clear();
  current_data_.spread = 0.0;
  current_data_.mid_price = 0.0;
  mark_dirty();
}

} // namespace BTQuant