/**
 * BTQuant Advanced Vulkan Dashboard - Log Display Component Implementation
 *
 * High-performance scrollable log display with real-time updates, filtering,
 * and professional styling for system monitoring and debugging.
 *
 * Features:
 * - Real-time log streaming with minimal latency
 * - Color-coded log levels (Debug, Info, Warning, Error)
 * - Smooth scrolling with mouse wheel and auto-scroll
 * - Text filtering and log level filtering
 * - Timestamp formatting and display
 * - Professional monospace font rendering
 * - Memory-efficient circular buffer for large logs
 * - Interactive selection and copying
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace BTQuant {

// Vertex structure for log text rendering
struct LogTextVertex {
  glm::vec2 position;
  glm::vec2 texcoord;
  glm::vec4 color;
  uint32_t glyph_id;
  float font_size;
  uint32_t log_level;
};

LogDisplayComponent::LogDisplayComponent(const glm::vec2 &position,
                                         const glm::vec2 &size)
    : UIComponent(position, size) {

  // Initialize with default settings
  max_entries_ = 1000;
  auto_scroll_ = true;
  min_log_level_ = Debug;
  text_filter_ = "";
  scroll_offset_ = 0.0f;

  // Reserve space for log entries if needed, but for deque we just start empty
  log_entries_.clear();
}

LogDisplayComponent::~LogDisplayComponent() {
  if (vulkan_core_) {
    VkDevice device = vulkan_core_->get_device();
    if (text_pipeline_ != VK_NULL_HANDLE)
      vkDestroyPipeline(device, text_pipeline_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
    if (descriptor_set_layout_ != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
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

void LogDisplayComponent::add_log_entry(LogLevel level,
                                        const std::string &message) {
  LogEntry entry;
  entry.timestamp = std::chrono::system_clock::now();
  entry.level = level;
  entry.message = message;
  entry.color = get_log_level_color(level);

  log_entries_.push_back(entry);

  // Remove old entries if we exceed the maximum
  while (log_entries_.size() > max_entries_) {
    log_entries_.pop_front();
  }

  // Auto-scroll to bottom if enabled
  if (auto_scroll_) {
    scroll_offset_ = 0.0f; // 0 means scrolled to bottom
  }

  mark_dirty();
}

void LogDisplayComponent::clear_logs() {
  log_entries_.clear();
  scroll_offset_ = 0.0f;
  mark_dirty();
}

void LogDisplayComponent::update(float delta_time) {
  if (is_dirty()) {
    rebuild_text_geometry();
    dirty_frames_--;
  }

  // Update any animations or smooth scrolling
  static float animation_time = 0.0f;
  animation_time += delta_time;

  // TODO: Implement smooth scrolling animations
}

void LogDisplayComponent::handle_trade(const RenderEngine::TradeData &trade) {
  char buffer[128];
  snprintf(buffer, sizeof(buffer), "TRADE: %s %.4f @ %.2f",
           trade.is_buy ? "BUY " : "SELL", trade.size, trade.price);
  add_log_entry(trade.is_buy ? Info : Warning, buffer);
}

void LogDisplayComponent::handle_orderbook(
    const RenderEngine::OrderbookData &orderbook) {
  // We don't log every orderbook update to avoid spam
  // But we could log significant spread changes or imbalances
}

void LogDisplayComponent::render(VkCommandBuffer cmd) {
  if (!visible_ || !text_vertex_buffer_.buffer)
    return;

  // 1. Update UBO
  if (ubo_buffer_.mapped_ptr) {
    TextUniformBuffer ubo{};
    VkExtent2D extent = vulkan_core_->get_swapchain_extent();
    // Swap 0.0f and extent.height to match Vulkan NDC Y direction
    ubo.projection = glm::ortho(0.0f, (float)extent.width, 0.0f,
                                (float)extent.height, -1.0f, 1.0f);
    ubo.view = glm::mat4(1.0f);
    ubo.viewport_size = glm::vec2(extent.width, extent.height);
    ubo.dpi_scale = glm::vec2(1.0f);
    ubo.time = 0.0f; // TODO: Get elapsed time
    ubo.global_text_color = glm::vec4(1.0f);
    ubo.render_flags = 0x10; // RENDER_GAMMA_CORRECT
    memcpy(ubo_buffer_.mapped_ptr, &ubo, sizeof(ubo));
  }

  // 2. Bind pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, text_pipeline_);

  // 3. Setup push constants
  struct {
    glm::vec2 text_offset;
    float text_scale_factor;
  } push;
  push.text_offset = position_;
  push.text_scale_factor = 1.0f;

  vkCmdPushConstants(cmd, pipeline_layout_,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(push), &push);

  // 3. Bind descriptor set
  if (descriptor_set_ != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0,
                            nullptr);
  }

  // 4. Bind vertex buffer
  VkBuffer buffers[] = {text_vertex_buffer_.buffer};
  VkDeviceSize offsets[] = {text_vertex_buffer_.offset};
  vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);

  // 5. Draw
  uint32_t vertex_count =
      static_cast<uint32_t>(text_vertex_buffer_.size / sizeof(LogTextVertex));
  if (vertex_count > 0) {
    vkCmdDraw(cmd, vertex_count, 1, 0, 0);
  }
}

void LogDisplayComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_FirstUseEver);

  ImGui::SetNextWindowCollapsed(minimized_, ImGuiCond_Appearing);
  if (!ImGui::Begin("System Logs", &visible_)) {
    minimized_ = true;
    ImGui::End();
    return;
  }
  minimized_ = false;

  // Filtering and Controls
  if (ImGui::Button("Clear")) {
    clear_logs();
  }
  ImGui::SameLine();
  ImGui::Checkbox("Auto-scroll", &auto_scroll_);

  ImGui::Separator();

  // Scrolling Region
  const float footer_height_to_reserve =
      ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
  ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve),
                    false, ImGuiWindowFlags_HorizontalScrollbar);

  for (const auto &entry : log_entries_) {
    // Basic filtering
    if (entry.level < min_log_level_)
      continue;
    if (!text_filter_.empty() &&
        entry.message.find(text_filter_) == std::string::npos)
      continue;

    ImVec4 color =
        ImVec4(entry.color.r, entry.color.g, entry.color.b, entry.color.a);
    ImGui::TextColored(
        color, "[%s] [%s] %s", format_timestamp(entry.timestamp).c_str(),
        get_log_level_string(entry.level).c_str(), entry.message.c_str());
  }

  if (auto_scroll_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
    ImGui::SetScrollHereY(1.0f);
  }

  ImGui::EndChild();
  ImGui::End();
}

void LogDisplayComponent::handle_input(const InputEvent &event) {
  switch (event.type) {
  case InputEventType::Scroll: {
    // Handle scrolling
    float line_height = 16.0f;
    float scroll_delta =
        event.scroll_delta.y * line_height * 3.0f; // 3 lines per scroll

    scroll_offset_ += scroll_delta;

    // Clamp scroll offset
    float max_scroll = std::max(
        0.0f, static_cast<float>(get_filtered_entries().size()) * line_height -
                  size_.y);
    scroll_offset_ = glm::clamp(scroll_offset_, 0.0f, max_scroll);

    // Disable auto-scroll if user scrolls up
    if (scroll_offset_ > 0.1f) {
      auto_scroll_ = false;
    } else {
      auto_scroll_ = true;
    }

    mark_dirty();
    break;
  }

  case InputEventType::MouseButton:
    if (event.pressed) {
      // Handle log line selection
      float line_height = 16.0f;
      float local_y = event.position.y - position_.y + scroll_offset_;
      int line_index = static_cast<int>(local_y / line_height);

      auto filtered_entries = get_filtered_entries();
      if (line_index >= 0 &&
          line_index < static_cast<int>(filtered_entries.size())) {
        // TODO: Handle log line selection
        // Could highlight the line or copy to clipboard
      }
    }
    break;

  case InputEventType::KeyDown:
    // Handle keyboard shortcuts
    if (event.key == 'C' && /* Ctrl pressed */ false) {
      // TODO: Copy selected log entries to clipboard
    } else if (event.key == 'F' && /* Ctrl pressed */ false) {
      // TODO: Open search/filter dialog
    }
    break;

  default:
    break;
  }
}

void LogDisplayComponent::initialize_vulkan_resources(VulkanCore *vulkan_core) {
  vulkan_core_ = vulkan_core;

  // 1. Create descriptor set layout
  VkDescriptorSetLayoutBinding bindings[3] = {};

  // Binding 0: Uniform Buffer (UBO)
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  // Binding 1: Font Metrics (SSBO)
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  // Binding 2: Font Atlas (Sampler)
  bindings[2].binding = 2;
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 3;
  layout_info.pBindings = bindings;

  VulkanErrorHandler::check_result(
      vkCreateDescriptorSetLayout(vulkan_core_->get_device(), &layout_info,
                                  nullptr, &descriptor_set_layout_),
      "vkCreateDescriptorSetLayout");

  // 2. Create pipeline layout with push constants
  VkPushConstantRange push_constant{};
  push_constant.offset = 0;
  push_constant.size =
      64; // Match shaders (vec2, float, vec4, uint, float, float, float)
  push_constant.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pPushConstantRanges = &push_constant;

  VulkanErrorHandler::check_result(
      vkCreatePipelineLayout(vulkan_core_->get_device(), &pipeline_layout_info,
                             nullptr, &pipeline_layout_),
      "vkCreatePipelineLayout");

  // 3. Define vertex input
  VkVertexInputBindingDescription binding_desc{};
  binding_desc.binding = 0;
  binding_desc.stride = sizeof(LogTextVertex);
  binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::vector<VkVertexInputAttributeDescription> attr_descs(6);
  attr_descs[0].location = 0;
  attr_descs[0].binding = 0;
  attr_descs[0].format = VK_FORMAT_R32G32_SFLOAT;
  attr_descs[0].offset = offsetof(LogTextVertex, position);
  attr_descs[1].location = 1;
  attr_descs[1].binding = 0;
  attr_descs[1].format = VK_FORMAT_R32G32_SFLOAT;
  attr_descs[1].offset = offsetof(LogTextVertex, texcoord);
  attr_descs[2].location = 2;
  attr_descs[2].binding = 0;
  attr_descs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attr_descs[2].offset = offsetof(LogTextVertex, color);
  attr_descs[3].location = 3;
  attr_descs[3].binding = 0;
  attr_descs[3].format = VK_FORMAT_R32_UINT;
  attr_descs[3].offset = offsetof(LogTextVertex, glyph_id);
  attr_descs[4].location = 4;
  attr_descs[4].binding = 0;
  attr_descs[4].format = VK_FORMAT_R32_SFLOAT;
  attr_descs[4].offset = offsetof(LogTextVertex, font_size);
  attr_descs[5].location = 5;
  attr_descs[5].binding = 0;
  attr_descs[5].format = VK_FORMAT_R32_UINT;
  attr_descs[5].offset = offsetof(LogTextVertex, log_level);

  // 4. Create graphics pipeline
  std::vector<VkVertexInputBindingDescription> bindings_vec = {binding_desc};
  text_pipeline_ = vulkan_core_->create_graphics_pipeline(
      "shaders/text_rendering.vert.spv", "shaders/text_rendering.frag.spv",
      bindings_vec, attr_descs, pipeline_layout_);

  // 5. Create and update descriptor set
  ubo_buffer_ = vulkan_core_->get_memory_manager().allocate_uniform_buffer(
      sizeof(TextUniformBuffer));

  // Create dummy font metrics (128 chars)
  font_metrics_buffer_ =
      vulkan_core_->get_memory_manager().allocate_storage_buffer(
          sizeof(GlyphMetric) * 128);

  if (font_metrics_buffer_.mapped_ptr) {
    GlyphMetric *metrics =
        static_cast<GlyphMetric *>(font_metrics_buffer_.mapped_ptr);
    for (int i = 0; i < 128; ++i) {
      metrics[i].atlas_coords = glm::vec4(0, 0, 1, 1); // Full atlas
      metrics[i].bearing = glm::vec2(0, 0);
      metrics[i].advance = 1.0f;
    }
  }

  // Placeholder font texture
  vulkan_core_->create_placeholder_texture(font_image_, font_memory_,
                                           font_image_view_, font_sampler_);

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = vulkan_core_->get_descriptor_pool();
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &descriptor_set_layout_;

  VulkanErrorHandler::check_result(
      vkAllocateDescriptorSets(vulkan_core_->get_device(), &alloc_info,
                               &descriptor_set_),
      "vkAllocateDescriptorSets");

  // Update descriptor set
  VkDescriptorBufferInfo ubo_info{};
  ubo_info.buffer = ubo_buffer_.buffer;
  ubo_info.offset = ubo_buffer_.offset;
  ubo_info.range = sizeof(TextUniformBuffer);

  VkDescriptorBufferInfo ssbo_info{};
  ssbo_info.buffer = font_metrics_buffer_.buffer;
  ssbo_info.offset = font_metrics_buffer_.offset;
  ssbo_info.range = sizeof(GlyphMetric) * 128;

  VkDescriptorImageInfo image_info{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  image_info.imageView = font_image_view_;
  image_info.sampler = font_sampler_;

  std::vector<VkWriteDescriptorSet> descriptor_writes(3);
  descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptor_writes[0].dstSet = descriptor_set_;
  descriptor_writes[0].dstBinding = 0;
  descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descriptor_writes[0].descriptorCount = 1;
  descriptor_writes[0].pBufferInfo = &ubo_info;

  descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptor_writes[1].dstSet = descriptor_set_;
  descriptor_writes[1].dstBinding = 1;
  descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_writes[1].descriptorCount = 1;
  descriptor_writes[1].pBufferInfo = &ssbo_info;

  descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptor_writes[2].dstSet = descriptor_set_;
  descriptor_writes[2].dstBinding = 2;
  descriptor_writes[2].descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  descriptor_writes[2].descriptorCount = 1;
  descriptor_writes[2].pImageInfo = &image_info;

  vkUpdateDescriptorSets(vulkan_core_->get_device(),
                         static_cast<uint32_t>(descriptor_writes.size()),
                         descriptor_writes.data(), 0, nullptr);

  fprintf(stderr,
          "[LogDisplayComponent] Vulkan resources initialized successfully\n");
  mark_dirty();
}

void LogDisplayComponent::rebuild_text_geometry() {
  if (!vulkan_core_)
    return;

  std::vector<LogTextVertex> vertices;

  float line_height = 16.0f;
  float font_size = 12.0f;
  float char_width = font_size * 0.6f; // Monospace approximation

  auto filtered_entries = get_filtered_entries();

  // Calculate which lines are visible based on scroll offset
  int start_line = static_cast<int>(scroll_offset_ / line_height);
  int visible_lines =
      static_cast<int>(size_.y / line_height) + 2; // +2 for partial lines
  int end_line = std::min(start_line + visible_lines,
                          static_cast<int>(filtered_entries.size()));

  for (int i = start_line; i < end_line; ++i) {
    const auto &entry = filtered_entries[i];

    float y = position_.y + (i * line_height) - scroll_offset_;

    // Skip if line is outside visible area
    if (y < position_.y - line_height ||
        y > position_.y + size_.y + line_height) {
      continue;
    }

    // Format the log line
    std::string timestamp_str = format_timestamp(entry.timestamp);
    std::string level_str = get_log_level_string(entry.level);
    std::string full_line =
        timestamp_str + " [" + level_str + "] " + entry.message;

    // Truncate line if it's too long
    float max_chars = size_.x / char_width;
    if (full_line.length() > max_chars) {
      full_line =
          full_line.substr(0, static_cast<size_t>(max_chars - 3)) + "...";
    }

    // Generate vertices for each character
    float current_x = position_.x + 5.0f; // Small left margin

    for (size_t char_idx = 0; char_idx < full_line.length(); ++char_idx) {
      char c = full_line[char_idx];
      uint32_t glyph_id = static_cast<uint32_t>(c);

      // Determine color based on position in the line
      glm::vec4 char_color = entry.color;
      if (char_idx < timestamp_str.length()) {
        char_color = theme_.text_muted; // Timestamp in muted color
      } else if (char_idx < timestamp_str.length() + level_str.length() + 3) {
        char_color = entry.color; // Log level in level color
      } else {
        char_color = theme_.text_primary; // Message in primary color
      }

      // Create quad for character
      vertices.push_back({{current_x, y},
                          {0.0f, 0.0f},
                          char_color,
                          glyph_id,
                          font_size,
                          static_cast<uint32_t>(entry.level)});
      vertices.push_back({{current_x + char_width, y},
                          {1.0f, 0.0f},
                          char_color,
                          glyph_id,
                          font_size,
                          static_cast<uint32_t>(entry.level)});
      vertices.push_back({{current_x + char_width, y + font_size},
                          {1.0f, 1.0f},
                          char_color,
                          glyph_id,
                          font_size,
                          static_cast<uint32_t>(entry.level)});

      vertices.push_back({{current_x, y},
                          {0.0f, 0.0f},
                          char_color,
                          glyph_id,
                          font_size,
                          static_cast<uint32_t>(entry.level)});
      vertices.push_back({{current_x + char_width, y + font_size},
                          {1.0f, 1.0f},
                          char_color,
                          glyph_id,
                          font_size,
                          static_cast<uint32_t>(entry.level)});
      vertices.push_back({{current_x, y + font_size},
                          {0.0f, 1.0f},
                          char_color,
                          glyph_id,
                          font_size,
                          static_cast<uint32_t>(entry.level)});

      current_x += char_width;

      // Stop if we exceed the component width
      if (current_x > position_.x + size_.x) {
        break;
      }
    }
  }

  if (vertices.empty()) {
    return;
  }

  size_t buffer_size = vertices.size() * sizeof(LogTextVertex);

  // Reallocate buffer if necessary
  if (!text_vertex_buffer_.buffer || text_vertex_buffer_.size < buffer_size) {
    if (text_vertex_buffer_.buffer) {
      vulkan_core_->get_memory_manager().deallocate_buffer(text_vertex_buffer_);
    }
    text_vertex_buffer_ =
        vulkan_core_->get_memory_manager().allocate_vertex_buffer(buffer_size);
  }

  // Copy data to GPU-mapped memory
  if (text_vertex_buffer_.mapped_ptr) {
    memcpy(text_vertex_buffer_.mapped_ptr, vertices.data(), buffer_size);
  }
}

glm::vec4 LogDisplayComponent::get_log_level_color(LogLevel level) {
  switch (level) {
  case Debug:
    return theme_.text_muted;
  case Info:
    return theme_.text_primary;
  case Warning:
    return glm::vec4(1.0f, 0.8f, 0.0f, 1.0f); // Yellow
  case Error:
    return theme_.price_down; // Red
  default:
    return theme_.text_primary;
  }
}

std::string LogDisplayComponent::get_log_level_string(LogLevel level) {
  switch (level) {
  case Debug:
    return "DEBUG";
  case Info:
    return "INFO ";
  case Warning:
    return "WARN ";
  case Error:
    return "ERROR";
  default:
    return "UNKN ";
  }
}

std::string LogDisplayComponent::format_timestamp(
    const std::chrono::system_clock::time_point &time) {
  auto time_t = std::chrono::system_clock::to_time_t(time);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                time.time_since_epoch()) %
            1000;

  std::ostringstream oss;
  oss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

  return oss.str();
}

std::vector<LogDisplayComponent::LogEntry>
LogDisplayComponent::get_filtered_entries() const {
  std::vector<LogEntry> filtered;

  for (const auto &entry : log_entries_) {
    // Filter by log level
    if (entry.level < min_log_level_) {
      continue;
    }

    // Filter by text
    if (!text_filter_.empty()) {
      std::string message_lower = entry.message;
      std::string filter_lower = text_filter_;

      // Convert to lowercase for case-insensitive search
      std::transform(message_lower.begin(), message_lower.end(),
                     message_lower.begin(), ::tolower);
      std::transform(filter_lower.begin(), filter_lower.end(),
                     filter_lower.begin(), ::tolower);

      if (message_lower.find(filter_lower) == std::string::npos) {
        continue;
      }
    }

    filtered.push_back(entry);
  }

  return filtered;
}

void LogDisplayComponent::clear_data() {
  clear_logs();
}

} // namespace BTQuant