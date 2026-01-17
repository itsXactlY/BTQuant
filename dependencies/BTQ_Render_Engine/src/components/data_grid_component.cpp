/**
 * BTQuant Advanced Vulkan Dashboard - Data Grid Component Implementation
 *
 * Professional-grade data grid for displaying tabular market data with
 * high-performance GPU rendering, sorting, filtering, and real-time updates.
 *
 * Features:
 * - GPU-accelerated text rendering with instancing
 * - Real-time data updates with minimal CPU overhead
 * - Column sorting and filtering capabilities
 * - Professional styling with color-coded data
 * - Smooth scrolling and selection highlighting
 * - High-DPI display support
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace BTQuant {

// Vertex structure for grid rendering
struct GridVertex {
  glm::vec2 position;
  glm::vec2 texcoord;
  glm::vec4 color;
  uint32_t instance_id;
};

// Uniform buffer for grid rendering
struct GridUniformBuffer {
  glm::mat4 projection;
  glm::mat4 view;
  glm::vec2 grid_size;
  glm::vec2 cell_size;
  float time;
  float highlight_intensity;
  glm::vec2 scroll_offset;
};

DataGridComponent::DataGridComponent(const glm::vec2 &position,
                                     const glm::vec2 &size, size_t rows,
                                     size_t columns)
    : UIComponent(position, size), rows_(rows), columns_(columns) {

  // Initialize grid data
  grid_data_.resize(rows_);
  for (auto &row : grid_data_) {
    row.resize(columns_);
  }

  // Initialize column headers and widths
  column_headers_.resize(columns_);
  column_widths_.resize(columns_, size_.x / columns_);

  // Set default column headers
  for (size_t i = 0; i < columns_; ++i) {
    column_headers_[i] = "Column " + std::to_string(i + 1);
  }

  // Initialize default cell data
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < columns_; ++col) {
      grid_data_[row][col] = {.text = "",
                              .color = theme_.text_primary,
                              .numeric_value = 0.0f,
                              .highlight = false,
                              .is_numeric = false};
    }
  }
}

DataGridComponent::~DataGridComponent() {
  // Cleanup will be handled by Vulkan core
}

void DataGridComponent::set_cell_data(size_t row, size_t col,
                                      const CellData &data) {
  if (row >= rows_ || col >= columns_)
    return;

  std::lock_guard lock(data_mutex_);
  grid_data_[row][col] = data;
  mark_dirty();
}

void DataGridComponent::set_row_data(size_t row,
                                     const std::vector<CellData> &row_data) {
  if (row >= rows_)
    return;

  std::lock_guard lock(data_mutex_);
  size_t cols_to_copy = std::min(row_data.size(), columns_);
  for (size_t col = 0; col < cols_to_copy; ++col) {
    grid_data_[row][col] = row_data[col];
  }
  mark_dirty();
}

void DataGridComponent::set_column_header(size_t col,
                                          const std::string &header) {
  if (col >= columns_)
    return;

  column_headers_[col] = header;
  mark_dirty();
}

void DataGridComponent::set_column_width(size_t col, float width) {
  if (col >= columns_)
    return;

  column_widths_[col] = width;
  mark_dirty();
}

void DataGridComponent::enable_sorting(size_t column, bool ascending) {
  if (column >= columns_)
    return;

  sort_column_ = static_cast<int>(column);
  sort_ascending_ = ascending;
  sort_data();
  mark_dirty();
}

void DataGridComponent::set_filter(const std::string &) {
  // TODO: Implement filtering logic
  // For now, just mark as dirty to trigger rebuild
  mark_dirty();
}

void DataGridComponent::update(float delta_time) {
  if (is_dirty()) {
    rebuild_geometry();
    dirty_frames_--;
  }

  // Update any animations or highlights
  static float highlight_time = 0.0f;
  highlight_time += delta_time;

  // Fade out highlights over time
  for (auto &row : grid_data_) {
    for (auto &cell : row) {
      if (cell.highlight) {
        // Implement highlight fade logic here
      }
    }
  }
}

void DataGridComponent::render(VkCommandBuffer cmd) {
  if (!visible_ || !vertex_buffer_.buffer || pipeline_ == VK_NULL_HANDLE)
    return;

  std::lock_guard lock(data_mutex_);
  // Bind pipeline and resources
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

  // Bind vertex buffer
  VkBuffer vertex_buffers[] = {vertex_buffer_.buffer};
  VkDeviceSize offsets[] = {vertex_buffer_.offset};
  vkCmdBindVertexBuffers(cmd, 0, 1, vertex_buffers, offsets);

  // Bind index buffer
  vkCmdBindIndexBuffer(cmd, index_buffer_.buffer, index_buffer_.offset,
                       VK_INDEX_TYPE_UINT32);

  // Update uniform buffer with current view parameters
  GridUniformBuffer ubo = {};
  ubo.projection = glm::ortho(0.0f, 1920.0f, 1080.0f, 0.0f, -1.0f,
                              1.0f); // TODO: Get from viewport
  ubo.view = glm::mat4(1.0f);
  ubo.grid_size = glm::vec2(columns_, rows_);
  ubo.cell_size =
      glm::vec2(size_.x / columns_, 25.0f); // Fixed row height for now
  ubo.time = static_cast<float>(
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count()) /
             1000.0f;
  ubo.highlight_intensity = 1.0f;
  ubo.scroll_offset = glm::vec2(0.0f); // TODO: Implement scrolling

  // TODO: Update uniform buffer

  // Draw the grid
  uint32_t index_count =
      static_cast<uint32_t>((rows_ + 1) * (columns_ + 1) * 6); // Approximate
  vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
}

void DataGridComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_FirstUseEver);

  ImGui::SetNextWindowCollapsed(minimized_, ImGuiCond_Appearing);
  if (!ImGui::Begin("Market Data Grid", &visible_)) {
    minimized_ = true;
    ImGui::End();
    return;
  }
  minimized_ = false;

  if (theme_.monospace_font)
    ImGui::PushFont((ImFont *)theme_.monospace_font);

  std::lock_guard lock(data_mutex_);
  if (ImGui::BeginTable("DataGridTable", (int)columns_,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_Sortable)) {
    for (size_t i = 0; i < columns_; ++i) {
      ImGui::TableSetupColumn(column_headers_[i].c_str());
    }
    ImGui::TableHeadersRow();

    for (size_t r = 0; r < grid_data_.size(); ++r) {
      ImGui::TableNextRow();
      const auto &row = grid_data_[r];
      for (size_t col = 0; col < columns_; ++col) {
        ImGui::TableSetColumnIndex((int)col);
        const auto &cell = row[col];
        ImVec4 color =
            ImVec4(cell.color.r, cell.color.g, cell.color.b, cell.color.a);

        if (col == 0) {
          // The first column is the symbol. Make it selectable.
          bool is_selected =
              (dashboard_ && dashboard_->get_active_symbol() == cell.text);
          if (ImGui::Selectable(cell.text.c_str(), is_selected,
                                ImGuiSelectableFlags_SpanAllColumns)) {
            if (dashboard_) {
              dashboard_->set_active_symbol(cell.text);
            }
          }
        } else {
          ImGui::TextColored(color, "%s", cell.text.c_str());
        }
      }
    }
    ImGui::EndTable();
  }

  if (theme_.monospace_font)
    ImGui::PopFont();

  ImGui::End();
}

void DataGridComponent::handle_input(const InputEvent &event) {
  switch (event.type) {
  case InputEventType::MouseButton: {
    // Calculate which cell was clicked
    float cell_width = size_.x / columns_;
    float cell_height = 25.0f; // Fixed row height

    int col = static_cast<int>((event.position.x - position_.x) / cell_width);
    int row = static_cast<int>((event.position.y - position_.y - 30.0f) /
                               cell_height); // Account for header

    if (row == -1 && col >= 0 && col < static_cast<int>(columns_)) {
      // Header clicked - enable sorting
      enable_sorting(col, sort_column_ != col || !sort_ascending_);
    }
    break;
  }

  case InputEventType::Scroll:
    // TODO: Implement scrolling
    break;

  default:
    break;
  }
}

void DataGridComponent::rebuild_geometry() {
  // Calculate geometry for grid lines, cells, and text
  std::vector<GridVertex> vertices;
  std::vector<uint32_t> indices;

  float cell_width = size_.x / columns_;
  float cell_height = 25.0f; // Fixed row height
  float header_height = 30.0f;

  uint32_t vertex_index = 0;

  // Generate header background
  for (size_t col = 0; col < columns_; ++col) {
    float x = position_.x + col * cell_width;
    float y = position_.y;

    // Header cell background
    vertices.push_back(
        {{x, y}, {0.0f, 0.0f}, theme_.background_secondary, vertex_index});
    vertices.push_back({{x + cell_width, y},
                        {1.0f, 0.0f},
                        theme_.background_secondary,
                        vertex_index});
    vertices.push_back({{x + cell_width, y + header_height},
                        {1.0f, 1.0f},
                        theme_.background_secondary,
                        vertex_index});
    vertices.push_back({{x, y + header_height},
                        {0.0f, 1.0f},
                        theme_.background_secondary,
                        vertex_index});

    // Header cell indices
    uint32_t base = vertex_index;
    indices.insert(indices.end(),
                   {base, base + 1, base + 2, base, base + 2, base + 3});
    vertex_index += 4;
  }

  // Generate data cell backgrounds
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < columns_; ++col) {
      float x = position_.x + col * cell_width;
      float y = position_.y + header_height + row * cell_height;

      const auto &cell_data = grid_data_[row][col];
      glm::vec4 bg_color =
          (row % 2 == 0) ? theme_.background_panel : theme_.background_primary;

      if (cell_data.highlight) {
        bg_color = glm::mix(bg_color, theme_.accent_primary, 0.3f);
      }

      // Data cell background
      vertices.push_back({{x, y}, {0.0f, 0.0f}, bg_color, vertex_index});
      vertices.push_back(
          {{x + cell_width, y}, {1.0f, 0.0f}, bg_color, vertex_index});
      vertices.push_back({{x + cell_width, y + cell_height},
                          {1.0f, 1.0f},
                          bg_color,
                          vertex_index});
      vertices.push_back(
          {{x, y + cell_height}, {0.0f, 1.0f}, bg_color, vertex_index});

      // Data cell indices
      uint32_t base = vertex_index;
      indices.insert(indices.end(),
                     {base, base + 1, base + 2, base, base + 2, base + 3});
      vertex_index += 4;
    }
  }

  // Generate grid lines
  glm::vec4 line_color = theme_.border_color;

  // Vertical lines
  for (size_t col = 0; col <= columns_; ++col) {
    float x = position_.x + col * cell_width;
    float y1 = position_.y;
    float y2 = position_.y + header_height + rows_ * cell_height;

    vertices.push_back({{x, y1}, {0.0f, 0.0f}, line_color, vertex_index});
    vertices.push_back(
        {{x + 1.0f, y2}, {1.0f, 1.0f}, line_color, vertex_index});

    uint32_t base = vertex_index;
    indices.insert(indices.end(), {base, base + 1});
    vertex_index += 2;
  }

  // Horizontal lines
  for (size_t row = 0; row <= rows_ + 1; ++row) { // +1 for header
    float y =
        position_.y + (row == 0 ? 0 : header_height + (row - 1) * cell_height);
    float x1 = position_.x;
    float x2 = position_.x + size_.x;

    vertices.push_back({{x1, y}, {0.0f, 0.0f}, line_color, vertex_index});
    vertices.push_back(
        {{x2, y + 1.0f}, {1.0f, 1.0f}, line_color, vertex_index});

    uint32_t base = vertex_index;
    indices.insert(indices.end(), {base, base + 1});
    vertex_index += 2;
  }

  // TODO: Allocate and update vertex/index buffers
  // This would require access to the VulkanCore instance
  // For now, we'll store the geometry data for later upload
}

void DataGridComponent::sort_data() {
  if (sort_column_ < 0 || sort_column_ >= static_cast<int>(columns_))
    return;

  std::sort(
      grid_data_.begin(), grid_data_.end(),
      [this](const std::vector<CellData> &a, const std::vector<CellData> &b) {
        const auto &cell_a = a[sort_column_];
        const auto &cell_b = b[sort_column_];

        if (cell_a.is_numeric && cell_b.is_numeric) {
          return sort_ascending_ ? cell_a.numeric_value < cell_b.numeric_value
                                 : cell_a.numeric_value > cell_b.numeric_value;
        } else {
          return sort_ascending_ ? cell_a.text < cell_b.text
                                 : cell_a.text > cell_b.text;
        }
      });
}

void DataGridComponent::initialize_vulkan_resources(VulkanCore *) {
  // TODO: Create pipelines, buffers, etc.
  // For now, just set the pointer
}

} // namespace BTQuant