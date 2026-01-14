/**
 * BTQuant Professional UI Enhancements
 *
 * Advanced UI features for professional trading dashboard including
 * resizable panels, customizable layouts, filtering, search, and themes.
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include "../include/interaction_manager.hpp"
#include <algorithm>
#include <fstream>
#include <json/json.h>
#include <regex>

namespace BTQuant {

// ============================================================================
// Resizable Panel System
// ============================================================================

class ResizablePanel : public UIComponent {
public:
  enum class ResizeHandle {
    NONE,
    TOP,
    BOTTOM,
    LEFT,
    RIGHT,
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT
  };

  ResizablePanel(const glm::vec2 &position, const glm::vec2 &size,
                 const std::string &title)
      : UIComponent(position, size), title_(title) {
    min_size_ = glm::vec2(100.0f, 50.0f);
    max_size_ = glm::vec2(2000.0f, 1500.0f);
    handle_size_ = 8.0f;
    snap_threshold_ = 10.0f;
  }

  void set_content(std::unique_ptr<UIComponent> content) {
    content_ = std::move(content);
    update_content_layout();
  }

  void set_resizable(bool resizable) { resizable_ = resizable; }
  void set_min_size(const glm::vec2 &min_size) { min_size_ = min_size; }
  void set_max_size(const glm::vec2 &max_size) { max_size_ = max_size; }
  void set_snap_to_grid(bool snap, float grid_size = 10.0f) {
    snap_to_grid_ = snap;
    grid_size_ = grid_size;
  }

  void update(float delta_time) override {
    if (content_) {
      content_->update(delta_time);
    }

    // Update resize animation
    if (resize_animation_active_) {
      resize_animation_time_ += delta_time;
      float t =
          std::min(resize_animation_time_ / resize_animation_duration_, 1.0f);

      // Smooth easing function
      t = t * t * (3.0f - 2.0f * t);

      glm::vec2 current_size =
          glm::mix(resize_start_size_, resize_target_size_, t);
      set_size(current_size);

      if (t >= 1.0f) {
        resize_animation_active_ = false;
      }
    }
  }

  void render(VkCommandBuffer cmd) override {
    // Render panel background
    render_panel_background(cmd);

    // Render title bar
    render_title_bar(cmd);

    // Render content
    if (content_ && content_->is_visible()) {
      content_->render(cmd);
    }

    // Render resize handles
    if (resizable_ && (hovered_ || resizing_)) {
      render_resize_handles(cmd);
    }

    // Render snap guides
    if (resizing_ && snap_to_grid_) {
      render_snap_guides(cmd);
    }
  }

  void handle_input(const InputEvent &event) override {
    if (!visible_)
      return;

    // Handle resize operations
    if (resizable_ && handle_resize_input(event)) {
      return;
    }

    // Handle title bar dragging
    if (handle_title_bar_input(event)) {
      return;
    }

    // Forward to content
    if (content_ && is_point_in_content_area(event.position)) {
      InputEvent content_event = event;
      content_event.position -= get_content_offset();
      content_->handle_input(content_event);
    }
  }

  void animate_resize(const glm::vec2 &target_size, float duration = 0.3f) {
    resize_start_size_ = size_;
    resize_target_size_ = clamp_size(target_size);
    resize_animation_duration_ = duration;
    resize_animation_time_ = 0.0f;
    resize_animation_active_ = true;
  }

private:
  std::string title_;
  std::unique_ptr<UIComponent> content_;

  // Resize properties
  bool resizable_ = true;
  glm::vec2 min_size_{100.0f, 50.0f};
  glm::vec2 max_size_{2000.0f, 1500.0f};
  float handle_size_ = 8.0f;

  // Snap to grid
  bool snap_to_grid_ = false;
  float grid_size_ = 10.0f;
  float snap_threshold_ = 10.0f;

  // Resize state
  bool resizing_ = false;
  bool hovered_ = false;
  ResizeHandle active_handle_ = ResizeHandle::NONE;
  glm::vec2 resize_start_pos_;
  glm::vec2 resize_start_size_;

  // Animation
  bool resize_animation_active_ = false;
  float resize_animation_time_ = 0.0f;
  float resize_animation_duration_ = 0.3f;
  glm::vec2 resize_target_size_;

  // Title bar
  float title_bar_height_ = 30.0f;
  bool dragging_title_ = false;
  glm::vec2 drag_offset_;

  // Rendering resources
  BufferAllocation panel_vertex_buffer_;
  BufferAllocation handle_vertex_buffer_;
  VkPipeline panel_pipeline_ = VK_NULL_HANDLE;
  VkPipeline handle_pipeline_ = VK_NULL_HANDLE;

  ResizeHandle get_resize_handle_at_position(const glm::vec2 &pos) {
    glm::vec2 relative_pos = pos - position_;

    // Check corners first (larger hit area)
    float corner_size = handle_size_ * 1.5f;

    if (relative_pos.x <= corner_size && relative_pos.y <= corner_size) {
      return ResizeHandle::TOP_LEFT;
    }
    if (relative_pos.x >= size_.x - corner_size &&
        relative_pos.y <= corner_size) {
      return ResizeHandle::TOP_RIGHT;
    }
    if (relative_pos.x <= corner_size &&
        relative_pos.y >= size_.y - corner_size) {
      return ResizeHandle::BOTTOM_LEFT;
    }
    if (relative_pos.x >= size_.x - corner_size &&
        relative_pos.y >= size_.y - corner_size) {
      return ResizeHandle::BOTTOM_RIGHT;
    }

    // Check edges
    if (relative_pos.y <= handle_size_) {
      return ResizeHandle::TOP;
    }
    if (relative_pos.y >= size_.y - handle_size_) {
      return ResizeHandle::BOTTOM;
    }
    if (relative_pos.x <= handle_size_) {
      return ResizeHandle::LEFT;
    }
    if (relative_pos.x >= size_.x - handle_size_) {
      return ResizeHandle::RIGHT;
    }

    return ResizeHandle::NONE;
  }

  bool handle_resize_input(const InputEvent &event) {
    if (event.type == InputEventType::MouseMove) {
      ResizeHandle handle = get_resize_handle_at_position(event.position);
      hovered_ = (handle != ResizeHandle::NONE);

      if (resizing_) {
        perform_resize(event.position);
        return true;
      }

      // Update cursor based on handle
      update_cursor_for_handle(handle);
      return hovered_;
    }

    if (event.type == InputEventType::MouseButton) {
      if (event.mouse_button == MouseButton::Left) {
        ResizeHandle handle = get_resize_handle_at_position(event.position);

        if (handle != ResizeHandle::NONE) {
          resizing_ = true;
          active_handle_ = handle;
          resize_start_pos_ = event.position;
          resize_start_size_ = size_;
          return true;
        } else if (resizing_) {
          resizing_ = false;
          active_handle_ = ResizeHandle::NONE;
          return true;
        }
      }
    }

    return false;
  }

  bool handle_title_bar_input(const InputEvent &event) {
    glm::vec2 title_bar_min = position_;
    glm::vec2 title_bar_max = position_ + glm::vec2(size_.x, title_bar_height_);

    bool in_title_bar = event.position.x >= title_bar_min.x &&
                        event.position.x <= title_bar_max.x &&
                        event.position.y >= title_bar_min.y &&
                        event.position.y <= title_bar_max.y;

    if (event.type == InputEventType::MouseButton &&
        event.mouse_button == MouseButton::Left && in_title_bar) {

      dragging_title_ = true;
      drag_offset_ = event.position - position_;
      return true;
    }

    if (event.type == InputEventType::MouseMove && dragging_title_) {
      glm::vec2 new_position = event.position - drag_offset_;

      // Snap to grid if enabled
      if (snap_to_grid_) {
        new_position = snap_to_grid_position(new_position);
      }

      set_position(new_position);
      return true;
    }

    if (dragging_title_ && event.type == InputEventType::MouseButton &&
        event.mouse_button == MouseButton::Left) {
      dragging_title_ = false;
      return true;
    }

    return false;
  }

  void perform_resize(const glm::vec2 &mouse_pos) {
    glm::vec2 delta = mouse_pos - resize_start_pos_;
    glm::vec2 new_size = resize_start_size_;
    glm::vec2 new_position = position_;

    switch (active_handle_) {
    case ResizeHandle::NONE:
      // No resize
      break;
    case ResizeHandle::RIGHT:
      new_size.x += delta.x;
      break;
    case ResizeHandle::BOTTOM:
      new_size.y += delta.y;
      break;
    case ResizeHandle::LEFT:
      new_size.x -= delta.x;
      new_position.x += delta.x;
      break;
    case ResizeHandle::TOP:
      new_size.y -= delta.y;
      new_position.y += delta.y;
      break;
    case ResizeHandle::BOTTOM_RIGHT:
      new_size += delta;
      break;
    case ResizeHandle::BOTTOM_LEFT:
      new_size.x -= delta.x;
      new_size.y += delta.y;
      new_position.x += delta.x;
      break;
    case ResizeHandle::TOP_RIGHT:
      new_size.x += delta.x;
      new_size.y -= delta.y;
      new_position.y += delta.y;
      break;
    case ResizeHandle::TOP_LEFT:
      new_size -= delta;
      new_position += delta;
      break;
    }

    // Clamp size
    new_size = clamp_size(new_size);

    // Snap to grid if enabled
    if (snap_to_grid_) {
      new_size = snap_to_grid_size(new_size);
      new_position = snap_to_grid_position(new_position);
    }

    set_size(new_size);
    set_position(new_position);
    update_content_layout();
  }

  glm::vec2 clamp_size(const glm::vec2 &size) {
    return glm::clamp(size, min_size_, max_size_);
  }

  glm::vec2 snap_to_grid_position(const glm::vec2 &pos) {
    return glm::vec2(std::round(pos.x / grid_size_) * grid_size_,
                     std::round(pos.y / grid_size_) * grid_size_);
  }

  glm::vec2 snap_to_grid_size(const glm::vec2 &size) {
    return glm::vec2(std::round(size.x / grid_size_) * grid_size_,
                     std::round(size.y / grid_size_) * grid_size_);
  }

  void update_cursor_for_handle(ResizeHandle handle) {
    // Update system cursor based on resize handle
    // Implementation would set appropriate cursor
  }

  bool is_point_in_content_area(const glm::vec2 &point) {
    glm::vec2 content_min = position_ + glm::vec2(0, title_bar_height_);
    glm::vec2 content_max = position_ + size_;

    return point.x >= content_min.x && point.x <= content_max.x &&
           point.y >= content_min.y && point.y <= content_max.y;
  }

  glm::vec2 get_content_offset() { return glm::vec2(0, title_bar_height_); }

  void update_content_layout() {
    if (content_) {
      glm::vec2 content_size = size_ - glm::vec2(0, title_bar_height_);
      content_->set_size(content_size);
      content_->set_position(position_ + get_content_offset());
    }
  }

  void render_panel_background(VkCommandBuffer cmd) {
    // Render panel background with theme colors
  }

  void render_title_bar(VkCommandBuffer cmd) {
    // Render title bar with title text
  }

  void render_resize_handles(VkCommandBuffer cmd) {
    // Render resize handles at panel edges
  }

  void render_snap_guides(VkCommandBuffer cmd) {
    // Render grid snap guides during resize
  }
};

// ============================================================================
// Layout Management System
// ============================================================================

class LayoutManager {
public:
  struct LayoutPreset {
    std::string name;
    std::string description;
    Json::Value layout_data;
    std::chrono::system_clock::time_point created_time;
    std::chrono::system_clock::time_point modified_time;
  };

  LayoutManager(const std::string &presets_directory = "layouts/")
      : presets_directory_(presets_directory) {
    create_directory_if_not_exists(presets_directory_);
    load_all_presets();
  }

  void save_layout(const std::string &name,
                   const std::string &description = "") {
    LayoutPreset preset;
    preset.name = name;
    preset.description = description;
    preset.layout_data = serialize_current_layout();
    preset.created_time = std::chrono::system_clock::now();
    preset.modified_time = preset.created_time;

    presets_[name] = preset;
    save_preset_to_file(preset);
  }

  bool load_layout(const std::string &name) {
    auto it = presets_.find(name);
    if (it == presets_.end()) {
      return false;
    }

    return deserialize_layout(it->second.layout_data);
  }

  void delete_layout(const std::string &name) {
    auto it = presets_.find(name);
    if (it != presets_.end()) {
      std::string filename = presets_directory_ + name + ".json";
      std::remove(filename.c_str());
      presets_.erase(it);
    }
  }

  std::vector<LayoutPreset> get_all_presets() const {
    std::vector<LayoutPreset> result;
    for (const auto &pair : presets_) {
      result.push_back(pair.second);
    }

    // Sort by modified time (most recent first)
    std::sort(result.begin(), result.end(),
              [](const LayoutPreset &a, const LayoutPreset &b) {
                return a.modified_time > b.modified_time;
              });

    return result;
  }

  void create_default_layouts() {
    // Trading layout
    create_trading_layout();

    // Analysis layout
    create_analysis_layout();

    // Monitoring layout
    create_monitoring_layout();
  }

private:
  std::string presets_directory_;
  std::unordered_map<std::string, LayoutPreset> presets_;

  void create_directory_if_not_exists(const std::string &path) {
    // Implementation would create directory
  }

  void load_all_presets() {
    // Implementation would scan directory and load all preset files
  }

  void save_preset_to_file(const LayoutPreset &preset) {
    std::string filename = presets_directory_ + preset.name + ".json";
    std::ofstream file(filename);

    Json::Value root;
    root["name"] = preset.name;
    root["description"] = preset.description;
    root["layout"] = preset.layout_data;
    root["created_time"] =
        static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                 preset.created_time.time_since_epoch())
                                 .count());
    root["modified_time"] =
        static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                 preset.modified_time.time_since_epoch())
                                 .count());

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &file);
  }

  Json::Value serialize_current_layout() {
    Json::Value layout;

    // Serialize panel positions, sizes, and content types
    // This would iterate through all panels and serialize their state

    return layout;
  }

  bool deserialize_layout(const Json::Value &layout_data) {
    // Deserialize and recreate layout from JSON data
    return true;
  }

  void create_trading_layout() {
    // Create a layout optimized for trading
    Json::Value trading_layout;

    // Main chart panel (60% width, 70% height)
    Json::Value chart_panel;
    chart_panel["type"] = "chart";
    chart_panel["position"]["x"] = 0;
    chart_panel["position"]["y"] = 0;
    chart_panel["size"]["width"] = 0.6;
    chart_panel["size"]["height"] = 0.7;
    trading_layout["panels"].append(chart_panel);

    // Order book panel (40% width, 35% height)
    Json::Value orderbook_panel;
    orderbook_panel["type"] = "orderbook";
    orderbook_panel["position"]["x"] = 0.6;
    orderbook_panel["position"]["y"] = 0;
    orderbook_panel["size"]["width"] = 0.4;
    orderbook_panel["size"]["height"] = 0.35;
    trading_layout["panels"].append(orderbook_panel);

    // Trade history panel (40% width, 35% height)
    Json::Value trades_panel;
    trades_panel["type"] = "trades";
    trades_panel["position"]["x"] = 0.6;
    trades_panel["position"]["y"] = 0.35;
    trades_panel["size"]["width"] = 0.4;
    trades_panel["size"]["height"] = 0.35;
    trading_layout["panels"].append(trades_panel);

    // Portfolio panel (60% width, 30% height)
    Json::Value portfolio_panel;
    portfolio_panel["type"] = "portfolio";
    portfolio_panel["position"]["x"] = 0;
    portfolio_panel["position"]["y"] = 0.7;
    portfolio_panel["size"]["width"] = 0.6;
    portfolio_panel["size"]["height"] = 0.3;
    trading_layout["panels"].append(portfolio_panel);

    // Log panel (40% width, 30% height)
    Json::Value log_panel;
    log_panel["type"] = "log";
    log_panel["position"]["x"] = 0.6;
    log_panel["position"]["y"] = 0.7;
    log_panel["size"]["width"] = 0.4;
    log_panel["size"]["height"] = 0.3;
    trading_layout["panels"].append(log_panel);

    LayoutPreset preset;
    preset.name = "Trading";
    preset.description = "Optimized layout for active trading";
    preset.layout_data = trading_layout;
    preset.created_time = std::chrono::system_clock::now();
    preset.modified_time = preset.created_time;

    presets_["Trading"] = preset;
    save_preset_to_file(preset);
  }

  void create_analysis_layout() {
    // Create a layout optimized for market analysis
    Json::Value analysis_layout;

    // Large chart panel for detailed analysis
    Json::Value chart_panel;
    chart_panel["type"] = "chart";
    chart_panel["position"]["x"] = 0;
    chart_panel["position"]["y"] = 0;
    chart_panel["size"]["width"] = 0.8;
    chart_panel["size"]["height"] = 0.6;
    analysis_layout["panels"].append(chart_panel);

    // Heatmap panel
    Json::Value heatmap_panel;
    heatmap_panel["type"] = "heatmap";
    heatmap_panel["position"]["x"] = 0.8;
    heatmap_panel["position"]["y"] = 0;
    heatmap_panel["size"]["width"] = 0.2;
    heatmap_panel["size"]["height"] = 0.6;
    analysis_layout["panels"].append(heatmap_panel);

    // Data grid for detailed analysis
    Json::Value data_panel;
    data_panel["type"] = "data_grid";
    data_panel["position"]["x"] = 0;
    data_panel["position"]["y"] = 0.6;
    data_panel["size"]["width"] = 1.0;
    data_panel["size"]["height"] = 0.4;
    analysis_layout["panels"].append(data_panel);

    LayoutPreset preset;
    preset.name = "Analysis";
    preset.description = "Layout for market analysis and research";
    preset.layout_data = analysis_layout;
    preset.created_time = std::chrono::system_clock::now();
    preset.modified_time = preset.created_time;

    presets_["Analysis"] = preset;
    save_preset_to_file(preset);
  }

  void create_monitoring_layout() {
    // Create a layout optimized for monitoring multiple markets
    Json::Value monitoring_layout;

    // Grid of small chart panels
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 3; ++col) {
        Json::Value chart_panel;
        chart_panel["type"] = "chart";
        chart_panel["position"]["x"] = col * 0.33;
        chart_panel["position"]["y"] = row * 0.4;
        chart_panel["size"]["width"] = 0.33;
        chart_panel["size"]["height"] = 0.4;
        chart_panel["symbol"] = "SYMBOL_" + std::to_string(row * 3 + col + 1);
        monitoring_layout["panels"].append(chart_panel);
      }
    }

    // Status panel at bottom
    Json::Value status_panel;
    status_panel["type"] = "log";
    status_panel["position"]["x"] = 0;
    status_panel["position"]["y"] = 0.8;
    status_panel["size"]["width"] = 1.0;
    status_panel["size"]["height"] = 0.2;
    monitoring_layout["panels"].append(status_panel);

    LayoutPreset preset;
    preset.name = "Monitoring";
    preset.description = "Multi-market monitoring layout";
    preset.layout_data = monitoring_layout;
    preset.created_time = std::chrono::system_clock::now();
    preset.modified_time = preset.created_time;

    presets_["Monitoring"] = preset;
    save_preset_to_file(preset);
  }
};

// ============================================================================
// Advanced Filtering and Search System
// ============================================================================

class DataFilter {
public:
  enum class FilterType { Text, Numeric, Date, Boolean, Enum };

  enum class ComparisonOperator {
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Contains,
    StartsWith,
    EndsWith,
    Regex
  };

  struct FilterCriteria {
    std::string field_name;
    FilterType type;
    ComparisonOperator operator_;
    std::string value;
    bool case_sensitive = false;
    bool enabled = true;
  };

  DataFilter() = default;

  void add_filter(const FilterCriteria &criteria) {
    filters_.push_back(criteria);
  }

  void remove_filter(size_t index) {
    if (index < filters_.size()) {
      filters_.erase(filters_.begin() + index);
    }
  }

  void clear_filters() { filters_.clear(); }

  void enable_filter(size_t index, bool enabled) {
    if (index < filters_.size()) {
      filters_[index].enabled = enabled;
    }
  }

  template <typename T>
  std::vector<T> apply_filters(const std::vector<T> &data) const {
    std::vector<T> result;

    for (const auto &item : data) {
      if (matches_all_filters(item)) {
        result.push_back(item);
      }
    }

    return result;
  }

  const std::vector<FilterCriteria> &get_filters() const { return filters_; }

private:
  std::vector<FilterCriteria> filters_;

  template <typename T> bool matches_all_filters(const T &item) const {
    for (const auto &filter : filters_) {
      if (!filter.enabled)
        continue;

      if (!matches_filter(item, filter)) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  bool matches_filter(const T &item, const FilterCriteria &filter) const {
    // Get field value from item (implementation would depend on data structure)
    std::string field_value = get_field_value(item, filter.field_name);

    switch (filter.type) {
    case FilterType::Text:
      return matches_text_filter(field_value, filter);
    case FilterType::Numeric:
      return matches_numeric_filter(field_value, filter);
    case FilterType::Date:
      return matches_date_filter(field_value, filter);
    case FilterType::Boolean:
      return matches_boolean_filter(field_value, filter);
    case FilterType::Enum:
      return matches_enum_filter(field_value, filter);
    }

    return false;
  }

  template <typename T>
  std::string get_field_value(const T &item,
                              const std::string &field_name) const {
    // Implementation would extract field value based on field name
    // This is a placeholder
    return "";
  }

  bool matches_text_filter(const std::string &value,
                           const FilterCriteria &filter) const {
    std::string filter_value = filter.value;
    std::string test_value = value;

    if (!filter.case_sensitive) {
      std::transform(filter_value.begin(), filter_value.end(),
                     filter_value.begin(), ::tolower);
      std::transform(test_value.begin(), test_value.end(), test_value.begin(),
                     ::tolower);
    }

    switch (filter.operator_) {
    case ComparisonOperator::Equal:
      return test_value == filter_value;
    case ComparisonOperator::NotEqual:
      return test_value != filter_value;
    case ComparisonOperator::Contains:
      return test_value.find(filter_value) != std::string::npos;
    case ComparisonOperator::StartsWith:
      return test_value.substr(0, filter_value.length()) == filter_value;
    case ComparisonOperator::EndsWith:
      return test_value.length() >= filter_value.length() &&
             test_value.substr(test_value.length() - filter_value.length()) ==
                 filter_value;
    case ComparisonOperator::Regex:
      try {
        std::regex pattern(filter_value);
        return std::regex_search(test_value, pattern);
      } catch (const std::regex_error &) {
        return false;
      }
    default:
      return false;
    }
  }

  bool matches_numeric_filter(const std::string &value,
                              const FilterCriteria &filter) const {
    try {
      double test_value = std::stod(value);
      double filter_value = std::stod(filter.value);

      switch (filter.operator_) {
      case ComparisonOperator::Equal:
        return std::abs(test_value - filter_value) < 1e-9;
      case ComparisonOperator::NotEqual:
        return std::abs(test_value - filter_value) >= 1e-9;
      case ComparisonOperator::Greater:
        return test_value > filter_value;
      case ComparisonOperator::GreaterEqual:
        return test_value >= filter_value;
      case ComparisonOperator::Less:
        return test_value < filter_value;
      case ComparisonOperator::LessEqual:
        return test_value <= filter_value;
      default:
        return false;
      }
    } catch (const std::exception &) {
      return false;
    }
  }

  bool matches_date_filter(const std::string &value,
                           const FilterCriteria &filter) const {
    // Implementation for date comparison
    return false;
  }

  bool matches_boolean_filter(const std::string &value,
                              const FilterCriteria &filter) const {
    bool test_value = (value == "true" || value == "1" || value == "yes");
    bool filter_value = (filter.value == "true" || filter.value == "1" ||
                         filter.value == "yes");

    switch (filter.operator_) {
    case ComparisonOperator::Equal:
      return test_value == filter_value;
    case ComparisonOperator::NotEqual:
      return test_value != filter_value;
    default:
      return false;
    }
  }

  bool matches_enum_filter(const std::string &value,
                           const FilterCriteria &filter) const {
    return matches_text_filter(value, filter);
  }
};

class SearchEngine {
public:
  struct SearchResult {
    std::string type;
    std::string title;
    std::string description;
    std::string data;
    float relevance_score;
    std::vector<std::pair<size_t, size_t>> highlight_ranges;
  };

  SearchEngine() = default;

  std::vector<SearchResult> search(const std::string &query,
                                   size_t max_results = 50) {
    std::vector<SearchResult> results;

    // Search symbols
    auto symbol_results = search_symbols(query);
    results.insert(results.end(), symbol_results.begin(), symbol_results.end());

    // Search indicators
    auto indicator_results = search_indicators(query);
    results.insert(results.end(), indicator_results.begin(),
                   indicator_results.end());

    // Search layouts
    auto layout_results = search_layouts(query);
    results.insert(results.end(), layout_results.begin(), layout_results.end());

    // Search help content
    auto help_results = search_help(query);
    results.insert(results.end(), help_results.begin(), help_results.end());

    // Sort by relevance score
    std::sort(results.begin(), results.end(),
              [](const SearchResult &a, const SearchResult &b) {
                return a.relevance_score > b.relevance_score;
              });

    // Limit results
    if (results.size() > max_results) {
      results.resize(max_results);
    }

    return results;
  }

  void index_symbol(const std::string &symbol, const std::string &description) {
    symbol_index_[symbol] = description;
  }

  void index_indicator(const std::string &name,
                       const std::string &description) {
    indicator_index_[name] = description;
  }

  void index_layout(const std::string &name, const std::string &description) {
    layout_index_[name] = description;
  }

private:
  std::unordered_map<std::string, std::string> symbol_index_;
  std::unordered_map<std::string, std::string> indicator_index_;
  std::unordered_map<std::string, std::string> layout_index_;

  std::vector<SearchResult> search_symbols(const std::string &query) {
    std::vector<SearchResult> results;

    for (const auto &pair : symbol_index_) {
      float score = calculate_relevance_score(query, pair.first, pair.second);
      if (score > 0.1f) {
        SearchResult result;
        result.type = "Symbol";
        result.title = pair.first;
        result.description = pair.second;
        result.data = pair.first;
        result.relevance_score = score;
        result.highlight_ranges =
            find_highlight_ranges(query, pair.first + " " + pair.second);
        results.push_back(result);
      }
    }

    return results;
  }

  std::vector<SearchResult> search_indicators(const std::string &query) {
    std::vector<SearchResult> results;

    for (const auto &pair : indicator_index_) {
      float score = calculate_relevance_score(query, pair.first, pair.second);
      if (score > 0.1f) {
        SearchResult result;
        result.type = "Indicator";
        result.title = pair.first;
        result.description = pair.second;
        result.data = pair.first;
        result.relevance_score = score;
        result.highlight_ranges =
            find_highlight_ranges(query, pair.first + " " + pair.second);
        results.push_back(result);
      }
    }

    return results;
  }

  std::vector<SearchResult> search_layouts(const std::string &query) {
    std::vector<SearchResult> results;

    for (const auto &pair : layout_index_) {
      float score = calculate_relevance_score(query, pair.first, pair.second);
      if (score > 0.1f) {
        SearchResult result;
        result.type = "Layout";
        result.title = pair.first;
        result.description = pair.second;
        result.data = pair.first;
        result.relevance_score = score;
        result.highlight_ranges =
            find_highlight_ranges(query, pair.first + " " + pair.second);
        results.push_back(result);
      }
    }

    return results;
  }

  std::vector<SearchResult> search_help(const std::string &query) {
    std::vector<SearchResult> results;

    // Search help content (implementation would search help database)

    return results;
  }

  float calculate_relevance_score(const std::string &query,
                                  const std::string &title,
                                  const std::string &description) {
    std::string combined = title + " " + description;
    std::transform(combined.begin(), combined.end(), combined.begin(),
                   ::tolower);

    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(),
                   ::tolower);

    float score = 0.0f;

    // Exact match in title gets highest score
    if (title.find(query) != std::string::npos) {
      score += 1.0f;
    }

    // Partial match in title
    if (combined.find(lower_query) != std::string::npos) {
      score += 0.5f;
    }

    // Word matches
    std::istringstream query_stream(lower_query);
    std::string word;
    while (query_stream >> word) {
      if (combined.find(word) != std::string::npos) {
        score += 0.2f;
      }
    }

    return score;
  }

  std::vector<std::pair<size_t, size_t>>
  find_highlight_ranges(const std::string &query, const std::string &text) {
    std::vector<std::pair<size_t, size_t>> ranges;

    std::string lower_text = text;
    std::string lower_query = query;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                   ::tolower);
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(),
                   ::tolower);

    size_t pos = 0;
    while ((pos = lower_text.find(lower_query, pos)) != std::string::npos) {
      ranges.emplace_back(pos, pos + lower_query.length());
      pos += lower_query.length();
    }

    return ranges;
  }
};

// ============================================================================
// Theme Management System
// ============================================================================

class ThemeManager {
public:
  struct ColorScheme {
    std::string name;
    std::string description;
    DashboardTheme theme{};
    bool is_dark_theme;
    std::unordered_map<std::string, glm::vec4> custom_colors;
  };

  ThemeManager() {
    create_default_themes();
    current_theme_ = "Dark Professional";
  }

  void set_theme(const std::string &theme_name) {
    auto it = themes_.find(theme_name);
    if (it != themes_.end()) {
      current_theme_ = theme_name;
      apply_theme(it->second);
    }
  }

  const std::string &get_current_theme() const { return current_theme_; }

  std::vector<std::string> get_available_themes() const {
    std::vector<std::string> names;
    for (const auto &pair : themes_) {
      names.push_back(pair.first);
    }
    return names;
  }

  void create_custom_theme(const std::string &name,
                           const DashboardTheme &theme) {
    ColorScheme scheme;
    scheme.name = name;
    scheme.description = "Custom theme";
    scheme.theme = theme;
    scheme.is_dark_theme = false;

    themes_[name] = scheme;
    save_theme_to_file(scheme);
  }

  void customize_color(const std::string &color_name, const glm::vec4 &color) {
    auto it = themes_.find(current_theme_);
    if (it != themes_.end()) {
      it->second.custom_colors[color_name] = color;
      apply_custom_color(color_name, color);
    }
  }

  glm::vec4 get_color(const std::string &color_name) const {
    auto it = themes_.find(current_theme_);
    if (it != themes_.end()) {
      auto custom_it = it->second.custom_colors.find(color_name);
      if (custom_it != it->second.custom_colors.end()) {
        return custom_it->second;
      }

      // Return default theme color
      return get_theme_color(it->second.theme, color_name);
    }

    return glm::vec4(1.0f); // Default white
  }

private:
  std::unordered_map<std::string, ColorScheme> themes_;
  std::string current_theme_;

  void create_default_themes() {
    // Dark Professional Theme
    create_dark_professional_theme();

    // Light Professional Theme
    create_light_professional_theme();

    // High Contrast Theme
    create_high_contrast_theme();

    // Bloomberg-style Theme
    create_bloomberg_theme();

    // Matrix Theme
    create_matrix_theme();
  }

  void create_dark_professional_theme() {
    ColorScheme scheme;
    scheme.name = "Dark Professional";
    scheme.description = "Professional dark theme for trading";
    scheme.is_dark_theme = true;

    scheme.theme.background_primary = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
    scheme.theme.background_secondary = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);
    scheme.theme.background_panel = glm::vec4(0.12f, 0.12f, 0.12f, 1.0f);

    scheme.theme.text_primary = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
    scheme.theme.text_secondary = glm::vec4(0.7f, 0.7f, 0.7f, 1.0f);
    scheme.theme.text_muted = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

    scheme.theme.price_up = glm::vec4(0.0f, 0.8f, 0.0f, 1.0f);
    scheme.theme.price_down = glm::vec4(0.8f, 0.0f, 0.0f, 1.0f);
    scheme.theme.price_neutral = glm::vec4(0.6f, 0.6f, 0.6f, 1.0f);

    scheme.theme.accent_primary = glm::vec4(0.2f, 0.6f, 1.0f, 1.0f);
    scheme.theme.accent_secondary = glm::vec4(0.8f, 0.4f, 0.0f, 1.0f);
    scheme.theme.border_color = glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);

    scheme.theme.status_connected = glm::vec4(0.0f, 0.8f, 0.0f, 1.0f);
    scheme.theme.status_disconnected = glm::vec4(0.8f, 0.0f, 0.0f, 1.0f);
    scheme.theme.status_warning = glm::vec4(0.8f, 0.8f, 0.0f, 1.0f);

    themes_[scheme.name] = scheme;
  }

  void create_light_professional_theme() {
    ColorScheme scheme;
    scheme.name = "Light Professional";
    scheme.description = "Professional light theme for trading";
    scheme.is_dark_theme = false;

    scheme.theme.background_primary = glm::vec4(0.95f, 0.95f, 0.95f, 1.0f);
    scheme.theme.background_secondary = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
    scheme.theme.background_panel = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

    scheme.theme.text_primary = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
    scheme.theme.text_secondary = glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);
    scheme.theme.text_muted = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

    scheme.theme.price_up = glm::vec4(0.0f, 0.6f, 0.0f, 1.0f);
    scheme.theme.price_down = glm::vec4(0.8f, 0.0f, 0.0f, 1.0f);
    scheme.theme.price_neutral = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);

    scheme.theme.accent_primary = glm::vec4(0.0f, 0.4f, 0.8f, 1.0f);
    scheme.theme.accent_secondary = glm::vec4(0.6f, 0.3f, 0.0f, 1.0f);
    scheme.theme.border_color = glm::vec4(0.7f, 0.7f, 0.7f, 1.0f);

    scheme.theme.status_connected = glm::vec4(0.0f, 0.6f, 0.0f, 1.0f);
    scheme.theme.status_disconnected = glm::vec4(0.8f, 0.0f, 0.0f, 1.0f);
    scheme.theme.status_warning = glm::vec4(0.8f, 0.6f, 0.0f, 1.0f);

    themes_[scheme.name] = scheme;
  }

  void create_high_contrast_theme() {
    ColorScheme scheme;
    scheme.name = "High Contrast";
    scheme.description = "High contrast theme for accessibility";
    scheme.is_dark_theme = true;

    scheme.theme.background_primary = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.background_secondary = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
    scheme.theme.background_panel = glm::vec4(0.05f, 0.05f, 0.05f, 1.0f);

    scheme.theme.text_primary = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    scheme.theme.text_secondary = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
    scheme.theme.text_muted = glm::vec4(0.7f, 0.7f, 0.7f, 1.0f);

    scheme.theme.price_up = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    scheme.theme.price_down = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.price_neutral = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);

    scheme.theme.accent_primary = glm::vec4(0.0f, 0.8f, 1.0f, 1.0f);
    scheme.theme.accent_secondary = glm::vec4(1.0f, 0.5f, 0.0f, 1.0f);
    scheme.theme.border_color = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

    scheme.theme.status_connected = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    scheme.theme.status_disconnected = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.status_warning = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);

    themes_[scheme.name] = scheme;
  }

  void create_bloomberg_theme() {
    ColorScheme scheme;
    scheme.name = "Bloomberg";
    scheme.description = "Bloomberg Terminal inspired theme";
    scheme.is_dark_theme = true;

    scheme.theme.background_primary = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.background_secondary = glm::vec4(0.05f, 0.05f, 0.1f, 1.0f);
    scheme.theme.background_panel = glm::vec4(0.0f, 0.0f, 0.05f, 1.0f);

    scheme.theme.text_primary =
        glm::vec4(1.0f, 0.6f, 0.0f, 1.0f); // Orange text
    scheme.theme.text_secondary =
        glm::vec4(0.8f, 0.8f, 0.0f, 1.0f); // Yellow text
    scheme.theme.text_muted = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

    scheme.theme.price_up = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    scheme.theme.price_down = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.price_neutral = glm::vec4(1.0f, 0.6f, 0.0f, 1.0f);

    scheme.theme.accent_primary = glm::vec4(1.0f, 0.6f, 0.0f, 1.0f);
    scheme.theme.accent_secondary = glm::vec4(0.8f, 0.8f, 0.0f, 1.0f);
    scheme.theme.border_color = glm::vec4(0.3f, 0.3f, 0.0f, 1.0f);

    scheme.theme.status_connected = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    scheme.theme.status_disconnected = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.status_warning = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);

    themes_[scheme.name] = scheme;
  }

  void create_matrix_theme() {
    ColorScheme scheme;
    scheme.name = "Matrix";
    scheme.description = "Matrix-inspired green theme";
    scheme.is_dark_theme = true;

    scheme.theme.background_primary = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.background_secondary = glm::vec4(0.0f, 0.05f, 0.0f, 1.0f);
    scheme.theme.background_panel = glm::vec4(0.0f, 0.02f, 0.0f, 1.0f);

    scheme.theme.text_primary = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    scheme.theme.text_secondary = glm::vec4(0.0f, 0.8f, 0.0f, 1.0f);
    scheme.theme.text_muted = glm::vec4(0.0f, 0.5f, 0.0f, 1.0f);

    scheme.theme.price_up = glm::vec4(0.0f, 1.0f, 0.5f, 1.0f);
    scheme.theme.price_down = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.price_neutral = glm::vec4(0.0f, 0.8f, 0.0f, 1.0f);

    scheme.theme.accent_primary = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    scheme.theme.accent_secondary = glm::vec4(0.0f, 0.8f, 0.8f, 1.0f);
    scheme.theme.border_color = glm::vec4(0.0f, 0.3f, 0.0f, 1.0f);

    scheme.theme.status_connected = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    scheme.theme.status_disconnected = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    scheme.theme.status_warning = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);

    themes_[scheme.name] = scheme;
  }

  void apply_theme(const ColorScheme &scheme) {
    // Apply theme to all UI components
    // This would update the global theme and notify all components
  }

  void apply_custom_color(const std::string &color_name,
                          const glm::vec4 &color) {
    // Apply custom color override
  }

  glm::vec4 get_theme_color(const DashboardTheme &theme,
                            const std::string &color_name) const {
    // Map color name to theme color
    if (color_name == "background_primary")
      return theme.background_primary;
    if (color_name == "background_secondary")
      return theme.background_secondary;
    if (color_name == "background_panel")
      return theme.background_panel;
    if (color_name == "text_primary")
      return theme.text_primary;
    if (color_name == "text_secondary")
      return theme.text_secondary;
    if (color_name == "text_muted")
      return theme.text_muted;
    if (color_name == "price_up")
      return theme.price_up;
    if (color_name == "price_down")
      return theme.price_down;
    if (color_name == "price_neutral")
      return theme.price_neutral;
    if (color_name == "accent_primary")
      return theme.accent_primary;
    if (color_name == "accent_secondary")
      return theme.accent_secondary;
    if (color_name == "border_color")
      return theme.border_color;
    if (color_name == "status_connected")
      return theme.status_connected;
    if (color_name == "status_disconnected")
      return theme.status_disconnected;
    if (color_name == "status_warning")
      return theme.status_warning;

    return glm::vec4(1.0f); // Default
  }

  void save_theme_to_file(const ColorScheme &scheme) {
    // Save theme to JSON file
    std::string filename = "themes/" + scheme.name + ".json";

    Json::Value root;
    root["name"] = scheme.name;
    root["description"] = scheme.description;
    root["is_dark_theme"] = scheme.is_dark_theme;

    // Save theme colors
    Json::Value theme_json;
    theme_json["background_primary"] =
        color_to_json(scheme.theme.background_primary);
    theme_json["background_secondary"] =
        color_to_json(scheme.theme.background_secondary);
    theme_json["background_panel"] =
        color_to_json(scheme.theme.background_panel);
    theme_json["text_primary"] = color_to_json(scheme.theme.text_primary);
    theme_json["text_secondary"] = color_to_json(scheme.theme.text_secondary);
    theme_json["text_muted"] = color_to_json(scheme.theme.text_muted);
    theme_json["price_up"] = color_to_json(scheme.theme.price_up);
    theme_json["price_down"] = color_to_json(scheme.theme.price_down);
    theme_json["price_neutral"] = color_to_json(scheme.theme.price_neutral);
    theme_json["accent_primary"] = color_to_json(scheme.theme.accent_primary);
    theme_json["accent_secondary"] =
        color_to_json(scheme.theme.accent_secondary);
    theme_json["border_color"] = color_to_json(scheme.theme.border_color);
    theme_json["status_connected"] =
        color_to_json(scheme.theme.status_connected);
    theme_json["status_disconnected"] =
        color_to_json(scheme.theme.status_disconnected);
    theme_json["status_warning"] = color_to_json(scheme.theme.status_warning);

    root["theme"] = theme_json;

    // Save custom colors
    Json::Value custom_colors_json;
    for (const auto &pair : scheme.custom_colors) {
      custom_colors_json[pair.first] = color_to_json(pair.second);
    }
    root["custom_colors"] = custom_colors_json;

    std::ofstream file(filename);
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &file);
  }

  Json::Value color_to_json(const glm::vec4 &color) {
    Json::Value json_color;
    json_color["r"] = color.r;
    json_color["g"] = color.g;
    json_color["b"] = color.b;
    json_color["a"] = color.a;
    return json_color;
  }
};

} // namespace BTQuant