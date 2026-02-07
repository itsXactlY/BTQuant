#include "layout/layout_presets.hpp"

#include <filesystem>
#include <fstream>

namespace BTQuant {
namespace Layout {

LayoutPresetManager::LayoutPresetManager() {
  initialize_presets_directory();
  load_presets();
}

LayoutPresetManager::~LayoutPresetManager() = default;

void LayoutPresetManager::load_presets() {
  presets_.clear();
  load_builtin_presets();
  load_user_presets();
}

bool LayoutPresetManager::save_preset(const std::string& name, const std::string& description,
                                      const std::string& json_data, const std::string& category) {
  LayoutPreset preset;
  preset.name = name;
  preset.description = description;
  preset.json_data = json_data;
  preset.category = category;
  preset.is_builtin = false;
  preset.author = "User";
  preset.version = "1.0";

  if (!validate_preset(preset)) {
    return false;
  }

  // Check if preset already exists and update it
  for (auto& existing : presets_) {
    if (existing.name == name) {
      existing = preset;
      return save_preset_to_file(preset);
    }
  }

  presets_.push_back(preset);
  return save_preset_to_file(preset);
}

bool LayoutPresetManager::apply_preset(const std::string& preset_name) {
  for (const auto& preset : presets_) {
    if (preset.name == preset_name) {
      // In a real implementation, this would apply the json_data to the layout
      return true;
    }
  }
  return false;
}

bool LayoutPresetManager::delete_preset(const std::string& preset_name) {
  for (auto it = presets_.begin(); it != presets_.end(); ++it) {
    if (it->name == preset_name && !it->is_builtin) {
      // Delete the file
      std::string file_path = get_preset_file_path(preset_name);
      std::filesystem::remove(file_path);
      presets_.erase(it);
      return true;
    }
  }
  return false;
}

std::vector<LayoutPreset> LayoutPresetManager::get_all_presets() const { return presets_; }

std::vector<LayoutPreset> LayoutPresetManager::get_presets_by_category(
    const std::string& category) const {
  std::vector<LayoutPreset> result;
  for (const auto& preset : presets_) {
    if (preset.category == category) {
      result.push_back(preset);
    }
  }
  return result;
}

bool LayoutPresetManager::export_preset(const std::string& preset_name,
                                        const std::string& file_path) {
  for (const auto& preset : presets_) {
    if (preset.name == preset_name) {
      std::ofstream file(file_path);
      if (file.is_open()) {
        file << preset.json_data;
        return true;
      }
    }
  }
  return false;
}

bool LayoutPresetManager::import_preset(const std::string& file_path) {
  auto preset = load_preset_from_file(file_path);
  if (!preset.name.empty()) {
    presets_.push_back(preset);
    return true;
  }
  return false;
}

bool LayoutPresetManager::create_thumbnail(const std::string& /*preset_name*/,
                                           const std::string& /*thumbnail_path*/) {
  // Thumbnail creation would be implemented here
  return true;
}

bool LayoutPresetManager::validate_preset(const LayoutPreset& preset) const {
  return !preset.name.empty() && !preset.json_data.empty();
}

void LayoutPresetManager::initialize_presets_directory() {
  presets_directory_ = "./presets";
  std::filesystem::create_directories(presets_directory_);
}

void LayoutPresetManager::load_builtin_presets() {
  // Scalper layout (DOM + Tape)
  LayoutPreset scalper;
  scalper.name = "Scalper";
  scalper.description = "Optimized layout for scalping with DOM and Time & Sales";
  scalper.category = "Trading";
  scalper.json_data = R"({
    "grid": {
        "columns": 6,
        "rows": 10
    },
    "panels": [
        {
            "grid_height": 4,
            "grid_width": 3,
            "grid_x": 0,
            "grid_y": 0,
            "position": [
                0.0,
                30.0
            ],
            "size": [
                960.0,
                420.0
            ],
            "symbol": "",
            "title": "DOM Ladder",
            "type": 2,
            "visible": true,
            "settings": {
                "auto_scale_price": true,
                "enable_fade_out": false,
                "heatmap_intensity": 1.0,
                "large_order_threshold": 10.0,
                "max_large_order_markers": 100,
                "persistence_threshold_ms": 5000,
                "persistence_timeout_ms": 30000.0,
                "price_bins": 100,
                "price_range": 0.02,
                "show_persistent_lines": true,
                "symbol_id": 0
            }
        },
        {
            "grid_height": 4,
            "grid_width": 3,
            "grid_x": 3,
            "grid_y": 0,
            "position": [
                960.0,
                30.0
            ],
            "size": [
                960.0,
                420.0
            ],
            "symbol": "",
            "title": "Time & Sales",
            "type": 13,
            "visible": true
        },
        {
            "grid_height": 3,
            "grid_width": 4,
            "grid_x": 0,
            "grid_y": 4,
            "position": [
                0.0,
                465.0
            ],
            "size": [
                1280.0,
                315.0
            ],
            "symbol": "",
            "title": "Price Chart",
            "type": 0,
            "visible": true
        },
        {
            "grid_height": 3,
            "grid_width": 2,
            "grid_x": 4,
            "grid_y": 4,
            "position": [
                1280.0,
                465.0
            ],
            "size": [
                640.0,
                315.0
            ],
            "symbol": "",
            "title": "Order Book",
            "type": 10,
            "visible": true
        },
        {
            "grid_height": 1,
            "grid_width": 2,
            "grid_x": 0,
            "grid_y": 7,
            "position": [
                0.0,
                780.0
            ],
            "size": [
                640.0,
                105.0
            ],
            "symbol": "",
            "title": "Active Orders",
            "type": 6,
            "visible": true
        },
        {
            "grid_height": 1,
            "grid_width": 2,
            "grid_x": 2,
            "grid_y": 7,
            "position": [
                640.0,
                780.0
            ],
            "size": [
                640.0,
                105.0
            ],
            "symbol": "",
            "title": "Positions",
            "type": 7,
            "visible": true
        },
        {
            "grid_height": 1,
            "grid_width": 2,
            "grid_x": 4,
            "grid_y": 7,
            "position": [
                1280.0,
                780.0
            ],
            "size": [
                640.0,
                105.0
            ],
            "symbol": "",
            "title": "Watchlist",
            "type": 11,
            "visible": true
        }
    ]
})";
  scalper.is_builtin = true;
  scalper.author = "BTQuant";
  scalper.version = "1.0";
  presets_.push_back(scalper);

  // Analyst layout (Charts)
  LayoutPreset analyst;
  analyst.name = "Analyst";
  analyst.description = "Optimized layout for market analysis with multiple charts";
  analyst.category = "Analysis";
  analyst.json_data = R"({
    "grid": {
        "columns": 6,
        "rows": 10
    },
    "panels": [
        {
            "grid_height": 5,
            "grid_width": 4,
            "grid_x": 0,
            "grid_y": 0,
            "position": [
                0.0,
                30.0
            ],
            "size": [
                1280.0,
                525.0
            ],
            "symbol": "",
            "title": "Main Price Chart",
            "type": 0,
            "visible": true,
            "settings": {
                "chart_style": "candlestick",
                "timeframe": "1m",
                "indicators": ["ema", "rsi", "macd"]
            }
        },
        {
            "grid_height": 5,
            "grid_width": 2,
            "grid_x": 4,
            "grid_y": 0,
            "position": [
                1280.0,
                30.0
            ],
            "size": [
                640.0,
                525.0
            ],
            "symbol": "",
            "title": "Market Depth",
            "type": 10,
            "visible": true
        },
        {
            "grid_height": 2,
            "grid_width": 2,
            "grid_x": 0,
            "grid_y": 5,
            "position": [
                0.0,
                555.0
            ],
            "size": [
                640.0,
                210.0
            ],
            "symbol": "",
            "title": "Volume Profile",
            "type": 14,
            "visible": true
        },
        {
            "grid_height": 2,
            "grid_width": 2,
            "grid_x": 2,
            "grid_y": 5,
            "position": [
                640.0,
                555.0
            ],
            "size": [
                640.0,
                210.0
            ],
            "symbol": "",
            "title": "TPO Profile",
            "type": 19,
            "visible": true,
            "settings": {
                "show_grid": true,
                "show_heatmap": true,
                "show_text": true,
                "symbol_id": 0,
                "time_window": 30.0
            }
        },
        {
            "grid_height": 2,
            "grid_width": 2,
            "grid_x": 4,
            "grid_y": 5,
            "position": [
                1280.0,
                555.0
            ],
            "size": [
                640.0,
                210.0
            ],
            "symbol": "",
            "title": "Footprint Chart",
            "type": 18,
            "visible": true
        },
        {
            "grid_height": 1,
            "grid_width": 3,
            "grid_x": 0,
            "grid_y": 7,
            "position": [
                0.0,
                765.0
            ],
            "size": [
                960.0,
                105.0
            ],
            "symbol": "",
            "title": "Market Statistics",
            "type": 20,
            "visible": true
        },
        {
            "grid_height": 1,
            "grid_width": 3,
            "grid_x": 3,
            "grid_y": 7,
            "position": [
                960.0,
                765.0
            ],
            "size": [
                960.0,
                105.0
            ],
            "symbol": "",
            "title": "Watchlist",
            "type": 11,
            "visible": true
        },
        {
            "grid_height": 2,
            "grid_width": 6,
            "grid_x": 0,
            "grid_y": 8,
            "position": [
                0.0,
                870.0
            ],
            "size": [
                1920.0,
                210.0
            ],
            "symbol": "",
            "title": "Multi-Timeframe Analysis",
            "type": 0,
            "visible": true,
            "settings": {
                "chart_style": "line",
                "timeframe": "5m",
                "indicators": ["sma", "bollinger_bands"]
            }
        }
    ]
})";
  analyst.is_builtin = true;
  analyst.author = "BTQuant";
  analyst.version = "1.0";
  presets_.push_back(analyst);

  // Options layout (Desk + Risk)
  LayoutPreset options;
  options.name = "Options";
  options.description = "Optimized layout for options trading with desk and risk management";
  options.category = "Options";
  options.json_data = R"({
    "grid": {
        "columns": 6,
        "rows": 10
    },
    "panels": [
        {
            "grid_height": 3,
            "grid_width": 4,
            "grid_x": 0,
            "grid_y": 0,
            "position": [
                0.0,
                30.0
            ],
            "size": [
                1280.0,
                315.0
            ],
            "symbol": "",
            "title": "Options Chain",
            "type": 15,
            "visible": true,
            "settings": {
                "display_mode": "greeks",
                "highlight_atm": true,
                "show_volume": true,
                "symbol_id": 0
            }
        },
        {
            "grid_height": 3,
            "grid_width": 2,
            "grid_x": 4,
            "grid_y": 0,
            "position": [
                1280.0,
                30.0
            ],
            "size": [
                640.0,
                315.0
            ],
            "symbol": "",
            "title": "Greeks Monitor",
            "type": 16,
            "visible": true,
            "settings": {
                "show_delta": true,
                "show_gamma": true,
                "show_theta": true,
                "show_vega": true,
                "show_rho": true
            }
        },
        {
            "grid_height": 4,
            "grid_width": 3,
            "grid_x": 0,
            "grid_y": 3,
            "position": [
                0.0,
                345.0
            ],
            "size": [
                960.0,
                420.0
            ],
            "symbol": "",
            "title": "Options Desk",
            "type": 17,
            "visible": true,
            "settings": {
                "order_entry_mode": "options",
                "show_quotes": true,
                "show_orders": true,
                "symbol_id": 0
            }
        },
        {
            "grid_height": 4,
            "grid_width": 3,
            "grid_x": 3,
            "grid_y": 3,
            "position": [
                960.0,
                345.0
            ],
            "size": [
                960.0,
                420.0
            ],
            "symbol": "",
            "title": "Risk Matrix",
            "type": 21,
            "visible": true,
            "settings": {
                "risk_model": "monte_carlo",
                "show_pnl": true,
                "show_exposure": true,
                "show_correlation": true
            }
        },
        {
            "grid_height": 2,
            "grid_width": 2,
            "grid_x": 0,
            "grid_y": 7,
            "position": [
                0.0,
                765.0
            ],
            "size": [
                640.0,
                210.0
            ],
            "symbol": "",
            "title": "Positions",
            "type": 7,
            "visible": true
        },
        {
            "grid_height": 2,
            "grid_width": 2,
            "grid_x": 2,
            "grid_y": 7,
            "position": [
                640.0,
                765.0
            ],
            "size": [
                640.0,
                210.0
            ],
            "symbol": "",
            "title": "Active Orders",
            "type": 6,
            "visible": true
        },
        {
            "grid_height": 2,
            "grid_width": 2,
            "grid_x": 4,
            "grid_y": 7,
            "position": [
                1280.0,
                765.0
            ],
            "size": [
                640.0,
                210.0
            ],
            "symbol": "",
            "title": "Risk Summary",
            "type": 22,
            "visible": true,
            "settings": {
                "show_var": true,
                "show_max_loss": true,
                "show_margin_usage": true
            }
        },
        {
            "grid_height": 1,
            "grid_width": 6,
            "grid_x": 0,
            "grid_y": 9,
            "position": [
                0.0,
                975.0
            ],
            "size": [
                1920.0,
                105.0
            ],
            "symbol": "",
            "title": "Options Strategy Builder",
            "type": 23,
            "visible": true
        }
    ]
})";
  options.is_builtin = true;
  options.author = "BTQuant";
  options.version = "1.0";
  presets_.push_back(options);

  // Default trading layout (keeping for compatibility)
  LayoutPreset trading;
  trading.name = "Trading Default";
  trading.description = "Default layout for trading operations";
  trading.category = "Trading";
  trading.json_data = "{}";
  trading.is_builtin = true;
  trading.author = "BTQuant";
  trading.version = "1.0";
  presets_.push_back(trading);

  // Default analysis layout (keeping for compatibility)
  LayoutPreset analysis;
  analysis.name = "Analysis Default";
  analysis.description = "Default layout for market analysis";
  analysis.category = "Analysis";
  analysis.json_data = "{}";
  analysis.is_builtin = true;
  analysis.author = "BTQuant";
  analysis.version = "1.0";
  presets_.push_back(analysis);
}

void LayoutPresetManager::load_user_presets() {
  if (!std::filesystem::exists(presets_directory_)) {
    return;
  }

  for (const auto& entry : std::filesystem::directory_iterator(presets_directory_)) {
    if (entry.path().extension() == ".json") {
      auto preset = load_preset_from_file(entry.path().string());
      if (!preset.name.empty()) {
        presets_.push_back(preset);
      }
    }
  }
}

LayoutPreset LayoutPresetManager::load_preset_from_file(const std::string& file_path) {
  LayoutPreset preset;
  std::ifstream file(file_path);
  if (file.is_open()) {
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    preset.json_data = content;
    preset.name = std::filesystem::path(file_path).stem().string();
    preset.category = "Custom";
    preset.is_builtin = false;
  }
  return preset;
}

bool LayoutPresetManager::save_preset_to_file(const LayoutPreset& preset) {
  std::string file_path = get_preset_file_path(preset.name);
  std::ofstream file(file_path);
  if (file.is_open()) {
    file << preset.json_data;
    return true;
  }
  return false;
}

std::string LayoutPresetManager::get_preset_file_path(const std::string& preset_name) const {
  return presets_directory_ + "/" + preset_name + ".json";
}

}  // namespace Layout
}  // namespace BTQuant
