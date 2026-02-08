#pragma once

#include <memory>
#include <mutex>

#include "../symbol_registry.hpp"  // Include for SymbolRegistry
#include "MarketMicrostructureRenderer.h"
#include "analytics/tpoengine.h"
#include "panel_base.hpp"
#include "panel_manager.hpp"  // Include for PanelManager methods

namespace BTQuant {

class TpoPanel : public PanelBase {
 public:
  TpoPanel(const PanelConfig& config, RenderEngine::MarketMicrostructureRenderer* renderer);

  void update(float dt) override;
  void render() override;

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) {
    bool symbol_changed = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (symbol_id_ != id) {
        symbol_id_ = id;
        symbol_changed = true;
      }
    }

    if (symbol_changed) {
      if (renderer_) renderer_->setSymbol(id);

      // Notify the panel manager about the symbol change to trigger symbol linking
      auto symbol_info_opt = SymbolRegistry::instance().get_symbol_info(id);
      if (symbol_info_opt && get_panel_manager()) {
        get_panel_manager()->propagate_symbol_to_linked_panels(get_panel_id(),
                                                               symbol_info_opt->name);
      }
    }
  }

  // Getter methods for serialization
  bool get_show_text() const { return show_text_; }
  bool get_show_grid() const { return show_grid_; }
  bool get_show_heatmap() const { return show_heatmap_; }
  float get_time_window() const { return time_window_; }

  // Setter methods for deserialization
  void set_show_text(bool show) { show_text_ = show; }
  void set_show_grid(bool show) { show_grid_ = show; }
  void set_show_heatmap(bool show) { show_heatmap_ = show; }
  void set_time_window(float window) { time_window_ = window; }

 private:
  mutable std::mutex data_mutex_;
  RenderEngine::MarketMicrostructureRenderer* renderer_;
  uint32_t symbol_id_ = 0;

  // TPO Engine for processing market data
  TPOEngine tpo_engine_{0.25};  // Default price bucket size of 0.25

  // Track the last processed timestamp to avoid duplicate processing
  uint64_t last_processed_timestamp_ns_ = 0;

  // UI state variables that should be persisted
  bool show_text_ = true;
  bool show_grid_ = true;
  bool show_heatmap_ = true;
  float time_window_ = 30.0f;
};

}  // namespace BTQuant
