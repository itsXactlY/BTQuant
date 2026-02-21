#pragma once

#include <atomic>
#include <memory>

#include "market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class DomSurfacePanel : public PanelBase {
 public:
  DomSurfacePanel(const PanelConfig& config,
                  std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  ~DomSurfacePanel() override;

  void render_content() override;
  void render_panel_header() override;
  void setSymbol(uint32_t symbol_id);
  void updateLivePrice(double price);

  uint32_t get_symbol_id() const { return current_symbol_id_; }

 private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  uint32_t current_symbol_id_ = 0;
  std::atomic<double> live_price_{0.0};
  int max_levels_ = 20;
};

}  // namespace BTQuant
