#pragma once

#include <atomic>
#include <memory>

#include "market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class DomSurfacePanel : public PanelBase {
 public:
  explicit DomSurfacePanel(std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  ~DomSurfacePanel() override;

  void render_content() override;
  void setSymbol(uint32_t symbol_id);
  void render_panel_header() override;
  void updateLivePrice(double price);

  uint32_t get_symbol_id() const { return current_symbol_id_; }

 private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  uint32_t current_symbol_id_ = 0;
  std::atomic<double> live_price_{0.0};

  void onDataUpdate(uint32_t symbol_id, RenderEngine::NotificationType type);
};

}  // namespace BTQuant
