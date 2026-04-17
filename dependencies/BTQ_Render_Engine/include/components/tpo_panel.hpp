#pragma once

#include <memory>

#include "../analytics/cluster_engine.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class TpoPanel : public PanelBase {
 public:
  TpoPanel(const PanelConfig& config);

  void update(float dt) override;
  void render() override;

  // Set the cluster engine to read TPO data from
  void set_cluster_engine(Analytics::ClusterEngine* engine) { cluster_engine_ = engine; }

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) {
    symbol_id_ = id;
    // Note: Exchange connection management has been moved out of the renderer
    // The renderer now only handles rendering, not data subscription
  }

  // TPO level data for each price
  struct TpoLevel {
    double price;
    uint16_t tpo_bits;
    int popcount;
  };

  // Value area and session key levels
  struct ValueArea {
    double poc_price = 0.0;
    double vah = 0.0;
    double val = 0.0;
    double session_high = 0.0;
    double session_low = 0.0;
    double ib_high = 0.0;
    double ib_low = 0.0;
  };

 private:
  uint32_t symbol_id_ = 0;
  Analytics::ClusterEngine* cluster_engine_ = nullptr;
};

}  // namespace BTQuant
