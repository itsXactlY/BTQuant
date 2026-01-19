#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../vulkan_dashboard_advanced.hpp"
#include "imgui.h"
#include "implot.h"
#include <memory>

namespace BTQuant {

class QuantWorkspaceComponent : public UIComponent {
public:
  explicit QuantWorkspaceComponent(std::shared_ptr<HotSpineDataBridge> bridge);
  virtual ~QuantWorkspaceComponent() = default;

  void update(float dt) override;
  void render_gui() override;

  void initialize_vulkan_resources(VulkanCore *core) override {}
  void clear_data() override {}

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  void render_instrument_chart(const std::string &symbol,
                               const InstrumentData &inst);
};

} // namespace BTQuant
