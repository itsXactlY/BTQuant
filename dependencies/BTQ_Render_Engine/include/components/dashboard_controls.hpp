#pragma once

#include "../ui/ui_base.hpp"
#include "../vulkan_base_types.hpp"

#include <memory>
#include <string>

namespace BTQuant {

// Forward declarations
class PanelManager;

class DashboardControls : public UIComponent {
 public:
  explicit DashboardControls(PanelManager* panel_manager);
  virtual ~DashboardControls() = default;

  void update(float dt) override;
  void render_gui() override;
  void initialize_vulkan_resources(VulkanCore* core) override;
  void clear_data() override;

  // Specific method to render dashboard controls
  void render_dashboard_controls();

 private:
  PanelManager* panel_manager_;
};

}  // namespace BTQuant