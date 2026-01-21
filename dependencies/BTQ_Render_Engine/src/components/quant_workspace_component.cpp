#include "../../include/components/quant_workspace_component.hpp"

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : UIComponent({0, 0}, {0, 0}), bridge_(bridge), processor_(processor) {

  // Ensure ImPlot context is created (must be called once)
  static bool implot_init = false;
  if (!implot_init) {
    ImPlot::CreateContext();
    implot_init = true;
  }

  // Initialize Trading Systems
  order_manager_ = std::make_shared<OrderManager>();
  position_manager_ = std::make_shared<PositionManager>();
  risk_assessment_ = std::make_shared<RiskAssessment>();

  // Set up callbacks for order execution -> position updates
  order_manager_->set_execution_callback(
      [this](const OrderManager::OrderExecution &execution) {
        position_manager_->update_position(execution);
      });

  // Initialize the new panel-based system
  panel_manager_ = std::make_unique<PanelManager>(
      bridge_, processor_, order_manager_, position_manager_, risk_assessment_);
  panel_manager_->initialize();
}

void QuantWorkspaceComponent::initialize_vulkan_resources(VulkanCore *core) {
  // Panel system handles its own Vulkan resources
  (void)core; // Suppress unused parameter warning
}

void QuantWorkspaceComponent::update(float dt) { panel_manager_->update(dt); }

void QuantWorkspaceComponent::render_gui() {
  // Render dashboard controls
  if (show_dashboard_controls_) {
    render_dashboard_controls();
  }

  // Render all panels through the panel manager
  panel_manager_->render();
}

void QuantWorkspaceComponent::render_dashboard_controls() {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Dashboard Controls", &show_dashboard_controls_)) {
    ImGui::Text("Ultra-Quantitative Dashboard");
    ImGui::Separator();

    // Panel management
    if (ImGui::CollapsingHeader("Add Panels", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Button("Add Chart Panel")) {
        panel_manager_->add_panel(PanelType::CHART);
      }
      ImGui::SameLine();
      if (ImGui::Button("Add Metrics Panel")) {
        panel_manager_->add_panel(PanelType::METRICS);
      }

      if (ImGui::Button("Add Heatmap Panel")) {
        panel_manager_->add_panel(PanelType::HEATMAP);
      }
      ImGui::SameLine();
      if (ImGui::Button("Add Histogram Panel")) {
        panel_manager_->add_panel(PanelType::HISTOGRAM);
      }
    }

    // Layout controls
    if (ImGui::CollapsingHeader("Layout")) {
      if (ImGui::Button("Auto Arrange")) {
        panel_manager_->auto_arrange_panels();
      }
    }

    // Performance info
    ImGui::Separator();
    ImGui::Text("Real-time Dashboard Active");
    ImGui::Text("Data Pipeline: Connected");
  }
  ImGui::End();
}

void QuantWorkspaceComponent::clear_data() { panel_manager_.reset(); }

} // namespace BTQuant
