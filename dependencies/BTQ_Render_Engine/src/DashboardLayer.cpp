#include "imgui_internal.h"
#include "vulkan_dashboard_advanced.hpp"
#include <algorithm>

namespace BTQuant {

DashboardLayer::DashboardLayer(VulkanCore *core) : core_(core) {
  chart_renderer_ = std::make_unique<OffscreenChartRenderer>(core);
  // Note: RenderPass is created inside chart_renderer after first resize or
  // manually For now we assume a default pass or let chart_renderer handle it
}

void DashboardLayer::OnUpdate(float) {
  if (!candle_pipeline_ && chart_renderer_->GetRenderPass() != VK_NULL_HANDLE) {
    candle_pipeline_ = std::make_unique<CandlePipeline>(
        core_, chart_renderer_->GetRenderPass());
    FetchData(current_symbol_);
  }
}

void DashboardLayer::OnUIRender() {
  SetupDockspace();
  DrawSymbolSelector();
  DrawChartWindow();
}

void DashboardLayer::SetupDockspace() {
  static bool opt_fullscreen = true;
  static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

  ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
  if (opt_fullscreen) {
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  }

  ImGui::Begin("DockSpace Demo", nullptr, window_flags);
  if (opt_fullscreen)
    ImGui::PopStyleVar(2);

  ImGuiIO &io = ImGui::GetIO();
  if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
  }
  ImGui::End();
}

void DashboardLayer::DrawSymbolSelector() {
  ImGui::Begin("Symbol Selector");
  const char *symbols[] = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"};
  static int current_idx = 0;

  if (ImGui::Combo("Symbol", &current_idx, symbols, IM_ARRAYSIZE(symbols))) {
    current_symbol_ = symbols[current_idx];
    FetchData(current_symbol_);
  }

  ImGui::Separator();
  ImGui::Text("Camera Offset: %.2f, %.2f", camera_.offset_x, camera_.offset_y);
  ImGui::Text("Camera Zoom: %.2f, %.2f", camera_.scale_x, camera_.scale_y);

  if (ImGui::Button("Reset View")) {
    camera_ = ChartCamera();
  }

  ImGui::End();
}

void DashboardLayer::DrawChartWindow() {
  ImGui::Begin("Price Chart");

  ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
  glm::vec2 size = {viewportPanelSize.x, viewportPanelSize.y};

  if (size.x > 0 && size.y > 0) {
    chart_renderer_->resize(static_cast<uint32_t>(size.x),
                            static_cast<uint32_t>(size.y));

    // 1. Inputs (Pan/Zoom)
    HandleInputs(glm::vec2(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y),
                 size);

    // 2. Render Vulkan Pass
    VkCommandBuffer cmd = core_->get_current_command_buffer();
    chart_renderer_->begin_render(cmd);

    if (candle_pipeline_) {
      CandlePipeline::PushConstants pc{};
      pc.projection = glm::ortho(0.0f, size.x, 0.0f, size.y, -1.0f, 1.0f);
      pc.chart_min = glm::vec2(camera_.offset_x, camera_.offset_y);
      pc.chart_max =
          glm::vec2(camera_.offset_x + (chart_range_.x / camera_.scale_x),
                    camera_.offset_y + (chart_range_.y / camera_.scale_y));
      pc.candle_width = 10.0f * static_cast<float>(camera_.scale_x);

      candle_pipeline_->Render(cmd, current_candles_, pc);
    }

    chart_renderer_->end_render(cmd);

    // 3. Display Image
    ImGui::Image((ImTextureID)chart_renderer_->GetDescriptor(),
                 viewportPanelSize, ImVec2(0, 0), ImVec2(1, 1));

    // 4. Overlays (Crosshair)
    if (ImGui::IsItemHovered()) {
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      ImVec2 m = ImGui::GetMousePos();
      ImVec2 start = ImGui::GetItemRectMin();
      ImVec2 end = ImGui::GetItemRectMax();

      draw_list->AddLine(ImVec2(m.x, start.y), ImVec2(m.x, end.y),
                         IM_COL32(255, 255, 255, 100));
      draw_list->AddLine(ImVec2(start.x, m.y), ImVec2(end.x, m.y),
                         IM_COL32(255, 255, 255, 100));

      // Mouse Info
      glm::vec2 chart_pos = camera_.ScreenToChart(
          {m.x - start.x, m.y - start.y}, size, chart_range_);
      ImGui::SetTooltip("Price: %.2f | TimeIndex: %.1f", chart_pos.y,
                        chart_pos.x);
    }
  }

  ImGui::End();
}

void DashboardLayer::HandleInputs(const glm::vec2 &,
                                  const glm::vec2 &window_size) {
  if (!ImGui::IsWindowHovered())
    return;

  ImGuiIO &io = ImGui::GetIO();

  // Zooming
  if (io.MouseWheel != 0.0f) {
    float zoom_speed = 0.1f;
    camera_.scale_x += io.MouseWheel * zoom_speed * camera_.scale_x;
    camera_.scale_y += io.MouseWheel * zoom_speed * camera_.scale_y;

    // Prevent negative zoom
    camera_.scale_x = std::max(0.01, camera_.scale_x);
    camera_.scale_y = std::max(0.01, camera_.scale_y);
  }

  // Panning
  if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    ImVec2 delta = io.MouseDelta;
    camera_.offset_x -=
        (delta.x / window_size.x) * (chart_range_.x / camera_.scale_x);
    camera_.offset_y +=
        (delta.y / window_size.y) * (chart_range_.y / camera_.scale_y);
  }
}

void DashboardLayer::FetchData(const std::string &symbol) {
  // Simulation: Create dummy candles for the selected symbol
  current_candles_.clear();
  // uint32_t color =
  //    (symbol == "BTC-USDT") ? 0xFF00FF00 : 0xFF0000FF; // Green vs Red

  for (int i = 0; i < 500; ++i) {
    float base_price = (symbol == "BTC-USDT") ? 45000.0f : 2400.0f;
    float noise = static_cast<float>(rand() % 1000) / 10.0f;

    CandleData c;
    c.x = static_cast<float>(i);
    c.open = base_price + noise;
    c.close = base_price + noise + (rand() % 20 - 10);
    c.high = std::max(c.open, c.close) + 5.0f;
    c.low = std::min(c.open, c.close) - 5.0f;
    c.color =
        c.close >= c.open ? 0xFF00B910 : 0xFF3034EF; // Professional Green/Red

    current_candles_.push_back(c);
  }

  // Adjust chart range to fit data
  data_min_ = {0.0f, 0.0f};
  data_max_ = {500.0f, 60000.0f};
  chart_range_ = {500.0f, (symbol == "BTC-USDT") ? 60000.0f : 5000.0f};
}

} // namespace BTQuant
