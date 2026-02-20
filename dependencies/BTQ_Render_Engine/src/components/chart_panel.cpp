// #include "components/chart_panel.hpp"
// #include <imgui.h>
// namespace BTQuant {
// ChartPanel::ChartPanel(const PanelConfig& config, std::shared_ptr<MarketDataProcessor> processor,
//                        ChartManager* manager, PanelManager* panel_manager)
//     : PanelBase(config), processor_(processor), chart_manager_(manager),
//     panel_manager_(panel_manager) {}
// void ChartPanel::initialize() {}
// void ChartPanel::update(float) {}
// void ChartPanel::render_content() {
//     ImVec2 canvas_size = ImGui::GetContentRegionAvail();
//     if (canvas_size.x > 50.0f && canvas_size.y > 50.0f) {
//         ImGui::InvisibleButton("VulkanChartCanvas", canvas_size);
//         ImGui::Text("GPU Hook Ready.");
//     }
// }
// void ChartPanel::set_symbol(const std::string&, const std::string&) {}
// void ChartPanel::set_timeframe(RenderEngine::TimeFrame) {}
// void ChartPanel::center_on_timestamp(uint64_t) {}
// void ChartPanel::set_global_crosshair_position(double, bool) {}
// std::pair<double, bool> ChartPanel::get_global_crosshair_state() const { return {0.0, false}; }
// void ChartPanel::set_global_crosshair_price(double) {}
// } // namespace BTQuant
#include "components/chart_panel.hpp"

#include <imgui.h>
#include <imgui_impl_vulkan.h>  // Essenziell für den RenderState!

namespace BTQuant {

ChartPanel::ChartPanel(const PanelConfig& config, std::shared_ptr<MarketDataProcessor> processor,
                       ChartManager* manager, PanelManager* panel_manager)
    : PanelBase(config),
      processor_(processor),
      chart_manager_(manager),
      panel_manager_(panel_manager) {}

void ChartPanel::initialize() {}
void ChartPanel::update(float) {}

void ChartPanel::render_content() {
  ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
  ImVec2 canvas_size = ImGui::GetContentRegionAvail();

  if (canvas_size.x < 50.0f || canvas_size.y < 50.0f) {
    ImGui::Text("Panel too small.");
    return;
  }

  // 1. Platzhalter setzen, fängt Maus-Events ab und hält den Platz frei
  ImGui::InvisibleButton("VulkanChartCanvas", canvas_size);
  bool is_hovered = ImGui::IsItemHovered();

  // 2. Simple Interaktions-Logik (Vorgriff auf später)
  if (is_hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    // Panning wird hier später die PushConstants verändern
  }

  // 3. Sicherheits-Check: Ist die Pipeline gebunden?
  if (!chart_pipeline_ || descriptor_set_ == VK_NULL_HANDLE) {
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Awaiting GPU Memory Allocation...");
    return;
  }

  // 4. Draw Data vorbereiten (wird asynchron gelesen)
  current_draw_data_.pipeline = chart_pipeline_;
  current_draw_data_.descriptor_set = descriptor_set_;
  current_draw_data_.instance_count = 10000;  // Hardcoded für den ersten Bluttest

  // MMT Neon-Theme
  current_draw_data_.push_constants.bull_color = {0.08f, 0.8f, 0.52f, 1.0f};
  current_draw_data_.push_constants.bear_color = {0.96f, 0.27f, 0.36f, 1.0f};
  current_draw_data_.push_constants.wick_color = {0.5f, 0.5f, 0.5f, 1.0f};
  current_draw_data_.push_constants.time_scale = 1.0f;
  current_draw_data_.push_constants.price_scale = 1.0f;
  current_draw_data_.push_constants.time_offset = 0.0f;
  current_draw_data_.push_constants.price_offset = 0.0f;

  // 5. DIE INJEKTION
  ImGui::GetWindowDrawList()->AddCallback(
      [](const ImDrawList* parent_list, const ImDrawCmd* cmd) {
        auto* data = static_cast<ChartDrawData*>(cmd->UserCallbackData);
        if (!data || !data->pipeline) return;

        // CommandBuffer vom ImGui-Backend abgreifen
        auto* render_state =
            static_cast<ImGui_ImplVulkan_RenderState*>(ImGui::GetPlatformIO().Renderer_RenderState);
        if (!render_state) return;
        VkCommandBuffer cb = render_state->CommandBuffer;

        // 1. Pipeline binden (Candlestick Bodies)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, data->pipeline->get_pipeline());

        // 2. Kamera & Farben via Push Constants senden
        vkCmdPushConstants(cb, data->pipeline->get_pipeline_layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(Rendering::ChartPushConstants), &data->push_constants);

        // 3. Uniforms & SSBO (Instances) binden
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                data->pipeline->get_pipeline_layout(), 0, 1, &data->descriptor_set,
                                0, nullptr);

        // 4. Der magische Befehl: 6 Vertices pro Instanz generieren
        vkCmdDraw(cb, 6, data->instance_count, 0, 0);

        // Optional: Wick-Pipeline hier später nachschieben!
      },
      &current_draw_data_);

  // Crosshair drüber zeichnen (CPU Overlay)
  if (is_hovered) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 mouse_pos = ImGui::GetMousePos();
    draw_list->AddLine(ImVec2(canvas_pos.x, mouse_pos.y),
                       ImVec2(canvas_pos.x + canvas_size.x, mouse_pos.y),
                       IM_COL32(255, 255, 255, 100), 1.0f);
    draw_list->AddLine(ImVec2(mouse_pos.x, canvas_pos.y),
                       ImVec2(mouse_pos.x, canvas_pos.y + canvas_size.y),
                       IM_COL32(255, 255, 255, 100), 1.0f);
  }
}

void ChartPanel::set_symbol(const std::string& symbol) {
  symbol_ = symbol;
  config_.symbol = symbol;
}

void ChartPanel::set_timeframe(RenderEngine::TimeFrame timeframe) { timeframe_ = timeframe; }

void ChartPanel::center_on_timestamp(uint64_t) {}

void ChartPanel::set_global_crosshair_position(double, bool) {}

std::pair<double, bool> ChartPanel::get_global_crosshair_state() const { return {0.0, false}; }

void ChartPanel::set_global_crosshair_price(double) {}

}  // namespace BTQuant