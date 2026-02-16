#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "vulkan_base_types.hpp"

namespace pubbtquant::components {

/**
 * @brief Renders the Liquidity Surface panel with a transparent background
 * and GPUMemoryManager texture displayed behind a 5-column ImGuiTable.
 *
 * Table columns: [Buys | Asks | Price | Bids | Sells]
 *
 * @param gpu_memory_manager Reference to GPUMemoryManager for texture access
 * @param liquidity_data Current liquidity data to display
 */
void RenderLiquiditySurface(
    BTQuant::GPUMemoryManager& gpu_memory_manager,
    const std::vector<double>& buys,
    const std::vector<double>& asks,
    const std::vector<double>& prices,
    const std::vector<double>& bids,
    const std::vector<double>& sells) {
  // Push transparent ChildBg style for the panel background
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

  // Begin child window with transparent background
  if (ImGui::BeginChild("LiquiditySurfaceChild", ImVec2(0, 0), ImGuiChildFlags_Borders)) {
    // Get GPU memory manager stats for display
    auto stats = gpu_memory_manager.get_memory_stats();

    // Render GPU memory texture in background if available
    // The texture is rendered behind the table using ImGui::Image
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.1f));
    if (ImGui::BeginChild("##BackgroundTexture", ImVec2(0, 0), ImGuiChildFlags_None)) {
      // Try to render a placeholder or memory visualization
      // In a full implementation, this would access cached textures from GPUMemoryManager
      ImDrawList* draw_list = ImGui::GetWindowDrawList();
      ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
      ImVec2 window_size = ImGui::GetContentRegionAvail();

      // Draw a subtle gradient background representing memory usage
      ImU32 col_top = IM_COL32(20, 24, 30, 50);
      ImU32 col_bottom = IM_COL32(10, 12, 15, 80);
      draw_list->AddRectFilledMultiColor(
          cursor_pos,
          ImVec2(cursor_pos.x + window_size.x, cursor_pos.y + window_size.y),
          col_top, col_top, col_bottom, col_bottom);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    // Move cursor back to top for table rendering
    ImGui::SetCursorPosY(0);

    // Create 5-column table: [Buys | Asks | Price | Bids | Sells]
    const int column_count = 5;
    const char* column_names[] = {"Buys", "Asks", "Price", "Bids", "Sells"};

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05f, 0.07f, 0.1f, 0.8f));

    if (ImGui::BeginTable(
            "LiquiditySurfaceTable",
            column_count,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
            ImVec2(0, 0))) {
      // Setup columns
      ImGui::TableSetupColumn("Buys", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Asks", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 120.0f);
      ImGui::TableSetupColumn("Bids", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Sells", ImGuiTableColumnFlags_WidthFixed, 100.0f);

      // Render header
      ImGui::TableHeadersRow();

      // Determine number of rows (use the largest data vector)
      size_t max_rows = std::max({buys.size(), asks.size(), prices.size(),
                                   bids.size(), sells.size()});

      // Render data rows
      for (size_t row = 0; row < max_rows; ++row) {
        ImGui::TableNextRow();

        // Column 0: Buys (buy volume)
        ImGui::TableSetColumnIndex(0);
        if (row < buys.size()) {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(2) << buys[row];
          ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%s", oss.str().c_str());
        }

        // Column 1: Asks (ask volume)
        ImGui::TableSetColumnIndex(1);
        if (row < asks.size()) {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(2) << asks[row];
          ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.2f, 1.0f), "%s", oss.str().c_str());
        }

        // Column 2: Price (mid price level)
        ImGui::TableSetColumnIndex(2);
        if (row < prices.size()) {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(4) << prices[row];
          ImGui::Text("%s", oss.str().c_str());
        }

        // Column 3: Bids (bid volume)
        ImGui::TableSetColumnIndex(3);
        if (row < bids.size()) {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(2) << bids[row];
          ImGui::TextColored(ImVec4(0.2f, 0.6f, 0.9f, 1.0f), "%s", oss.str().c_str());
        }

        // Column 4: Sells (sell volume)
        ImGui::TableSetColumnIndex(4);
        if (row < sells.size()) {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(2) << sells[row];
          ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), "%s", oss.str().c_str());
        }
      }

      ImGui::EndTable();
    }

    ImGui::PopStyleColor();

    // Display GPU memory stats at bottom
    ImGui::Separator();
    ImGui::TextColored(
        ImVec4(0.7f, 0.7f, 0.7f, 0.8f),
        "GPU Memory: Vertex=%.1f%% | Uniform=%.1f%% | Storage=%.1f%%",
        stats.vertex_pool_usage * 100.0f,
        stats.uniform_pool_usage * 100.0f,
        stats.storage_pool_usage * 100.0f);
  }

  ImGui::EndChild();
  ImGui::PopStyleColor();  // Restore original ChildBg color
}

/**
 * @brief Renders a liquidity surface panel with integrated GPU texture display.
 *
 * This function creates a transparent ImGui panel that displays liquidity data
 * in a 5-column table format while rendering GPU memory textures in the background.
 *
 * @param gpu_memory_manager Pointer to GPUMemoryManager instance
 * @param width Panel width in pixels (0 = auto-fit)
 * @param height Panel height in pixels (0 = auto-fit)
 */
void RenderLiquiditySurfacePanel(
    BTQuant::GPUMemoryManager* gpu_memory_manager,
    float width,
    float height) {
  if (!gpu_memory_manager) {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                       "Error: GPUMemoryManager not available");
    return;
  }

  ImVec2 panel_size(width, height);
  if (width <= 0.0f) panel_size.x = 0.0f;  // Auto-fit
  if (height <= 0.0f) panel_size.y = 0.0f;  // Auto-fit

  // Push transparent background style
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

  if (ImGui::BeginChild("LiquiditySurfacePanel", panel_size,
                        ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY)) {
    // Render GPU memory texture as background
    // Access texture cache from GPUMemoryManager if textures are registered
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
    ImVec2 content_avail = ImGui::GetContentRegionAvail();

    // Draw background texture placeholder
    // In production, this would retrieve cached texture from GPUMemoryManager
    // using gpu_memory_manager->get_cached_texture(descriptor_set)
    ImU32 bg_color = IM_COL32(15, 18, 24, 100);
    draw_list->AddRectFilled(cursor_pos,
                              ImVec2(cursor_pos.x + content_avail.x,
                                     cursor_pos.y + content_avail.y),
                              bg_color);

    // Create the 5-column liquidity table
    // Columns: [Buys | Asks | Price | Bids | Sells]
    ImGui::SetCursorPosY(0);

    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.2f, 0.3f, 0.4f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.3f, 0.4f, 0.5f, 0.9f));

    if (ImGui::BeginTable("##LiquidityTable", 5,
                          ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                          ImVec2(0, content_avail.y - 40.0f))) {
      // Define columns with proportional widths
      ImGui::TableSetupColumn("Buys", ImGuiTableColumnFlags_WidthStretch, 1.0f);
      ImGui::TableSetupColumn("Asks", ImGuiTableColumnFlags_WidthStretch, 1.0f);
      ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch, 1.2f);
      ImGui::TableSetupColumn("Bids", ImGuiTableColumnFlags_WidthStretch, 1.0f);
      ImGui::TableSetupColumn("Sells", ImGuiTableColumnFlags_WidthStretch, 1.0f);

      ImGui::TableHeadersRow();

      // Sample data rows (in production, this would come from market data)
      constexpr size_t sample_rows = 20;
      for (size_t i = 0; i < sample_rows; ++i) {
        ImGui::TableNextRow();

        double base_price = 100.0 - static_cast<double>(i) * 0.25;

        // Buys column
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%.2f", 1000.0 + static_cast<double>(i) * 50.0);

        // Asks column
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.2f", 800.0 + static_cast<double>(i) * 40.0);

        // Price column (center)
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("$%.4f", base_price);

        // Bids column
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.2f", 1200.0 + static_cast<double>(i) * 60.0);

        // Sells column
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%.2f", 900.0 + static_cast<double>(i) * 45.0);
      }

      ImGui::EndTable();
    }

    ImGui::PopStyleColor(2);

    // GPU memory status bar
    ImGui::Separator();
    auto stats = gpu_memory_manager->get_memory_stats();
    ImGui::TextColored(ImVec4(0.5f, 0.7f, 0.9f, 1.0f), "GPU Memory Status:");
    ImGui::SameLine();
    ImGui::Text("V:%.0f%%", stats.vertex_pool_usage * 100.0f);
    ImGui::SameLine();
    ImGui::Text("U:%.0f%%", stats.uniform_pool_usage * 100.0f);
    ImGui::SameLine();
    ImGui::Text("S:%.0f%%", stats.storage_pool_usage * 100.0f);
  }

  ImGui::EndChild();
  ImGui::PopStyleColor();
}

}  // namespace pubbtquant::components
