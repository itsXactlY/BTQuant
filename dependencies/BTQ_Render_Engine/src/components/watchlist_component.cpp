#include "vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <chrono>
#include <imgui.h>
#include <imgui_internal.h>

namespace BTQuant {

WatchlistComponent::WatchlistComponent(const glm::vec2 &position,
                                       const glm::vec2 &size)
    : UIComponent(position, size) {
  // Add some default symbols
  add_symbol("BTC-USD");
  add_symbol("ETH-USD");
  add_symbol("SOL-USD");
  add_symbol("XRP-USD");
}

void WatchlistComponent::add_symbol(const std::string &symbol) {
  std::string upper_symbol = symbol;
  std::transform(upper_symbol.begin(), upper_symbol.end(), upper_symbol.begin(),
                 ::toupper);

  for (const auto &entry : entries_) {
    if (entry.symbol == upper_symbol)
      return;
  }
  WatchlistEntry entry;
  entry.symbol = upper_symbol;
  entries_.push_back(entry);
  mark_dirty();
}

void WatchlistComponent::remove_symbol(const std::string &symbol) {
  auto it = std::remove_if(
      entries_.begin(), entries_.end(),
      [&](const WatchlistEntry &e) { return e.symbol == symbol; });
  if (it != entries_.end()) {
    entries_.erase(it, entries_.end());
    mark_dirty();
  }
}

void WatchlistComponent::update_quote(const std::string &symbol, double price,
                                      double change, double volume) {
  for (auto &entry : entries_) {
    if (entry.symbol == symbol) {
      entry.price = price;
      entry.change_24h = change;
      entry.volume_24h = volume;
      entry.last_update_ts =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
      mark_dirty();
      return;
    }
  }
}

void WatchlistComponent::update(float) {}

void WatchlistComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Watchlist", &visible_)) {
    // Search/Add bar
    static char search_buffer[64] = {0};
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 60);
    if (ImGui::InputTextWithHint("##Search", "Add Symbol...", search_buffer,
                                 sizeof(search_buffer),
                                 ImGuiInputTextFlags_EnterReturnsTrue)) {
      if (strlen(search_buffer) > 0) {
        add_symbol(search_buffer);
        search_buffer[0] = '\0';
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("ADD", ImVec2(50, 0))) {
      if (strlen(search_buffer) > 0) {
        add_symbol(search_buffer);
        search_buffer[0] = '\0';
      }
    }

    ImGui::Separator();

    if (ImGui::BeginTable("WatchlistTable", 4,
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_NoBordersInBody |
                              ImGuiTableFlags_SizingFixedFit)) {
      ImGui::TableSetupColumn("Symbol", ImGuiTableColumnFlags_WidthFixed,
                              75.0f);
      ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80.0f);
      ImGui::TableSetupColumn("Change%", ImGuiTableColumnFlags_WidthFixed,
                              65.0f);
      ImGui::TableSetupColumn("Vol 24h", ImGuiTableColumnFlags_WidthFixed,
                              70.0f);

      ImGui::TableNextRow(ImGuiTableRowFlags_Headers, 20.0f);
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("SYMBOL");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("PRICE");
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("CHG%%");
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("VOL");

      for (size_t i = 0; i < entries_.size(); ++i) {
        auto &entry = entries_[i];
        ImGui::TableNextRow(ImGuiTableRowFlags_None, 18.0f);

        ImGui::TableSetColumnIndex(0);
        bool is_active =
            (dashboard_ && dashboard_->get_active_symbol() ==
                               entry.symbol); // Check dashboard_ for null
        if (is_active)
          ImGui::TableSetBgColor(
              ImGuiTableBgTarget_RowBg0,
              ImGui::GetColorU32(ImVec4(0.0f, 0.5f, 0.5f, 0.2f)));

        if (ImGui::Selectable(entry.symbol.c_str(), is_active,
                              ImGuiSelectableFlags_SpanAllColumns |
                                  ImGuiSelectableFlags_AllowItemOverlap)) {
          if (dashboard_)
            dashboard_->set_active_symbol(
                entry.symbol); // Check dashboard_ for null
        }

        // Context menu for removal
        if (ImGui::BeginPopupContextItem()) {
          if (ImGui::MenuItem("Remove Symbol")) {
            remove_symbol(entry.symbol);
          }
          ImGui::EndPopup();
        }

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.2f", entry.price);

        ImGui::TableSetColumnIndex(2);
        // Use a default theme if theme_ is not initialized, or use the provided
        // theme
        ImVec4 change_color =
            entry.change_24h >= 0 ? theme_.price_up : theme_.price_down;
        ImGui::TextColored(change_color, "%+.2f%%", entry.change_24h);

        ImGui::TableSetColumnIndex(3);
        if (entry.volume_24h > 1000000.0)
          ImGui::Text("%.2fM", entry.volume_24h / 1000000.0);
        else
          ImGui::Text("%.2fK", entry.volume_24h / 1000.0);
      }
      ImGui::EndTable();
    }
  }
  ImGui::End();
}

} // namespace BTQuant
