#include "vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <imgui.h>

MarketScreenerComponent::MarketScreenerComponent(const glm::vec2 &position,
                                                 const glm::vec2 &size)
    : UIComponent(position, size) {
  // Initial mock data for the primary universe
  results_.push_back({"BTC-USD", 43200.50, 1.25, 4500000000.0, 1.1});
  results_.push_back({"ETH-USD", 2250.75, -0.85, 1200000000.0, 0.9});
  results_.push_back({"SOL-USD", 95.20, 5.40, 800000000.0, 2.5});
  results_.push_back({"XRP-USD", 0.58, 0.15, 300000000.0, 1.0});
  results_.push_back({"ADA-USD", 0.45, -2.10, 150000000.0, 0.8});
  results_.push_back({"DOT-USD", 7.20, 1.50, 100000000.0, 1.2});
  results_.push_back({"LINK-USD", 15.40, 3.20, 250000000.0, 1.8});
  results_.push_back({"AVAX-USD", 35.80, -4.50, 180000000.0, 0.7});
}

MarketScreenerComponent::~MarketScreenerComponent() {}

void MarketScreenerComponent::update(float delta_time) {
  // In a real system, this would throttle-scan the symbol registry
}

void MarketScreenerComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Market Screener", &visible_)) {
    if (ImGui::BeginTable("ScreenerTable", 5,
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_Sortable |
                              ImGuiTableFlags_BordersOuter)) {
      ImGui::TableSetupColumn("Symbol", ImGuiTableColumnFlags_WidthFixed,
                              85.0f);
      ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80.0f);
      ImGui::TableSetupColumn("24h Change%", ImGuiTableColumnFlags_WidthFixed,
                              95.0f);
      ImGui::TableSetupColumn("Vol 24h", ImGuiTableColumnFlags_WidthFixed,
                              90.0f);
      ImGui::TableSetupColumn("Activity", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      // Handle sorting logic
      if (ImGuiTableSortSpecs *sort_specs = ImGui::TableGetSortSpecs()) {
        if (sort_specs->SpecsDirty) {
          std::sort(
              results_.begin(), results_.end(),
              [&](const ScreenerResult &a, const ScreenerResult &b) {
                for (int n = 0; n < sort_specs->SpecsCount; n++) {
                  const ImGuiTableColumnSortSpecs *spec = &sort_specs->Specs[n];
                  int res = 0;
                  if (spec->ColumnIndex == 0)
                    res = a.symbol.compare(b.symbol);
                  else if (spec->ColumnIndex == 1)
                    res = (a.price < b.price)   ? -1
                          : (a.price > b.price) ? 1
                                                : 0;
                  else if (spec->ColumnIndex == 2)
                    res = (a.change_24h < b.change_24h)   ? -1
                          : (a.change_24h > b.change_24h) ? 1
                                                          : 0;
                  else if (spec->ColumnIndex == 3)
                    res = (a.volume_24h < b.volume_24h)   ? -1
                          : (a.volume_24h > b.volume_24h) ? 1
                                                          : 0;
                  else if (spec->ColumnIndex == 4)
                    res = (a.vol_spike_ratio < b.vol_spike_ratio)   ? -1
                          : (a.vol_spike_ratio > b.vol_spike_ratio) ? 1
                                                                    : 0;

                  if (res != 0) {
                    if (spec->SortDirection == ImGuiSortDirection_Ascending)
                      return res < 0;
                    return res > 0;
                  }
                }
                return false;
              });
          sort_specs->SpecsDirty = false;
        }
      }

      for (const auto &res : results_) {
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        if (ImGui::Selectable(res.symbol.c_str(), false,
                              ImGuiSelectableFlags_SpanAllColumns)) {
          // Selection logic would update active charts
        }

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.2f", res.price);

        ImGui::TableSetColumnIndex(2);
        ImVec4 change_color = res.change_24h >= 0 ? ImVec4(0, 1, 0.4f, 1)
                                                  : ImVec4(1, 0.2f, 0.2f, 1);
        ImGui::TextColored(change_color, "%+.2f%%", res.change_24h);

        ImGui::TableSetColumnIndex(3);
        if (res.volume_24h >= 1e9)
          ImGui::Text("%.1fB", res.volume_24h / 1e9);
        else
          ImGui::Text("%.1fM", res.volume_24h / 1e6);

        ImGui::TableSetColumnIndex(4);
        if (res.vol_spike_ratio > 1.5) {
          float t = std::min(1.0f, (float)((res.vol_spike_ratio - 1.0) / 4.0));
          ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                ImVec4(1.0f, 0.5f, 0.0f, 0.8f));
          ImGui::ProgressBar(t, ImVec2(-FLT_MIN, 0), "SPIKE");
          ImGui::PopStyleColor();
        } else {
          ImGui::Text("%.1fx avg", res.vol_spike_ratio);
        }
      }
      ImGui::EndTable();
    }
  }
  ImGui::End();
}
