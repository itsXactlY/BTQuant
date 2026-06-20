#include "watchlist_widget.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <imgui.h>

namespace btquant::ui {

void WatchlistWidget::setSymbols(const std::vector<std::string>& syms) {
    m_symbols = syms;
    for (const auto& s : syms) {
        if (m_rows.find(s) == m_rows.end()) m_rows[s] = Row{s, {}, 0, 0, 0, 0, 0, 0, 0};
    }
}

WatchlistWidget::Row* WatchlistWidget::row(const std::string& symbol) {
    auto it = m_rows.find(symbol);
    return it == m_rows.end() ? nullptr : &it->second;
}

void WatchlistWidget::update(const std::string& symbol, double price, double size,
                             bool isBuy, uint64_t timestamp) {
    auto& r = m_rows[symbol];
    if (r.symbol.empty()) r.symbol = symbol;
    if (r.lastPrice != 0.0) r.prevPrice = r.lastPrice;
    r.lastPrice = price;
    r.lastTs    = timestamp;
    r.totalVol += std::fabs(size);
    if (isBuy) r.buyVol += std::fabs(size);
    else       r.sellVol += std::fabs(size);
    ++r.tickCount;
    r.spark.push_back(price);
    while (r.spark.size() > kMaxSparkPoints) r.spark.pop_front();
}

void WatchlistWidget::render() {
    if (!ImGui::Begin("Watchlist", nullptr, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }
    ImGui::Text("Symbols: %zu", m_rows.size());
    ImGui::SameLine();
    if (ImGui::SmallButton("+ Default")) {
        static const char* defaults[] = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"};
        for (auto* s : defaults) m_symbols.emplace_back(s);
        setSymbols(m_symbols);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear All")) { clear(); m_symbols.clear(); }

    ImGui::Separator();
    if (ImGui::BeginTable("watch", 6, ImGuiTableFlags_RowBg |
                                          ImGuiTableFlags_ScrollY |
                                          ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("Symbol",   ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Last",     ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Δ%",       ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Vol",      ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Buy/Sell", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Sparkline", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        // Display in stable order — use m_symbols, fall back to map insertion order.
        std::vector<const Row*> ordered;
        for (const auto& sym : m_symbols) {
            if (auto* r = row(sym)) ordered.push_back(r);
        }
        for (const auto& [k, v] : m_rows) {
            if (std::find(m_symbols.begin(), m_symbols.end(), k) == m_symbols.end()) {
                ordered.push_back(&v);
            }
        }

        for (const Row* r : ordered) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r->symbol.c_str());

            ImGui::TableNextColumn();
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f", r->lastPrice);
            ImGui::TextUnformatted(buf);

            ImGui::TableNextColumn();
            if (r->prevPrice > 0.0) {
                double pct = (r->lastPrice - r->prevPrice) / r->prevPrice * 100.0;
                ImVec4 col = pct > 0 ? ImVec4(0.30f, 0.95f, 0.40f, 1.0f)
                           : pct < 0 ? ImVec4(0.95f, 0.30f, 0.30f, 1.0f)
                                     : ImGui::GetStyleColorVec4(ImGuiCol_Text);
                ImGui::TextColored(col, "%+.2f%%", pct);
            } else {
                ImGui::TextDisabled("--");
            }

            ImGui::TableNextColumn();
            std::snprintf(buf, sizeof(buf), "%.2f", r->totalVol);
            ImGui::TextUnformatted(buf);

            ImGui::TableNextColumn();
            std::snprintf(buf, sizeof(buf), "%.0f / %.0f", r->buyVol, r->sellVol);
            ImGui::TextUnformatted(buf);

            ImGui::TableNextColumn();
            drawSparkline(*r, 120.0f, 28.0f);
        }

        ImGui::EndTable();
    }
    ImGui::End();
}

void WatchlistWidget::drawSparkline(const Row& r, float width, float height) const {
    if (r.spark.size() < 2) {
        ImGui::Dummy(ImVec2(width, height));
        return;
    }
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    double lo = *std::min_element(r.spark.begin(), r.spark.end());
    double hi = *std::max_element(r.spark.begin(), r.spark.end());
    double span = std::max(hi - lo, 1e-9);
    ImVec2 size(width, height);
    dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y),
                      IM_COL32(20, 20, 28, 255));
    float dx = size.x / static_cast<float>(r.spark.size() - 1);
    ImU32 col = IM_COL32(80, 200, 120, 255);
    for (size_t i = 1; i < r.spark.size(); ++i) {
        float x0 = p.x + dx * static_cast<float>(i - 1);
        float x1 = p.x + dx * static_cast<float>(i);
        float y0 = p.y + size.y - static_cast<float>((r.spark[i - 1] - lo) / span) * size.y;
        float y1 = p.y + size.y - static_cast<float>((r.spark[i]     - lo) / span) * size.y;
        dl->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), col, 1.5f);
    }
    ImGui::Dummy(size);
}

} // namespace btquant::ui
