#include "position_panel.hpp"

#include "../data/position_book.hpp"
#include "log_panel.hpp"

#include <cstdio>
#include <imgui.h>

namespace btquant::ui {

void PositionPanel::recordFill(const FillRecord& r) {
    FillRecord copy = r;
    copy.seq = ++m_seq;
    m_history.insert(m_history.begin(), copy);
    if (m_history.size() > kMaxHistory) {
        m_history.resize(kMaxHistory);
    }
}

void PositionPanel::render() {
    if (!m_open) return;

    ImGui::SetNextWindowSize(ImVec2(420, 460), ImGuiCond_Appearing);
    if (!ImGui::Begin("Position Panel", &m_open,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    if (!m_book || !m_book->hasPosition()) {
        ImGui::TextDisabled("No open position. Submit an order from the "
                            "Order Ticket (Ctrl+Enter) to open one.");
        ImGui::Separator();
        if (!m_history.empty()) {
            ImGui::Text("Recent fills (closed):");
            ImGui::Columns(6, "fills_flat", false);
            ImGui::Text("#");        ImGui::NextColumn();
            ImGui::Text("Symbol");   ImGui::NextColumn();
            ImGui::Text("Side");     ImGui::NextColumn();
            ImGui::Text("Qty");      ImGui::NextColumn();
            ImGui::Text("Price");    ImGui::NextColumn();
            ImGui::Text("Tag");      ImGui::NextColumn();
            for (const auto& r : m_history) {
                ImGui::Text("%d",   r.seq);                 ImGui::NextColumn();
                ImGui::Text("%s",   r.symbol.c_str());      ImGui::NextColumn();
                ImGui::Text("%s",   r.isLong ? "BUY" : "SELL");
                ImGui::NextColumn();
                ImGui::Text("%.4f", r.qty);                 ImGui::NextColumn();
                ImGui::Text("$%.2f",r.price);               ImGui::NextColumn();
                if (r.tag.empty()) ImGui::TextDisabled("(untagged)");
                else               ImGui::Text("%s", r.tag.c_str());
                ImGui::NextColumn();
            }
            ImGui::Columns(1);
        }
        ImGui::End();
        return;
    }

    const auto& p = m_book->position();

    ImGui::Text("Open position on %s", p.symbol.c_str());
    ImGui::Separator();

    ImGui::Columns(2, "open_pos", false);
    ImGui::SetColumnWidth(0, 160);

    ImGui::Text("Side");            ImGui::NextColumn();
    ImVec4 sideCol = p.isLong ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                              : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    ImGui::TextColored(sideCol, "%s", p.isLong ? "LONG" : "SHORT");
    ImGui::NextColumn();

    ImGui::Text("Size (base)");     ImGui::NextColumn();
    ImGui::Text("%.6f", p.size);    ImGui::NextColumn();

    ImGui::Text("Avg entry");       ImGui::NextColumn();
    ImGui::Text("$%.2f", p.avgEntry); ImGui::NextColumn();

    ImGui::Text("Fills applied");   ImGui::NextColumn();
    ImGui::Text("%d", p.fillCount); ImGui::NextColumn();

    ImGui::Text("Realized P&L");    ImGui::NextColumn();
    ImVec4 rcol = p.realizedPnL >= 0 ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                                     : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    ImGui::TextColored(rcol, "%s$%.2f",
                       p.realizedPnL >= 0 ? "+" : "", p.realizedPnL);
    ImGui::NextColumn();

    ImGui::Text("Unrealized P&L");  ImGui::NextColumn();
    ImVec4 ucol = p.unrealizedPnL >= 0 ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                                       : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    ImGui::TextColored(ucol, "%s$%.2f",
                       p.unrealizedPnL >= 0 ? "+" : "",
                       p.unrealizedPnL);
    ImGui::NextColumn();

    ImGui::Text("Total P&L");       ImGui::NextColumn();
    double total = m_book->totalPnL();
    ImVec4 tcol = total >= 0 ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                             : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    ImGui::TextColored(tcol, "%s$%.2f", total >= 0 ? "+" : "", total);
    ImGui::Columns(1);

    ImGui::Separator();
    if (ImGui::Button("Flatten at market")) {
        // Snapshot path is owned by WindowManager; here we just signal.
        BTQ_LOG_INFO("PositionPanel: flatten requested (handler in main)");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Closes the open position at the next snapshot "
                          "price. Realizes all unrealized P&L.");
    }

    ImGui::Separator();
    ImGui::Text("Recent fills:");
    ImGui::Columns(7, "fills_open", false);
    ImGui::Text("#");        ImGui::NextColumn();
    ImGui::Text("Symbol");   ImGui::NextColumn();
    ImGui::Text("Side");     ImGui::NextColumn();
    ImGui::Text("Qty");      ImGui::NextColumn();
    ImGui::Text("Price");    ImGui::NextColumn();
    ImGui::Text("ΔRealized");ImGui::NextColumn();
    ImGui::Text("Tag");      ImGui::NextColumn();
    for (const auto& r : m_history) {
        ImGui::Text("%d",   r.seq);            ImGui::NextColumn();
        ImGui::Text("%s",   r.symbol.c_str()); ImGui::NextColumn();
        ImGui::Text("%s",   r.isLong ? "BUY" : "SELL");
        ImGui::NextColumn();
        ImGui::Text("%.4f", r.qty);            ImGui::NextColumn();
        ImGui::Text("$%.2f",r.price);          ImGui::NextColumn();
        ImVec4 dcol = r.realizedDelta >= 0
                        ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                        : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
        ImGui::TextColored(dcol, "%s$%.2f",
            r.realizedDelta >= 0 ? "+" : "", r.realizedDelta);
        ImGui::NextColumn();
        if (r.tag.empty()) ImGui::TextDisabled("(untagged)");
        else               ImGui::Text("%s", r.tag.c_str());
        ImGui::NextColumn();
    }
    ImGui::Columns(1);

    ImGui::End();
}

} // namespace btquant::ui
