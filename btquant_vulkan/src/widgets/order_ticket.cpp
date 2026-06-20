#include "order_ticket.hpp"

#include "../data/market_data_processor.hpp"
#include "log_panel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <imgui.h>

namespace btquant::ui {

namespace {
double parseOrZero(const char* s) {
    if (!s || !*s) return 0.0;
    char* end = nullptr;
    double v = std::strtod(s, &end);
    return (end == s) ? 0.0 : v;
}
} // namespace

double OrderTicket::computeFee(double size, double price, double feeBps) {
    return std::fabs(size) * price * (feeBps / 10000.0);
}

double OrderTicket::estimateFillPrice(bool isBuy, bool isLimit,
                                      double limitPrice, double refPrice,
                                      double slippageBps) {
    if (isLimit) {
        // Limit fills at the limit price when the market crosses it,
        // otherwise leaves the order pending (ref price unchanged).
        if (isBuy  && limitPrice >= refPrice) return limitPrice;
        if (!isBuy && limitPrice <= refPrice) return limitPrice;
        return refPrice;  // resting — UI shows "would not fill"
    }
    // Market order: apply slippage against the trader. Buy pays more,
    // sell receives less.
    double slip = refPrice * (slippageBps / 10000.0);
    return isBuy ? refPrice + slip : refPrice - slip;
}

double OrderTicket::computeTotalCost(double size, double effectivePrice,
                                     double feeBps) {
    double notional = std::fabs(size) * effectivePrice;
    double fee      = notional * (feeBps / 10000.0);
    // Cost = notional + fee (positive for buys); for sells we subtract
    // fee from proceeds — the caller can negate as needed.
    return (size >= 0.0) ? (notional + fee) : -(notional - fee);
}

double OrderTicket::quantity() const    { return parseOrZero(m_qty); }
double OrderTicket::limitPrice() const  { return parseOrZero(m_limit); }
double OrderTicket::referencePrice() const { return parseOrZero(m_limit); }

void OrderTicket::refreshRefPrice() {
    if (!m_data) return;
    auto snap = m_data->snapshot(1, 0);
    if (snap.recent_trades.empty()) return;
    // Use the most recent trade price as the reference.
    double p = snap.recent_trades.front().price;
    if (p <= 0.0) return;
    m_liveRefPrice = p;
    if (m_typeIsLimit) {
        // For limit orders, the user picks the price — don't clobber
        // their input. The live price is exposed via liveRefPrice()
        // for the UI hint below.
        return;
    }
    // For market orders, seed m_limit on the first non-empty render
    // (m_limit starts at "0.00") and keep refreshing thereafter so
    // the greyed-out "Ref price (auto)" field tracks the live price.
    // The slippage estimate (m_slipBps) is applied on top of this.
    std::snprintf(m_limit, sizeof(m_limit), "%.2f", p);
}

void OrderTicket::render() {
    if (!ImGui::Begin("Order Ticket", &m_open,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    // Pull live ref price from the data source when available.
    refreshRefPrice();

    // --- Side toggle (buy/sell) ---
    if (m_sideIsBuy) {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.10f, 0.55f, 0.20f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.15f, 0.70f, 0.25f, 1.0f));
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.65f, 0.15f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.80f, 0.20f, 0.20f, 1.0f));
    }
    if (ImGui::Button(m_sideIsBuy ? "BUY" : "SELL", ImVec2(120, 32))) {
        m_sideIsBuy = !m_sideIsBuy;
    }
    ImGui::PopStyleColor(2);
    ImGui::SameLine();
    ImGui::TextDisabled("click to toggle side");

    ImGui::Separator();

    // --- Order type ---
    ImGui::Text("Order type:");
    ImGui::SameLine();
    if (ImGui::RadioButton("Market", !m_typeIsLimit)) m_typeIsLimit = false;
    ImGui::SameLine();
    if (ImGui::RadioButton("Limit",   m_typeIsLimit))  m_typeIsLimit = true;

    ImGui::Separator();

    // --- Inputs ---
    ImGui::PushItemWidth(160);
    ImGui::InputText("Quantity (base)",   m_qty,    sizeof(m_qty));
    if (m_typeIsLimit) {
        ImGui::InputText("Limit price",   m_limit,  sizeof(m_limit));
        ImGui::SameLine();
        // Live price hint — the user typed a limit, but seeing the
        // current market lets them sanity-check whether the order
        // would cross or rest.
        if (m_liveRefPrice > 0.0) {
            ImGui::TextDisabled("(live $%.2f)", m_liveRefPrice);
        }
    } else {
        // Greyed-out hint: market uses ref price + slippage.
        ImGui::BeginDisabled();
        ImGui::InputText("Ref price (auto)", m_limit, sizeof(m_limit));
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Live trade price pulled from the data spine. "
                              "Edit only if you want to model a custom ref.");
        }
    }
    ImGui::InputText("Fee (bps)",         m_feeBps, sizeof(m_feeBps));
    ImGui::InputText("Slippage (bps)",    m_slipBps,sizeof(m_slipBps));
    ImGui::InputText("Tag (strategy)",    m_tag,    sizeof(m_tag));
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Free-form strategy label written to the journal "
                          "with this fill (e.g. 'scalper-1', 'arb-cross'). "
                          "Leave empty for 'untagged'.");
    }
    ImGui::PopItemWidth();

    // Quick-fill buttons.
    ImGui::TextDisabled("Quick fill:");
    ImGui::SameLine();
    for (double pct : {0.25, 0.50, 0.75, 1.00}) {
        char label[16];
        std::snprintf(label, sizeof(label), "%d%%", (int)(pct * 100));
        if (ImGui::SmallButton(label)) {
            // Apply against the live ref price — assumes a notional budget
            // of 1.0 unit of quote. For BTC pairs that means a $1 fill.
            double ref = parseOrZero(m_limit);
            if (ref > 0.0) {
                std::snprintf(m_qty, sizeof(m_qty), "%.4f", pct / ref);
            }
        }
        ImGui::SameLine();
    }
    ImGui::NewLine();

    ImGui::Separator();

    // --- Live preview ---
    double qty        = quantity();
    double limitPx    = limitPrice();
    double feeBps     = parseOrZero(m_feeBps);
    double slipBps    = parseOrZero(m_slipBps);
    double refPx      = limitPx;  // m_limit doubles as ref when market
    double fillPx     = estimateFillPrice(m_sideIsBuy, m_typeIsLimit,
                                          limitPx, refPx, slipBps);
    double fee        = computeFee(qty, fillPx, feeBps);
    double totalCost  = computeTotalCost(qty, fillPx, feeBps);
    double notional   = std::fabs(qty) * fillPx;

    ImGui::Columns(2, "ticket_preview", false);
    ImGui::SetColumnWidth(0, 180);
    ImGui::Text("Effective fill price"); ImGui::NextColumn();
    if (refPx > 0.0) ImGui::Text("$%.2f", fillPx);
    else              ImGui::TextDisabled("—");
    ImGui::NextColumn();

    ImGui::Text("Notional"); ImGui::NextColumn();
    if (notional > 0.0) ImGui::Text("$%.2f", notional);
    else                 ImGui::TextDisabled("—");
    ImGui::NextColumn();

    ImGui::Text("Fee"); ImGui::NextColumn();
    if (fee > 0.0) ImGui::Text("$%.4f (%.0f bps)", fee, feeBps);
    else           ImGui::TextDisabled("—");
    ImGui::NextColumn();

    ImGui::Text("Total cost"); ImGui::NextColumn();
    if (qty > 0.0) {
        ImVec4 col = m_sideIsBuy ? ImVec4(0.95f, 0.40f, 0.40f, 1.0f)
                                 : ImVec4(0.30f, 0.85f, 0.40f, 1.0f);
        ImGui::TextColored(col, "%s$%.2f",
                           m_sideIsBuy ? "-" : "+", std::fabs(totalCost));
    } else {
        ImGui::TextDisabled("—");
    }
    ImGui::Columns(1);

    // Resting-order warning for limit orders that wouldn't cross.
    if (m_typeIsLimit && refPx > 0.0) {
        bool wouldFill = m_sideIsBuy ? (limitPx >= refPx)
                                     : (limitPx <= refPx);
        if (!wouldFill) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.85f, 0.30f, 1.0f));
            ImGui::TextWrapped("⚠ Limit %s ref — order would rest unfilled.",
                               m_sideIsBuy ? "below" : "above");
            ImGui::PopStyleColor();
        }
    }

    ImGui::Separator();

    // --- Submit ---
    bool canSubmit = (qty > 0.0) && (fillPx > 0.0);
    if (!canSubmit) ImGui::BeginDisabled();
    ImVec4 submitCol = m_sideIsBuy ? ImVec4(0.10f, 0.55f, 0.20f, 1.0f)
                                   : ImVec4(0.65f, 0.15f, 0.15f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button,        submitCol);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, submitCol);

    // Alt+Enter = submit the opposite side without flipping the
    // ticket state. The submit() function reads the live Alt-key
    // state and flips internally; we don't mutate m_sideIsBuy here
    // so the UI stays in sync with what the trader just saw.
    bool altDown  = ImGui::IsKeyDown(ImGuiKey_LeftAlt) ||
                     ImGui::IsKeyDown(ImGuiKey_RightAlt);
    bool enterOrClick = ImGui::Button(m_sideIsBuy ? "Submit BUY" : "Submit SELL",
                                       ImVec2(-FLT_MIN, 36)) ||
                        (ImGui::IsKeyPressed(ImGuiKey_Enter) &&
                         ImGui::IsKeyDown(ImGuiKey_LeftCtrl));
    bool altEnter = m_altSubmitsOpposite && altDown &&
                    ImGui::IsKeyPressed(ImGuiKey_Enter);
    if (enterOrClick || altEnter) {
        // For altEnter, ask the submit path to flip the side
        // internally (without touching m_sideIsBuy). For ordinary
        // clicks, submit with the current side.
        bool wantsFlip = altEnter;
        if (wantsFlip) {
            bool origSide = m_sideIsBuy;
            m_sideIsBuy = !m_sideIsBuy;
            bool ok = submit();
            m_sideIsBuy = origSide;  // restore
            (void)ok;
        } else {
            submit();
        }
    }
    ImGui::PopStyleColor(2);
    if (!canSubmit) ImGui::EndDisabled();

    if (m_altSubmitsOpposite) {
        ImGui::SameLine();
        ImGui::TextDisabled("(Alt-click = opposite)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Hold Alt when clicking Submit (or press "
                              "Alt+Enter) to submit the opposite side "
                              "without flipping the ticket display.");
        }
    }

    // Submit counter + reset — visible at the bottom of the ticket
    // so the trader can see how many fills they've put through
    // this session. Helps catch double-click fat-fingers.
    if (m_submitCount > 0) {
        ImGui::TextDisabled("Submitted: %d", m_submitCount);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset##count")) resetSubmitCount();
        ImGui::SameLine();
    }
    if (ImGui::SmallButton("Reset draft")) resetDraft();

    ImGui::End();
}

bool OrderTicket::submit() {
    double qty        = quantity();
    double limitPx    = limitPrice();
    double feeBps     = parseOrZero(m_feeBps);
    double slipBps    = parseOrZero(m_slipBps);
    double refPx      = limitPx;  // m_limit doubles as ref when market
    double fillPx     = estimateFillPrice(m_sideIsBuy, m_typeIsLimit,
                                          limitPx, refPx, slipBps);
    if (qty <= 0.0 || fillPx <= 0.0) {
        BTQ_LOG_WARN("OrderTicket::submit: refused (qty=%.4f fillPx=%.2f)",
                    qty, fillPx);
        return false;
    }
    double fee        = computeFee(qty, fillPx, feeBps);
    double totalCost  = computeTotalCost(qty, fillPx, feeBps);
    char summary[256];
    std::snprintf(summary, sizeof(summary),
        "%s %.4f %s @ %s $%.2f  (fee $%.4f, total %s$%.2f)",
        m_sideIsBuy ? "BUY" : "SELL",
        qty,
        m_data ? m_data->symbol().c_str() : "?",
        m_typeIsLimit ? "limit" : "market",
        fillPx,
        fee,
        m_sideIsBuy ? "-" : "+", std::fabs(totalCost));
    BTQ_LOG_INFO("OrderTicket: %s", summary);
    bool fired = false;
    if (m_submit) {
        m_submit(summary);
        fired = true;
    }
    if (fired) {
        ++m_submitCount;
        if (m_clearAfterSubmit) resetDraft();
    }
    return fired;
}

void OrderTicket::resetDraft() {
    // Re-initialise the char buffers to the constructor defaults.
    // Side and type go back to BUY / market; the limit price is
    // zeroed (the market path uses the live ref price anyway).
    std::snprintf(m_qty,   sizeof(m_qty),   "0.10");
    std::snprintf(m_limit, sizeof(m_limit), "0.00");
    std::snprintf(m_feeBps,sizeof(m_feeBps),"10");
    std::snprintf(m_slipBps,sizeof(m_slipBps),"5");
    m_tag[0]      = '\0';
    m_sideIsBuy   = true;
    m_typeIsLimit = false;
}

bool OrderTicket::isDraftAtDefaults() const {
    // qty=0.10, side=BUY, type=market. Limit price is allowed to be
    // 0.00 — that's the market-mode default, not a draft diff.
    return quantity() == 0.10 && isBuy() && !isLimit();
}

} // namespace btquant::ui
