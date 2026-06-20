#include "risk_limits_panel.hpp"

#include "../data/risk_guard.hpp"
#include "../data/position_book.hpp"
#include "log_panel.hpp"

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

double RiskLimitsPanel::editedMaxPositionSizeUSD() const {
    return parseOrZero(m_maxPos);
}
double RiskLimitsPanel::editedMaxLeverage() const {
    return parseOrZero(m_maxLev);
}
double RiskLimitsPanel::editedKillOnDailyLossUSD() const {
    return parseOrZero(m_killUSD);
}
double RiskLimitsPanel::editedEquityUSD() const {
    return parseOrZero(m_equity);
}

void RiskLimitsPanel::syncFromGuard() {
    if (!m_guard) return;
    const auto& c = m_guard->config();
    std::snprintf(m_maxPos,  sizeof(m_maxPos),  "%.2f", c.maxPositionSizeUSD);
    std::snprintf(m_maxLev,  sizeof(m_maxLev),  "%.2f", c.maxLeverage);
    std::snprintf(m_killUSD, sizeof(m_killUSD), "%.2f", c.killOnDailyLossUSD);
    std::snprintf(m_equity,  sizeof(m_equity),  "%.2f", c.equityUSD);
}

void RiskLimitsPanel::applyToGuard() {
    if (!m_guard) return;
    ::btquant::RiskConfig c = m_guard->config();
    c.maxPositionSizeUSD = editedMaxPositionSizeUSD();
    c.maxLeverage        = editedMaxLeverage();
    c.killOnDailyLossUSD = editedKillOnDailyLossUSD();
    c.equityUSD          = editedEquityUSD();
    m_guard->setConfig(c);
    m_lastAppliedKillUSD = c.killOnDailyLossUSD;
    BTQ_LOG_INFO("RiskLimits: applied caps $%.0f / %.2fx / kill $%.0f / eq $%.0f",
                 c.maxPositionSizeUSD, c.maxLeverage,
                 c.killOnDailyLossUSD, c.equityUSD);
    if (m_persistFn) m_persistFn(*m_guard);
}

void RiskLimitsPanel::applyBufferToGuardField() {
    if (!m_guard) return;
    // Push only the kill-threshold field for snappy live updates; the
    // other caps are usually set once and rarely tweaked mid-session,
    // so per-keystroke syncing them is overkill. The next Apply will
    // flush all four.
    ::btquant::RiskConfig c = m_guard->config();
    c.killOnDailyLossUSD = editedKillOnDailyLossUSD();
    m_guard->setConfig(c);
    m_lastAppliedKillUSD = c.killOnDailyLossUSD;
}

void RiskLimitsPanel::render() {
    if (!m_open) return;

    // Lazy sync — first render pulls current values from the guard.
    static bool synced = false;
    if (!synced && m_guard) { syncFromGuard(); synced = true; }

    ImGui::SetNextWindowSize(ImVec2(440, 520), ImGuiCond_Appearing);
    if (!ImGui::Begin("Risk Dashboard", &m_open,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    if (!m_guard) {
        ImGui::TextDisabled("No RiskGuard bound — limits panel inert.");
        ImGui::End();
        return;
    }

    // ---- Status banner ----
    bool tripped = m_guard->isKillTripped();
    double remaining = m_guard->remainingLossBudget();
    double sessionReal = m_guard->sessionRealized();
    double fracLeft = m_guard->config().killOnDailyLossUSD > 0.0
                        ? std::max(0.0, remaining /
                                       m_guard->config().killOnDailyLossUSD)
                        : 0.0;

    ImVec4 bannerCol;
    const char* bannerText;
    if (tripped) {
        bannerCol = ImVec4(0.85f, 0.15f, 0.15f, 1.0f);
        bannerText = "KILL SWITCH TRIPPED — orders blocked";
    } else if (fracLeft < 0.20) {
        bannerCol = ImVec4(1.00f, 0.70f, 0.20f, 1.0f);
        bannerText = "WARNING — loss budget low";
    } else {
        bannerCol = ImVec4(0.20f, 0.70f, 0.30f, 1.0f);
        bannerText = "ARMED — risk checks active";
    }
    ImGui::PushStyleColor(ImGuiCol_Header,        bannerCol);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, bannerCol);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  bannerCol);
    ImGui::Selectable(bannerText, false,
                      ImGuiSelectableFlags_None);
    ImGui::PopStyleColor(3);

    ImGui::Separator();

    // ---- Session stats ----
    ImGui::Columns(2, "risk_stats", false);
    ImGui::SetColumnWidth(0, 180);
    ImGui::Text("Session realized"); ImGui::NextColumn();
    ImVec4 rc = sessionReal >= 0 ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                                 : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    ImGui::TextColored(rc, "%s$%.2f",
                       sessionReal >= 0 ? "+" : "", sessionReal);
    ImGui::NextColumn();

    ImGui::Text("Remaining loss budget"); ImGui::NextColumn();
    ImVec4 bc = remaining > 0 ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                              : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    ImGui::TextColored(bc, "$%.2f (%.0f%%)", remaining, fracLeft * 100.0);
    ImGui::NextColumn();

    ImGui::Text("Kill threshold"); ImGui::NextColumn();
    ImGui::Text("-$%.2f", m_guard->config().killOnDailyLossUSD);
    ImGui::NextColumn();

    // Visual progress bar — red as it fills toward kill.
    ImGui::Text("Budget used"); ImGui::NextColumn();
    float used = std::max(0.0f, std::min(1.0f, 1.0f - (float)fracLeft));
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
        used > 0.8f ? ImVec4(0.85f, 0.15f, 0.15f, 1.0f)
        : used > 0.5f ? ImVec4(1.00f, 0.70f, 0.20f, 1.0f)
                     : ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
    ImGui::ProgressBar(used, ImVec2(-FLT_MIN, 0), "");
    ImGui::PopStyleColor();
    ImGui::Columns(1);

    ImGui::Separator();

    // ---- Current exposure ----
    ImGui::Text("Current exposure:");
    if (m_book && m_book->hasPosition()) {
        const auto& p = m_book->position();
        double notional = p.size * p.avgEntry;
        double fracNotional = m_guard->config().maxPositionSizeUSD > 0.0
            ? notional / m_guard->config().maxPositionSizeUSD
            : 0.0;
        ImGui::BulletText("%s %.4f @ $%.2f = $%.2f (%.0f%% of cap)",
                          p.isLong ? "LONG" : "SHORT",
                          p.size, p.avgEntry, notional, fracNotional * 100.0);
    } else {
        ImGui::BulletText("flat (no open position)");
    }

    ImGui::Separator();

    // ---- Edit limits ----
    ImGui::Text("Limits:");
    ImGui::SameLine();
    if (ImGui::Checkbox("Live update", &m_liveUpdate)) {
        if (m_liveUpdate) {
            // First time the user flips this on, push the current
            // buffers into the guard so the progress bar snaps to
            // what the edit fields show — otherwise there's a moment
            // where the buffer says X but the guard says Y.
            applyBufferToGuardField();
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Stream kill-threshold edits to the RiskPanel "
                          "progress bar in real time (no Apply click).");
    }
    ImGui::PushItemWidth(180);
    if (ImGui::InputText("Max position (USD)",   m_maxPos,  sizeof(m_maxPos)) ||
        ImGui::InputText("Max leverage (x)",     m_maxLev,  sizeof(m_maxLev)) ||
        ImGui::InputText("Kill on loss (USD)",   m_killUSD, sizeof(m_killUSD)) ||
        ImGui::InputText("Account equity (USD)", m_equity,  sizeof(m_equity))) {
        if (m_liveUpdate) applyBufferToGuardField();
    }
    ImGui::PopItemWidth();
    if (isKillDirty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f), "* unsaved");
    } else if (m_liveUpdate) {
        ImGui::SameLine();
        ImGui::TextDisabled("(live)");
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply")) {
        applyToGuard();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        syncFromGuard();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear session")) {
        m_guard->resetSession();
        BTQ_LOG_WARN("RiskLimits: session P&L cleared (kill-trip reset)");
    }

    ImGui::Separator();
    if (ImGui::Button("Apply conservative preset")) {
        m_guard->setConfig(::btquant::RiskConfig::conservative());
        syncFromGuard();
        BTQ_LOG_INFO("RiskLimits: switched to conservative preset");
        if (m_persistFn) m_persistFn(*m_guard);
    }
    if (ImGui::Button("Apply aggressive preset")) {
        m_guard->setConfig(::btquant::RiskConfig::aggressive());
        syncFromGuard();
        BTQ_LOG_INFO("RiskLimits: switched to aggressive preset");
        if (m_persistFn) m_persistFn(*m_guard);
    }

    ImGui::Separator();

    // ---- Per-symbol order notional caps ----
    //
    // The global maxPositionSizeUSD applies to every order, but some
    // traders want tighter (or looser) caps on specific symbols —
    // e.g. cap BTCUSDT at $250k while leaving smaller coins at the
    // global $100k. This section lets the trader add / edit / clear
    // per-symbol overrides that the RiskGuard enforces at order
    // entry. Empty override = fall back to global cap.
    ImGui::Text("Per-symbol order caps (USD notional):");
    auto overrides = m_guard->maxOrderNotionalBySymbol();
    if (overrides.empty()) {
        ImGui::TextDisabled("(no per-symbol overrides — all symbols use "
                            "the global cap)");
    } else {
        // Resize edit buffers to match the current override count
        // (preserves any in-flight edits across re-renders).
        if (m_perSymbolEdit.size() != overrides.size())
            m_perSymbolEdit.resize(overrides.size());
        if (ImGui::BeginTable("PerSymbolCaps",
                              3,
                              ImGuiTableFlags_BordersInnerH |
                              ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Symbol",  ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Cap (USD)", ImGuiTableColumnFlags_WidthFixed, 130.0f);
            ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 90.0f);
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < overrides.size(); ++i) {
                const auto& kv = overrides[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", kv.first.c_str());
                ImGui::TableSetColumnIndex(1);
                // Edit buffer: if empty (first frame after rebuild),
                // seed it with the current cap so the trader can edit
                // in place. After they touch it, the buffer diverges
                // and only an Apply commits it.
                if (m_perSymbolEdit[i].empty()) {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%.2f", kv.second);
                    m_perSymbolEdit[i] = buf;
                }
                char editBuf[32];
                std::snprintf(editBuf, sizeof(editBuf), "%s",
                              m_perSymbolEdit[i].c_str());
                ImGui::PushItemWidth(120);
                if (ImGui::InputText(("##cap_" + kv.first).c_str(),
                                     editBuf, sizeof(editBuf))) {
                    m_perSymbolEdit[i] = editBuf;
                }
                ImGui::PopItemWidth();
                ImGui::TableSetColumnIndex(2);
                ImGui::PushID(("apply_cap_" + kv.first).c_str());
                if (ImGui::SmallButton("Apply")) {
                    double v = parseOrZero(m_perSymbolEdit[i].c_str());
                    m_guard->setMaxOrderNotionalUSDForSymbol(kv.first, v);
                    BTQ_LOG_INFO("RiskLimits: %s cap set to $%.2f",
                                 kv.first.c_str(), v);
                    if (m_persistFn) m_persistFn(*m_guard);
                    m_perSymbolEdit[i].clear();  // re-seed next frame
                }
                ImGui::PopID();
                ImGui::SameLine();
                ImGui::PushID(("clear_cap_" + kv.first).c_str());
                if (ImGui::SmallButton("Clear")) {
                    m_guard->clearMaxOrderNotionalUSDForSymbol(kv.first);
                    BTQ_LOG_INFO("RiskLimits: %s per-symbol cap cleared",
                                 kv.first.c_str());
                    if (m_persistFn) m_persistFn(*m_guard);
                    m_perSymbolEdit[i].clear();
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }
    // Add-row inputs at the bottom. Trader types symbol + cap,
    // hits Add. Empty symbol is ignored (prevents accidental
    // empty-key entries). Cap ≤ 0 also ignored.
    ImGui::Text("Add per-symbol override:");
    ImGui::PushItemWidth(140);
    ImGui::InputText("Symbol##addsym",  m_pendingAddSymbol, sizeof(m_pendingAddSymbol));
    ImGui::SameLine();
    ImGui::InputText("Cap USD##addcap", m_pendingAddCapUSD,  sizeof(m_pendingAddCapUSD));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Add##addcap")) {
        std::string sym = m_pendingAddSymbol;
        double v = parseOrZero(m_pendingAddCapUSD);
        if (!sym.empty() && v > 0.0) {
            m_guard->setMaxOrderNotionalUSDForSymbol(sym, v);
            BTQ_LOG_INFO("RiskLimits: %s per-symbol cap set to $%.2f",
                         sym.c_str(), v);
            if (m_persistFn) m_persistFn(*m_guard);
            m_pendingAddSymbol[0] = '\0';
            m_pendingAddCapUSD[0] = '\0';
        } else {
            BTQ_LOG_WARN("RiskLimits: per-symbol add ignored "
                         "(symbol empty or cap <= 0)");
        }
    }
    ImGui::TextDisabled("(empty symbol or cap ≤ 0 is silently ignored; "
                        "Clear removes an existing override)");

    ImGui::End();
}

} // namespace btquant::ui
