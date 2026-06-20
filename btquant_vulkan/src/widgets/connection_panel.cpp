#include "connection_panel.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <imgui.h>

#include "../data/market_data_processor.hpp"

namespace btquant::ui {

void ConnectionPanel::render() {
    if (!ImGui::Begin("Connection", nullptr, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    State state = State::Disconnected;
    std::string path = "(not started)";
    std::string sym  = "—";
    uint64_t seq     = 0;
    uint64_t ticks   = 0;
    uint64_t errors  = 0;
    double   latency = -1.0;
    size_t   tradeCount = 0;
    uint64_t newestTs = 0;

    if (m_data) {
        if (m_data->isRunning()) {
            path = m_data->sourcePath();
            sym  = m_data->symbol();
            ticks = m_data->ticksSeen();
            errors = m_data->parseErrors();
            auto snap = m_data->snapshot(1, 0);
            seq = snap.snapshot_seq;
            tradeCount = snap.recent_trades.size();
            if (!snap.recent_trades.empty()) {
                newestTs = snap.recent_trades.front().timestamp;
                auto nowUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                if (nowUs > newestTs) latency = (nowUs - newestTs) / 1000.0;
                state = (path == "/dev/shm/btquant_hotspine") ? State::Live
                                                              : State::Synthetic;
                // Detect actual open: if path doesn't include "hotspine" and
                // ticks is 0, we are still in the post-open synthetic mode.
                if (ticks == 0 && state == State::Synthetic) {
                    // Could be either — keep synthetic.
                }
            } else {
                state = State::Synthetic;
            }
        }
    }

    // Tick-rate EWMA — only when we have a delta.
    double nowSec = 0.0;
    {
        static const auto startT = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::steady_clock::now() - startT;
        nowSec = std::chrono::duration<double>(elapsed).count();
    }
    if (m_lastT > 0.0 && nowSec > m_lastT) {
        double dt = nowSec - m_lastT;
        if (ticks > m_lastTicksSeen) {
            double rate = (ticks - m_lastTicksSeen) / dt;
            if (m_tickRateEwma == 0.0) m_tickRateEwma = rate;
            else                       m_tickRateEwma = m_tickRateEwma * 0.9 + rate * 0.1;
        } else if (m_lastTicksSeen > ticks) {
            m_tickRateEwma = 0.0;  // reset (restart detected)
        }
    }
    m_lastTicksSeen = ticks;
    m_lastSeq       = seq;
    m_lastT         = nowSec;

    // State badge.
    ImVec4 stateCol = state == State::Live       ? ImVec4(0.30f, 0.95f, 0.40f, 1.0f)
                    : state == State::Synthetic  ? ImVec4(1.00f, 0.85f, 0.30f, 1.0f)
                                                 : ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, stateCol);
    ImGui::Text("● %s", stateName(state));
    ImGui::PopStyleColor();

    ImGui::Separator();
    ImGui::Columns(2, "ConnTable", false);
    ImGui::SetColumnWidth(0, 160);

    ImGui::Text("Source path");     ImGui::NextColumn();
    ImGui::TextUnformatted(path.c_str());   ImGui::NextColumn();
    ImGui::Text("Symbol");           ImGui::NextColumn();
    ImGui::TextUnformatted(sym.c_str());    ImGui::NextColumn();
    ImGui::Text("Snapshot seq #");   ImGui::NextColumn();
    ImGui::Text("%lu", (unsigned long)seq); ImGui::NextColumn();
    ImGui::Text("Ticks received");   ImGui::NextColumn();
    ImGui::Text("%lu", (unsigned long)ticks); ImGui::NextColumn();
    ImGui::Text("Parse errors");     ImGui::NextColumn();
    if (errors > 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
    ImGui::Text("%lu", (unsigned long)errors);
    if (errors > 0) ImGui::PopStyleColor(); ImGui::NextColumn();
    ImGui::Text("Tick rate (t/s, EWMA)"); ImGui::NextColumn();
    ImGui::Text("%.1f", m_tickRateEwma);   ImGui::NextColumn();
    ImGui::Text("Trades in buffer"); ImGui::NextColumn();
    ImGui::Text("%zu", tradeCount);        ImGui::NextColumn();
    ImGui::Text("Newest trade ts (μs)"); ImGui::NextColumn();
    ImGui::Text("%lu", (unsigned long)newestTs); ImGui::NextColumn();
    ImGui::Text("Latency estimate"); ImGui::NextColumn();
    if (latency < 0) {
        ImGui::TextDisabled("—");
    } else if (latency > 1000.0) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
        ImGui::Text("%.1f ms (stale)", latency);
        ImGui::PopStyleColor();
    } else {
        ImGui::Text("%.2f ms", latency);
    }
    ImGui::NextColumn();
    ImGui::Columns(1);

    ImGui::Separator();
    ImGui::TextDisabled("Notes: latency = newest_trade_ts → now(). "
                        "Tick rate is EWMA over 0.1 new + 0.9 old. "
                        "Synthetic fallback when spine fails to open.");

    ImGui::End();
}

const char* ConnectionPanel::stateName(State s) const {
    switch (s) {
        case State::Disconnected: return "DISCONNECTED";
        case State::Synthetic:    return "SYNTHETIC FALLBACK";
        case State::Live:         return "LIVE";
    }
    return "?";
}

} // namespace btquant::ui
