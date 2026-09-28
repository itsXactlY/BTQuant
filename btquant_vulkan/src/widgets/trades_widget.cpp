#include "trades_widget.hpp"
#include "log_panel.hpp"  // BTQ_LOG_INFO / BTQ_LOG_WARN
#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"
#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdio>
#include <fstream>

namespace btquant::ui {

TradesWidget::TradesWidget() = default;
TradesWidget::~TradesWidget() = default;

void TradesWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

void TradesWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }

    ImGui::Begin("Trades", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Controls:");
    static const double filter_min = 0.0, filter_max = 1000.0;
    ImGui::SliderScalar("Min Size Filter", ImGuiDataType_Double, &m_filterSize, &filter_min, &filter_max, "%.2f");
    ImGui::SameLine();
    // CSV export trigger — opens a modal popup for the filename.
    if (ImGui::Button("Export CSV…")) {
        m_exportModalOpen = true;
    }

    // Modal popup for filename entry. Lives inside the Trades window
    // so the user doesn't have to chase a separate floating dialog.
    if (m_exportModalOpen) {
        ImGui::OpenPopup("Export trades to CSV");
        if (ImGui::BeginPopupModal("Export trades to CSV",
                                   &m_exportModalOpen)) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s", m_exportFilename.c_str());
            if (ImGui::InputText("Filename", buf, sizeof(buf))) {
                m_exportFilename = buf;
            }
            ImGui::SameLine();
            if (ImGui::Button("Save")) {
                if (exportCSV(m_exportFilename)) {
                    BTQ_LOG_INFO("exported %zu trades to %s",
                                 snapshotTrades(50).size(),
                                 m_exportFilename.c_str());
                    m_exportModalOpen = false;
                } else {
                    BTQ_LOG_WARN("CSV export failed: %s",
                                 m_exportFilename.c_str());
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                m_exportModalOpen = false;
            }
            ImGui::EndPopup();
        }
    }

    ImGui::Separator();

    // Source: live snapshot OR synthetic fallback.
    std::vector<data::Trade> trades = snapshotTrades(50);
    bool isLive = !trades.empty() && m_data != nullptr;

    ImGui::Text("Recent Trades (%s, count=%zu)", isLive ? "LIVE" : "synthetic", trades.size());
    ImGui::Columns(4, "TradesTable", true);
    ImGui::SetColumnWidth(0, 80);
    ImGui::SetColumnWidth(1, 80);
    ImGui::SetColumnWidth(2, 80);
    ImGui::SetColumnWidth(3, 60);

    ImGui::Text("Time"); ImGui::NextColumn();
    ImGui::Text("Price"); ImGui::NextColumn();
    ImGui::Text("Size"); ImGui::NextColumn();
    ImGui::Text("Side"); ImGui::NextColumn();
    ImGui::Separator();

    for (const auto& trade : trades) {
        if (trade.size < m_filterSize) continue;

        auto timePoint = std::chrono::system_clock::time_point(std::chrono::microseconds(trade.timestamp));
        auto timeT = std::chrono::system_clock::to_time_t(timePoint);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            timePoint.time_since_epoch()) % 1000;
        std::stringstream ss;
        ss << std::put_time(std::localtime(&timeT), "%H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        ImGui::Text("%s", ss.str().c_str());
        ImGui::NextColumn();

        ImGui::Text("%.4f", trade.price);
        ImGui::NextColumn();
        ImGui::Text("%.2f", trade.size);
        ImGui::NextColumn();
        if (trade.isBuy) {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
            ImGui::Text("BUY ");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
            ImGui::Text("SELL");
        }
        ImGui::PopStyleColor();
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
    ImGui::Separator();

    double totalVol = 0, buyVol = 0, sellVol = 0;
    for (const auto& t : trades) {
        totalVol += t.size;
        if (t.isBuy) buyVol += t.size; else sellVol += t.size;
    }
    ImGui::Text("Total: %zu   Volume: %.2f   Buy: %.2f   Sell: %.2f   Delta: %.2f%s",
                trades.size(), totalVol, buyVol, sellVol, buyVol - sellVol,
                isLive ? "" : "   [snap#0]");

    ImGui::End();
}

void TradesWidget::setFilter(double minSize) {
    m_filterSize = minSize;
}

void TradesWidget::reset() {
    m_filterSize = 0;
}

std::vector<data::Trade> TradesWidget::snapshotTrades(size_t maxCount) const {
    std::vector<data::Trade> trades;
    if (m_data) {
        auto snap = m_data->snapshot(maxCount);
        if (snap.snapshot_seq > 0) {
            return snap.recent_trades;  // live snapshot
        }
    }
    // Synthetic fallback when no producer is running — note: this
    // uses function-local statics so each render call would otherwise
    // leak them across widget instances. Use a static map keyed off
    // `this` so two TradesWidget instances don't share state.
    static thread_local std::unordered_map<const TradesWidget*,
                                           std::vector<data::Trade>> fbMap;
    static thread_local std::unordered_map<const TradesWidget*,
                                           std::chrono::steady_clock::time_point> tsMap;
    auto& fallback = fbMap[this];
    auto& lastUpdate = tsMap[this];
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastUpdate).count() > 500) {
        for (int i = 0; i < 3; ++i) {
            data::Trade trade;
            trade.id = fallback.size();
            trade.price = 99.5 + (rand() % 100) / 100.0;
            trade.size  = 10.0 + (rand() % 100);
            trade.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            trade.isBuy = (rand() % 2 == 0);
            fallback.insert(fallback.begin(), trade);
            if (fallback.size() > maxCount) fallback.pop_back();
        }
        lastUpdate = now;
    }
    return fallback;
}

std::string TradesWidget::formatTradesCSV(
        const std::vector<data::Trade>& trades) {
    std::ostringstream os;
    os << "id,timestamp_iso,price,size,side\n";
    for (const auto& t : trades) {
        // ISO-8601 UTC timestamp (microsecond precision).
        auto tp = std::chrono::system_clock::time_point(
                      std::chrono::microseconds(t.timestamp));
        auto timeT = std::chrono::system_clock::to_time_t(tp);
        std::tm tmUtc{};
    #if defined(_WIN32)
        gmtime_s(&tmUtc, &timeT);
    #else
        gmtime_r(&timeT, &tmUtc);
    #endif
        char tsBuf[40];
        std::snprintf(tsBuf, sizeof(tsBuf),
                      "%04d-%02d-%02dT%02d:%02d:%02d.%06lldZ",
                      tmUtc.tm_year + 1900, tmUtc.tm_mon + 1, tmUtc.tm_mday,
                      tmUtc.tm_hour, tmUtc.tm_min, tmUtc.tm_sec,
                      static_cast<long long>(t.timestamp % 1000000));
        // RFC-4180 quoting: wrap fields with commas/quotes in double
        // quotes, double internal quotes. None of the trade fields need
        // quoting today (id/side/timestamp are ASCII-clean, price/size
        // are locale-neutral via snprintf), but we keep the hook for
        // future fields that might.
        auto quoteIfNeeded = [](const std::string& s) -> std::string {
            if (s.find(',') == std::string::npos &&
                s.find('"') == std::string::npos &&
                s.find('\n') == std::string::npos) {
                return s;
            }
            std::string out;
            out.reserve(s.size() + 2);
            out.push_back('"');
            for (char c : s) {
                if (c == '"') out.push_back('"');
                out.push_back(c);
            }
            out.push_back('"');
            return out;
        };
        char priceBuf[32], sizeBuf[32];
        std::snprintf(priceBuf, sizeof(priceBuf), "%.8f", t.price);
        std::snprintf(sizeBuf,  sizeof(sizeBuf),  "%.8f", t.size);
        std::string side = t.isBuy ? "BUY" : "SELL";
        char idBuf[24];
        std::snprintf(idBuf, sizeof(idBuf), "%llu",
                      static_cast<unsigned long long>(t.id));
        os << quoteIfNeeded(idBuf) << ","
           << quoteIfNeeded(tsBuf) << ","
           << quoteIfNeeded(priceBuf) << ","
           << quoteIfNeeded(sizeBuf) << ","
           << quoteIfNeeded(side) << "\n";
    }
    return os.str();
}

bool TradesWidget::exportCSV(const std::string& path) const {
    auto all = const_cast<TradesWidget*>(this)->snapshotTrades(50);
    // Respect the same size filter the user sees on screen — what they
    // see is what they get in the CSV.
    if (m_filterSize > 0.0) {
        std::vector<data::Trade> filtered;
        filtered.reserve(all.size());
        for (const auto& t : all) {
            if (t.size >= m_filterSize) filtered.push_back(t);
        }
        all = std::move(filtered);
    }
    auto csv = formatTradesCSV(all);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    out.write(csv.data(), static_cast<std::streamsize>(csv.size()));
    out.close();
    return !out.fail();
}

}  // namespace btquant::ui
