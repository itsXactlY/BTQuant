#include "log_panel.hpp"

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <imgui.h>

namespace btquant::ui {

LogPanel& LogPanel::instance() {
    static LogPanel s;
    return s;
}

void log_v(int sev, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    LogPanel::instance().push(sev, buf);
}

void LogPanel::push(int sev, const std::string& text) {
    ringPush(Line{m_seq.fetch_add(1, std::memory_order_relaxed) + 1,
                  0.0, sev, text});
}

void LogPanel::ringPush(Line&& ln) {
    std::lock_guard<std::mutex> lk(m_mx);
    if (m_lines.size() >= kMaxLines) m_lines.pop_front();
    m_lines.push_back(std::move(ln));
}

void LogPanel::render() {
    if (!ImGui::Begin("Log", nullptr, ImGuiWindowFlags_MenuBar)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginMenuBar()) {
        if (ImGui::SmallButton("Clear")) clear();
        ImGui::SameLine();
        ImGui::Checkbox("Auto-scroll", &m_autoScroll);
        ImGui::SameLine();
        ImGui::Checkbox("Debug", &m_showDebug);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        const char* sevNames[] = {"Debug","Info","Warn","Error"};
        ImGui::Combo("Min", &m_minSevShown, sevNames, 4);
        ImGui::SameLine();
        ImGui::TextDisabled("%zu lines", lineCount());
        ImGui::EndMenuBar();
    }
    ImGui::Separator();

    ImGui::BeginChild("log_scroll", ImVec2(0, 0), false,
                      ImGuiWindowFlags_HorizontalScrollbar);

    std::deque<Line> snapshot;
    {
        std::lock_guard<std::mutex> lk(m_mx);
        snapshot = m_lines;
    }
    for (const auto& ln : snapshot) {
        if (ln.sev < m_minSevShown) continue;
        if (ln.sev == Debug && !m_showDebug) continue;
        ImVec4 col = ln.sev == Error ? ImVec4(1.0f, 0.30f, 0.30f, 1.0f)
                   : ln.sev == Warn  ? ImVec4(1.0f, 0.85f, 0.30f, 1.0f)
                   : ln.sev == Info  ? ImGui::GetStyleColorVec4(ImGuiCol_Text)
                                     : ImVec4(0.55f, 0.55f, 0.55f, 1.0f);
        char tag = ln.sev == Error ? 'E' : ln.sev == Warn ? 'W'
                 : ln.sev == Info  ? 'I' : 'D';
        ImGui::TextColored(ImVec4(0.55f,0.55f,0.55f,1.0f), "[%lu]", (unsigned long)ln.seq);
        ImGui::SameLine();
        ImGui::TextColored(col, "%c %s", tag, ln.text.c_str());
    }
    if (m_autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
    ImGui::End();
}

// drainImGuiLog is intentionally a no-op here. ImGui's LogToBuffer captures
// to a caller-owned buffer; the BTQ_LOG_* macros already capture everything
// we need. This method is kept on the API surface for future expansion (e.g.
// parsing Vulkan validation layer output via ImGui::LogToBuffer wrappers).
void LogPanel::drainImGuiLog() {}

} // namespace btquant::ui
