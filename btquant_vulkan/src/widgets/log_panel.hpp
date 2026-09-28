#ifndef BTQUANT_LOG_PANEL_HPP
#define BTQUANT_LOG_PANEL_HPP

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace btquant::ui {

// Thread-safe ring buffer of log lines + ImGui render panel. Lines can be
// pushed from any thread via the BTQ_LOG_INFO/WARN/ERROR macros below or
// directly via log_push(). The panel renders a scrollable, severity-coloured
// table.
class LogPanel {
public:
    static constexpr size_t kMaxLines = 4096;

    enum Severity : int { Debug = 0, Info = 1, Warn = 2, Error = 3 };

    struct Line {
        uint64_t seq = 0;     // monotonic
        double   t   = 0.0;   // seconds since render-loop start
        int      sev = Info;
        std::string text;
    };

    void render();
    void clear() {
        std::lock_guard<std::mutex> lk(m_mx);
        m_lines.clear();
    }
    size_t lineCount() const {
        std::lock_guard<std::mutex> lk(m_mx);
        return m_lines.size();
    }

    // Push from any thread — locks, trims oldest if over kMaxLines.
    void push(int sev, const std::string& text);

    // Convenience: pull from ImGui's debug log (imgui_internal.h), which
    // is what IM_LOG/ImGui::LogToBuffer uses. Called once per frame from
    // the render loop.
    void drainImGuiLog();

    // Singleton accessor — used by the BTQ_LOG_* macros.
    static LogPanel& instance();

private:
    LogPanel() = default;
    void ringPush(Line&& ln);

    mutable std::mutex m_mx;
    std::deque<Line>   m_lines;
    std::atomic<uint64_t> m_seq{0};
    bool   m_autoScroll   = true;
    bool   m_showDebug    = false;
    int    m_minSevShown  = Debug;
};

// Thread-safe printf-style logger.
void log_v(int sev, const char* fmt, ...);

} // namespace btquant::ui

// Macros — cheap when the level is filtered. (We always log for now, since
// the panel just trims by kMaxLines; filtering is done at render time.)
#define BTQ_LOG_DEBUG(...) ::btquant::ui::log_v(::btquant::ui::LogPanel::Debug, __VA_ARGS__)
#define BTQ_LOG_INFO(...)  ::btquant::ui::log_v(::btquant::ui::LogPanel::Info,  __VA_ARGS__)
#define BTQ_LOG_WARN(...)  ::btquant::ui::log_v(::btquant::ui::LogPanel::Warn,  __VA_ARGS__)
#define BTQ_LOG_ERROR(...) ::btquant::ui::log_v(::btquant::ui::LogPanel::Error, __VA_ARGS__)

#endif
