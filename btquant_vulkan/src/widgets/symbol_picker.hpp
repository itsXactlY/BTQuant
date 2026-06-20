#ifndef BTQUANT_SYMBOL_PICKER_HPP
#define BTQUANT_SYMBOL_PICKEL_HPP

#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace btquant::ui {

// Symbol picker modal. Opens on demand (Ctrl+P), filters a list of candidate
// symbols by substring (case-insensitive), fires a callback when the user
// selects one. Single-symbol MVP — the callback updates the active symbol
// on the MarketDataProcessor; multi-symbol spine would dispatch the new
// symbol across all widgets.
//
// Recent history (Sprint #60) — every successful selection is recorded in
// a bounded deque (max 10), deduped, most-recent-first. Rendered as a
// separate "Recent" section at the top of the modal so the trader can
// re-pick a frequently-used symbol with one click instead of typing or
// scrolling the filter list.
class SymbolPicker {
public:
    static constexpr std::size_t kMaxRecent = 10;

    // Available symbols the user can pick from. Default is a short list of
    // common USDT pairs.
    void setCandidates(const std::vector<std::string>& syms) { m_candidates = syms; refresh(); }
    const std::vector<std::string>& candidates() const { return m_candidates; }

    // Currently selected (filter input) symbol — empty if nothing typed.
    const std::string& current() const { return m_filter; }

    // Test-only API: directly set the filter text and refresh the filtered
    // list. The production path is via render() reading the ImGui input,
    // but tests need to exercise the filter without spinning up a context.
    void setFilter(const std::string& f) { m_filter = f; refresh(); }

    // Open / close the modal.
    void setOpen(bool v) { m_open = v; if (v) refresh(); }
    bool isOpen() const   { return m_open; }

    // Selection callback — fired with the picked symbol.
    using SelectFn = std::function<void(const std::string& symbol)>;
    void setSelectFn(SelectFn fn) { m_select = std::move(fn); }

    // Add a symbol to the recent-history deque. Dedupes (re-pushing an
    // existing entry moves it to the front instead of duplicating).
    // Capped at kMaxRecent (10). Empty symbols are silently ignored.
    void addRecent(const std::string& symbol);

    // Recent-history snapshot (most-recent first). Test surface +
    // state.ini persistence path.
    const std::deque<std::string>& recent() const { return m_recent; }
    void setRecent(const std::deque<std::string>& r) { m_recent = r; }
    void clearRecent() { m_recent.clear(); }

    // Render the modal (call once per frame).
    void render();

    // Test accessors.
    const std::vector<std::string>& filtered() const { return m_filtered; }
    size_t filteredCount() const { return m_filtered.size(); }

private:
    void refresh();

    std::vector<std::string> m_candidates = {
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
        "MATIC/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT"
    };
    std::vector<std::string> m_filtered;
    std::deque<std::string>  m_recent;       // Sprint #60: most-recent first
    std::string m_filter;
    char m_input[64] = "";
    bool m_open = false;
    int  m_selected = 0;
    SelectFn m_select;
};

}  // namespace btquant::ui

#endif
