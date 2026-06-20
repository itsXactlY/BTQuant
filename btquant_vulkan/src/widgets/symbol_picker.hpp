#ifndef BTQUANT_SYMBOL_PICKER_HPP
#define BTQUANT_SYMBOL_PICKEL_HPP

#include <functional>
#include <string>
#include <vector>

namespace btquant::ui {

// Symbol picker modal. Opens on demand (Ctrl+P), filters a list of candidate
// symbols by substring (case-insensitive), fires a callback when the user
// selects one. Single-symbol MVP — the callback updates the active symbol
// on the MarketDataProcessor; multi-symbol spine would dispatch the new
// symbol across all widgets.
class SymbolPicker {
public:
    // Available symbols the user can pick from. Default is a short list of
    // common USDT pairs.
    void setCandidates(const std::vector<std::string>& syms) { m_candidates = syms; refresh(); }
    const std::vector<std::string>& candidates() const { return m_candidates; }

    // Currently selected (filter input) symbol — empty if nothing typed.
    const std::string& current() const { return m_filter; }

    // Open / close the modal.
    void setOpen(bool v) { m_open = v; if (v) refresh(); }
    bool isOpen() const   { return m_open; }

    // Selection callback — fired with the picked symbol.
    using SelectFn = std::function<void(const std::string& symbol)>;
    void setSelectFn(SelectFn fn) { m_select = std::move(fn); }

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
    std::string m_filter;
    char m_input[64] = "";
    bool m_open = false;
    int  m_selected = 0;
    SelectFn m_select;
};

} // namespace btquant::ui

#endif
