#ifndef BTQUANT_TRADE_JOURNAL_HPP
#define BTQUANT_TRADE_JOURNAL_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace btquant {

// One persisted fill record. Mirrors the live session state so the
// journal survives restarts.
struct JournalFill {
    uint64_t    timestamp_us = 0;   // monotonic; 0 = unset
    std::string symbol;             // "BTC/USDT"
    bool        isLong = true;      // buy = long, sell = short
    double      qty    = 0.0;       // base units
    double      price  = 0.0;       // fill price
    double      realizedDelta = 0.0;  // P&L realized on this fill (0 on open)
};

// Append-only JSON-lines journal at a fixed path. Each line is a single
// fill record serialized as a flat JSON object. The journal survives
// process restarts — callers can loadAll() on startup to recover session
// history. loadAll() skips malformed lines and continues.
class TradeJournal {
public:
    explicit TradeJournal(const std::string& path);

    // Append one fill. Returns true on success. Creates the parent
    // directory if missing.
    bool append(const JournalFill& r);

    // Load every persisted fill (oldest first). Skips malformed lines
    // and counts them in skippedCount (if non-null).
    std::vector<JournalFill> loadAll(int* skippedCount = nullptr) const;

    // Return the last `n` fills, newest first.
    std::vector<JournalFill> recent(size_t n) const;

    // Total fills currently on disk (cheap; line count).
    size_t count() const;

    // Delete the journal file. Returns true if removed or never existed.
    bool clear();

    const std::string& path() const { return m_path; }

    // ---- Pure serialization (test surface) ----

    // Serialize one fill to a single-line JSON string (no trailing \n).
    static std::string toJsonLine(const JournalFill& r);

    // Parse one JSON line. Returns std::nullopt on malformed input —
    // callers (loadAll) use this to skip bad rows gracefully.
    static std::optional<JournalFill> fromJsonLine(const std::string& line);

private:
    std::string m_path;
};

} // namespace btquant

#endif
