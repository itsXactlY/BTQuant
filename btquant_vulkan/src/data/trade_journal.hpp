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
    std::string tag;                // free-form strategy/strategy-id label
                                    // (e.g. "manual", "scalper-1", "arb")
                                    // — empty when the fill wasn't tagged
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

    // ---- CSV export ----

    // Serialize fills to a CSV string. Header line first, then one row
    // per fill, ISO-8601 timestamps, RFC-4180-style quoting (none of
    // the current fields need it, but the hook stays for future
    // extensions like user-supplied tags).
    static std::string formatFillsCSV(const std::vector<JournalFill>& fills);

    // Serialize a tag-filtered subset of fills to a CSV string. Only
    // fills whose tag matches `tag` (or, when `includeUntagged=true`,
    // fills with an empty tag) are included. When `tag` is empty and
    // `includeUntagged=true`, the result equals formatFillsCSV(fills).
    // Static, pure, and easy to test.
    static std::string formatFillsCSVByTag(
        const std::vector<JournalFill>& fills,
        const std::string& tag,
        bool includeUntagged = false);

    // Write every persisted fill to `path` as CSV. Returns true on
    // success, false on any I/O error (and logs the reason). Existing
    // files are overwritten — CSV export is one-shot, not append.
    bool exportCSV(const std::string& path) const;

    // Write a tag-filtered subset of persisted fills to `path` as
    // CSV. `tag` selects the bucket; `includeUntagged` controls
    // whether empty-tag fills are written alongside. Returns true
    // on success, false on I/O error. Existing files overwritten.
    bool exportCSVByTag(const std::string& path,
                        const std::string& tag,
                        bool includeUntagged = false) const;

    // Return the subset of fills whose tag matches `tag` (or whose
    // tag is empty, when `includeUntagged=true`). Loads from disk,
    // filters in-memory — no streaming yet because the journal is
    // expected to fit in RAM.
    std::vector<JournalFill> loadByTag(const std::string& tag,
                                        bool includeUntagged = false) const;

    // ---- Post-hoc fill editing (Sprint #62) ----
    //
    // The journal is conceptually append-only, but in practice the
    // trader occasionally mistypes a tag (e.g. "scaler-1" instead
    // of "scalper-1") and wants to correct it without losing the
    // rest of the session. These methods rewrite the journal file
    // atomically (write to .tmp, rename) so a crash mid-edit doesn't
    // corrupt the file.
    //
    // setTagAt(index, newTag) — edit by 0-based index in loadAll()
    //   order. Out-of-range indices return false; the file is left
    //   untouched.
    //
    // setTagByTimestamp(ts, sym, newTag) — find the first fill
    //   whose timestamp_us and symbol match, then edit. Returns
    //   false if no match (caller logs / surfaces to UI).
    //
    // Both methods are O(N) — load + modify + rewrite. Fine for
    // typical session sizes (hundreds to low-thousands of fills).

    bool setTagAt(size_t index, const std::string& newTag);

    bool setTagByTimestamp(uint64_t timestamp_us,
                           const std::string& symbol,
                           const std::string& newTag);

private:
    std::string m_path;
};

} // namespace btquant

#endif
