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

    // Sum of realizedDelta across every fill on disk. Returns 0.0
    // when the journal is empty. Useful as an all-time-P&L
    // readout in the status bar / dashboard ("since install").
    // Sprint #67 — O(N) load; the journal is small enough that
    // streaming isn't worth the complexity yet.
    double totalRealized() const;

    // Per-symbol all-time realized (Sprint #72). Sorted by absolute
    // contribution DESCENDING so the biggest gainers/losers surface
    // first. Mirrors RiskGuard::sessionRealizedBySymbol() but reads
    // from the persisted journal — survives restarts, accumulates
    // across days.
    std::vector<std::pair<std::string, double>>
    realizedBySymbol() const;

    // Per-tag all-time realized (Sprint #73). Sorted by absolute
    // contribution DESCENDING so the biggest gainers/losers surface
    // first. Mirrors realizedBySymbol() but groups by JournalFill::tag
    // instead of symbol — answers "is my scalper-1 strategy net
    // positive over 6 months?" without exporting to CSV.
    //
    // When `includeUntagged=true`, fills with empty tag are aggregated
    // under the synthetic key "__untagged__". When false (default),
    // empty-tag fills are skipped entirely — a trader who never tags
    // anything gets an empty breakdown rather than a misleading
    // 100%-of-P&L entry.
    std::vector<std::pair<std::string, double>>
    realizedByTag(bool includeUntagged = false) const;

    // Per-day realized (Sprint #77). Buckets persisted fills by
    // local-time calendar day (the trader's day, not UTC) and
    // returns one row per day that had at least one fill, sorted by
    // date ASCENDING (oldest first).
    //
    // Format: "YYYY-MM-DD" — sortable as a string and matches the
    // timestamp format used by formatFillsCSV() (#66), so the two
    // can be joined in a downstream tool without translation.
    //
    // Open fills (realized == 0) DO contribute to the bucket — a
    // day with only opens is still a trading day, and the trader
    // wants to see "I traded but didn't close anything" without a
    // second pass. Days with no fills at all are simply absent
    // from the output.
    //
    // Wall-clock time matters: timestamp_us is system_clock::now()
    // at fill time, so localtime_r groups by the trader's local
    // midnight — same convention as RiskGuard's auto-reset (#69).
    std::vector<std::pair<std::string, double>>
    realizedByDay() const;

    // Max drawdown across the persisted history (Sprint #80).
    // Computed on the daily equity curve derived from realizedByDay():
    //
    //   equity[t]   = cumulative sum of daily realized, oldest → t
    //   peak[t]     = max(equity[0..t])
    //   drawdown[t] = peak[t] - equity[t]   (>= 0 by construction)
    //   maxDD       = max over t of drawdown[t]
    //
    // The result captures the worst peak-to-trough decline seen in
    // the persisted history. Fields:
    //
    //   maxDrawdown — positive dollar amount of the worst peak-
    //                 to-trough drop. Zero when the equity curve
    //                 never declines from a prior peak (no
    //                 drawdown yet, or all-time-high right now).
    //   peakDate    — date of the high that preceded the worst
    //                 drawdown ("YYYY-MM-DD"). Empty when
    //                 maxDrawdown == 0.
    //   troughDate  — date of the low that ended the worst
    //                 drawdown ("YYYY-MM-DD"). Empty when
    //                 maxDrawdown == 0.
    //   currentDD   — drawdown as of the most recent day in the
    //                 series. Always >= 0; equals maxDrawdown when
    //                 the worst drawdown is the one we're still
    //                 inside. Zero when equity is at all-time high.
    struct Drawdown {
        double maxDrawdown = 0.0;
        std::string peakDate;     // YYYY-MM-DD or ""
        std::string troughDate;   // YYYY-MM-DD or ""
        double currentDD = 0.0;
    };
    Drawdown maxDrawdown() const;

    // Streak stats (Sprint #82) — consecutive winning/losing
    // round-trips in the persisted history. Walks every fill in
    // loadAll() order, treating each round-trip (realizedDelta
    // != 0) as W/L based on sign. Open fills (realized == 0)
    // don't break or extend a streak.
    //
    // Fields:
    //   currentWinStreak  — number of consecutive wins ending at
    //                       the most recent round-trip. 0 if the
    //                       most recent round-trip was a loss.
    //   currentLossStreak — number of consecutive losses ending at
    //                       the most recent round-trip. 0 if the
    //                       most recent round-trip was a win.
    //   longestWinStreak  — longest run of consecutive wins seen.
    //   longestLossStreak — longest run of consecutive losses seen.
    //
    // Both current* fields can't be > 0 simultaneously — exactly
    // one of them tracks the current state.
    struct Streaks {
        size_t currentWinStreak  = 0;
        size_t currentLossStreak = 0;
        size_t longestWinStreak  = 0;
        size_t longestLossStreak = 0;
    };
    Streaks streaks() const;

    // Risk-adjusted return on the daily series (Sprint #84).
    // Sharpe ratio — mean daily return / stddev of daily returns,
    // annualized by sqrt(252) (trading-days-per-year convention).
    // Sample stddev (Bessel-corrected, n-1) so a single day
    // produces zero stddev and Sharpe = 0 (no signal, not inf).
    //
    // Fields:
    //   dailySharpe       — mean / stddev of the daily return series.
    //   annualizedSharpe  — dailySharpe * sqrt(252). The "headline"
    //                       number — comparable across strategies
    //                       of different frequencies.
    //   meanDailyReturn   — sum(dailyReturns) / N. Dollars/day.
    //   stddevDailyReturn — sample stddev of daily returns. 0 when
    //                       fewer than 2 distinct days.
    //   sampleSize        — number of distinct trading days in the
    //                       series (== size of realizedByDay()).
    //
    // When sampleSize < 2, dailySharpe and annualizedSharpe stay
    // 0 (no division by zero, no NaN). meanDailyReturn is still
    // meaningful (single-day average) and stddevDailyReturn is 0
    // (single observation can't have a meaningful stddev).
    struct Sharpe {
        double dailySharpe       = 0.0;
        double annualizedSharpe  = 0.0;
        double meanDailyReturn   = 0.0;
        double stddevDailyReturn = 0.0;
        size_t sampleSize        = 0;
    };
    Sharpe sharpe() const;

    // All-time aggregate stats (Sprint #75). The journal-wide
    // counterpart to RiskMetrics (#Sprint #46) — same fields, but
    // computed across every persisted fill rather than a rolling
    // in-memory window. Answers "what's my all-time win rate?"
    // without exporting to CSV.
    //
    // Field semantics match RiskMetrics 1:1 so a future "Stats" tab
    // can render both side-by-side without a translation layer.
    //
    //   * fillCount      — every persisted fill, open or close.
    //   * roundTripCount — count of fills with realized != 0
    //                      (closing fills). Wins and losses count
    //                      from here, not from open fills.
    //   * winCount       — roundTripCount where realized > +eps.
    //   * lossCount      — roundTripCount where realized < -eps.
    //   * winRate        — winCount / roundTripCount (0 when no
    //                      rounds yet; not NaN).
    //   * avgWinner      — mean realized across wins (positive).
    //   * avgLoser       — mean realized across losses (negative or
    //                      zero when no losses).
    //   * profitFactor   — gross wins / abs(gross losses). Infinity
    //                      when losses are zero and wins > 0 (mirror
    //                      RiskMetrics sentinel). Zero when no fills.
    //   * expectancy     — mean realized per round-trip fill.
    //   * netRealized    — sum of realized across all fills
    //                      (== totalRealized(); kept here for
    //                      stats-block convenience).
    struct Stats {
        size_t fillCount      = 0;
        size_t roundTripCount = 0;
        size_t winCount       = 0;
        size_t lossCount      = 0;
        double winRate        = 0.0;
        double avgWinner      = 0.0;
        double avgLoser       = 0.0;
        double profitFactor   = 0.0;
        double expectancy     = 0.0;
        double netRealized    = 0.0;
    };
    Stats stats() const;

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
