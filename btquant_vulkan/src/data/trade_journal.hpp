#ifndef BTQUANT_TRADE_JOURNAL_HPP
#define BTQUANT_TRADE_JOURNAL_HPP

#include <cstdint>
#include <map>
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
        std::string recoveryDate; // YYYY-MM-DD or "" (Sprint #95)
        size_t      recoveryDays = 0;   // days from trough → ATH or 0 (Sprint #95)
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

    // Per-symbol performance breakdown (Sprint #86). For each
    // symbol with at least one persisted fill, returns the same
    // aggregate fields as stats() (#75) but scoped to that
    // symbol's fills alone. Sorted by absolute realized DESCENDING
    // so the biggest gainers/losers surface first — matches
    // realizedBySymbol() (#72) ordering so the two can be read
    // side-by-side without re-sorting.
    //
    // Field semantics match Stats 1:1. A symbol with only opens
    // (no round-trips) gets zeroed stats but a non-zero realized
    // (== sum of opens, all zero).
    //
    // Cost: O(N) over fills + O(K) over distinct symbols where K
    // is the number of distinct symbols. The map-based aggregation
    // matches realizedBySymbol's pattern so both could be derived
    // from a single pass if performance becomes a concern — for
    // now they're separate to keep the code paths readable.
    struct PerSymbolStats {
        std::string symbol;
        double realized         = 0.0;
        size_t   roundTripCount = 0;
        size_t   winCount       = 0;
        size_t   lossCount      = 0;
        double   winRate        = 0.0;
        double   avgWinner      = 0.0;
        double   avgLoser       = 0.0;
        double   profitFactor   = 0.0;
        double   expectancy     = 0.0;
    };
    std::vector<PerSymbolStats> perSymbolStats() const;

    // Per-tag performance breakdown (Sprint #88). Same shape as
    // perSymbolStats() (#86) but grouped by JournalFill::tag
    // instead of symbol. Answers "is my scalper-1 strategy net
    // positive?" with the full W/L/PF breakdown.
    //
    // When `includeUntagged=true`, fills with empty tag are
    // aggregated under the synthetic key "__untagged__" (matches
    // realizedByTag() — #73). When false (default), untagged
    // fills are skipped — same policy as realizedByTag().
    //
    // Field semantics match PerSymbolStats 1:1 so a future
    // "Per-Strategy" tab can render both with shared code.
    struct PerTagStats {
        std::string tag;
        double realized         = 0.0;
        size_t   roundTripCount = 0;
        size_t   winCount       = 0;
        size_t   lossCount      = 0;
        double   winRate        = 0.0;
        double   avgWinner      = 0.0;
        double   avgLoser       = 0.0;
        double   profitFactor   = 0.0;
        double   expectancy     = 0.0;
    };
    std::vector<PerTagStats> perTagStats(bool includeUntagged = false) const;

    // Per-symbol risk metrics (Sprint #91). For each symbol
    // with at least one persisted fill, computes the worst
    // peak-to-trough decline seen in that symbol's per-day equity
    // curve. Sorted by maxDrawdown DESCENDING so the symbol that
    // caused the worst drop surfaces first.
    //
    // The internal algorithm matches maxDrawdown() (#80) exactly:
    //   equity[t]   = cumulative sum of realized on date t
    //   peak[t]     = max(equity[0..t])
    //   drawdown[t] = peak[t] - equity[t]
    //   maxDD       = max over t of drawdown[t]
    // ... but applied to per-symbol daily series instead of the
    // journal-wide one.
    //
    // Fields:
    //   symbol      — the symbol this entry describes.
    //   fillCount   — number of fills in this symbol (open + close).
    //   maxDrawdown — worst peak-to-trough drop, in dollars.
    //   peakDate    — date of the high that preceded the worst
    //                 drop ("YYYY-MM-DD"); empty when maxDD == 0.
    //   troughDate  — date of the low that ended the worst drop;
    //                 empty when maxDD == 0.
    //   currentDD   — drawdown as of the most recent day in the
    //                 symbol's series. Zero when at symbol-ATH.
    struct PerSymbolDrawdown {
        std::string symbol;
        size_t      fillCount   = 0;
        double      maxDrawdown = 0.0;
        std::string peakDate;     // YYYY-MM-DD or ""
        std::string troughDate;   // YYYY-MM-DD or ""
        std::string recoveryDate; // YYYY-MM-DD or "" (Sprint #95)
        size_t      recoveryDays = 0;   // (Sprint #95)
        double      currentDD   = 0.0;
    };
    std::vector<PerSymbolDrawdown> perSymbolDrawdown() const;

    // Per-tag risk metrics (Sprint #93). For each tag with at
    // least one persisted fill, computes the worst peak-to-
    // trough decline seen in that tag's per-day equity curve.
    // Sorted by maxDrawdown DESCENDING so the worst tag surfaces
    // first — matches "which strategy caused my worst drop?".
    //
    // Same algorithm as maxDrawdown() (#80) / perSymbolDrawdown()
    // (#91) — applies to per-tag daily series instead.
    //
    // `includeUntagged` mirrors perTagStats() (#88): when true,
    // fills with empty tag aggregate under "__untagged__"; when
    // false (default), they're skipped entirely.
    struct PerTagDrawdown {
        std::string tag;
        size_t      fillCount   = 0;
        double      maxDrawdown = 0.0;
        std::string peakDate;     // YYYY-MM-DD or ""
        std::string troughDate;   // YYYY-MM-DD or ""
        std::string recoveryDate; // YYYY-MM-DD or "" (Sprint #95)
        size_t      recoveryDays = 0;   // (Sprint #95)
        double      currentDD   = 0.0;
    };
    std::vector<PerTagDrawdown> perTagDrawdown(
        bool includeUntagged = false) const;

    // Per-tag risk-adjusted return (Sprint #93). For each tag,
    // Sharpe on the daily series — same algorithm as sharpe()
    // (#84) but applied to the tag's own daily series. Sorted by
    // annualized Sharpe DESCENDING.
    //
    // `includeUntagged` mirrors perTagStats() (#88).
    struct PerTagSharpe {
        std::string tag;
        double dailySharpe       = 0.0;
        double annualizedSharpe  = 0.0;
        double meanDailyReturn   = 0.0;
        double stddevDailyReturn = 0.0;
        size_t sampleSize        = 0;
    };
    std::vector<PerTagSharpe> perTagSharpe(
        bool includeUntagged = false) const;

    // Risk-adjusted return normalized by worst drawdown (Sprint
    // #95). Calmar ratio = annualized return / |max drawdown|.
    // Tells the trader "how much return do I get per unit of
    // worst peak-to-trough drop?" — Sharpe penalizes volatility
    // indiscriminately, Calmar only penalizes the bad kind
    // (drawdowns).
    //
    // Annualized return is computed as meanDailyReturn × 252,
    // matching sharpe()'s convention. Max drawdown sourced from
    // maxDrawdown().
    //
    //   annualizedReturn  — mean(daily realized) × 252.
    //                       Same convention as annualized Sharpe.
    //   maxDrawdown       — maxDrawdown().maxDrawdown (positive).
    //   calmarRatio       — annualizedReturn / maxDrawdown.
    //                       Sentinel: 0 when maxDrawdown is 0
    //                       (no DD yet) so the panel can render
    //                       "—" without special-casing.
    //                       Negative when annualizedReturn < 0
    //                       (losing year) and DD > 0.
    struct Calmar {
        double annualizedReturn = 0.0;
        double maxDrawdown      = 0.0;
        double calmarRatio      = 0.0;
    };
    Calmar calmar() const;

    // Per-symbol Calmar (Sprint #97). For each symbol with at
    // least one persisted fill, computes Calmar ratio on the
    // symbol's own daily series — same algorithm as calmar()
    // (#95) but applied to the symbol's per-day equity curve
    // and daily return series.
    //
    // Sorted by calmarRatio DESCENDING so the symbol with the
    // best return-per-unit-DD tops the list.
    //
    // A symbol with maxDrawdown == 0 gets calmarRatio = 0
    // (sentinel; metric undefined). A symbol with negative
    // annualized return gets calmarRatio < 0 (stay-away signal).
    struct PerSymbolCalmar {
        std::string symbol;
        double annualizedReturn = 0.0;
        double maxDrawdown      = 0.0;
        double calmarRatio      = 0.0;
    };
    std::vector<PerSymbolCalmar> perSymbolCalmar() const;

    // Per-tag Calmar (Sprint #97). Per-tag mirror of
    // perSymbolCalmar(). `includeUntagged` matches perTagStats()
    // (#88).
    struct PerTagCalmar {
        std::string tag;
        double annualizedReturn = 0.0;
        double maxDrawdown      = 0.0;
        double calmarRatio      = 0.0;
    };
    std::vector<PerTagCalmar> perTagCalmar(
        bool includeUntagged = false) const;

    // Sortino ratio (Sprint #99). Risk-adjusted return
    // normalized by DOWNSIDE volatility — same idea as Sharpe
    // (#84) but the denominator uses only negative returns, so
    // upside volatility doesn't penalize the score. Sortino
    // answers: "how much return do I get per unit of BAD vol?"
    //
    // Formula (target = 0):
    //   downsideDeviation = sqrt(mean(min(0, r)²))
    //   dailySortino      = mean(daily) / downsideDeviation
    //   annualizedSortino = dailySortino × sqrt(252)
    //
    // Sentinel: when no daily return is negative (every day
    // profitable), downsideDeviation = 0 and Sortino = 0 (panel
    // renders as "∞" via the same convention used for profit
    // factor — the trader reads "no bad days" intuitively).
    struct Sortino {
        double dailySortino       = 0.0;
        double annualizedSortino  = 0.0;
        double meanDailyReturn    = 0.0;
        double downsideDeviation  = 0.0;
        size_t sampleSize         = 0;
    };
    Sortino sortino() const;

    // Per-symbol Sortino (Sprint #99). Per-symbol mirror.
    struct PerSymbolSortino {
        std::string symbol;
        double dailySortino       = 0.0;
        double annualizedSortino  = 0.0;
        double meanDailyReturn    = 0.0;
        double downsideDeviation  = 0.0;
        size_t sampleSize         = 0;
    };
    std::vector<PerSymbolSortino> perSymbolSortino() const;

    // Per-symbol daily stats — heatmap-ready grid (Sprint #102).
    // For each (symbol, date) bucket with at least one fill,
    // computes realized + W/L/round-trip counts. Returns the
    // data shaped for direct rendering as a 2-D grid:
    //   rows = symbols (sorted ASC)
    //   cols = dates   (sorted ASC, chronological)
    //   cells = realized / roundTrips / wins / losses
    //
    // Grid dimensions: symbols.size() × dates.size(). A cell of
    // NaN in the grid means "no fills on this day for this
    // symbol" — the heatmap renders it as a neutral color.
    //
    // Designed for a calendar-style P&L heatmap widget where the
    // trader can see at a glance which symbols performed on
    // which days ("Mondays are bad for SOL", etc.).
    struct DayCell {
        double realized    = 0.0;   // sum of realizedDelta on this day
        size_t roundTrips  = 0;     // round-trip fills (realized != 0)
        size_t wins        = 0;
        size_t losses      = 0;
    };
    struct PerSymbolDayStats {
        std::vector<std::string> symbols;      // rows (sorted ASC)
        std::vector<std::string> dates;        // cols (sorted ASC)
        // Outer index = symbol index, inner = date index.
        // Indexed access: grid[symbolIdx * dates.size() + dateIdx].
        std::vector<DayCell> grid;
    };
    PerSymbolDayStats perSymbolDayStats() const;

    // Single-trade extremes — Sprint #111. Return the single
    // fill with the highest (or lowest) realizedDelta, plus the
    // fill's full identity (timestamp, symbol, tag). Used by
    // the headline summary to surface "my best trade ever
    // was +$2,400 SOL on 2026-06-12 (scalp tag)".
    //
    // Empty journal → { realized=0, ts=0, sym="", tag="" }.
    // The struct deliberately keeps all fields default-
    // initializable so the empty case is just a zero struct.
    struct BestTrade {
        uint64_t    timestamp_us = 0;
        std::string symbol;
        std::string tag;
        double      realized     = 0.0;
    };
    BestTrade bestTrade()  const;   // max realizedDelta
    BestTrade worstTrade() const;   // min realizedDelta
    // Per-symbol best/worst: the same struct but only over
    // fills for that symbol. Returns zero struct when symbol
    // has no fills.
    BestTrade bestTradeBySymbol(const std::string& symbol) const;
    BestTrade worstTradeBySymbol(const std::string& symbol) const;
    // Per-tag best/worst. includeUntagged controls whether
    // untagged fills roll up under "__untagged__".
    BestTrade bestTradeByTag(const std::string& tag,
                             bool includeUntagged = false) const;
    BestTrade worstTradeByTag(const std::string& tag,
                              bool includeUntagged = false) const;

    // CSV export (Sprint #112). Two writers:
    //   exportFillsToCsv(path)  — one row per fill (ts, symbol,
    //                             realized, tag). For traders who
    //                             want to analyze in Excel.
    //   exportStatsToCsv(path)  — two sections (per-symbol +
    //                             per-tag) of summary analytics
    //                             including realized, W/L/PF/
    //                             expectancy, Sharpe/Sortino/
    //                             Calmar. Single file with '#'
    //                             section headers.
    //
    // Returns true on success, false on I/O error. The path's
    // parent dirs are created if they don't exist.
    bool exportFillsToCsv(const std::string& path) const;
    bool exportStatsToCsv(const std::string& path) const;

    // Per-tag daily stats — per-tag mirror. `includeUntagged`
    // matches perTagStats() (#88).
    struct PerTagDayStats {
        std::vector<std::string> tags;
        std::vector<std::string> dates;
        std::vector<DayCell> grid;
    };
    PerTagDayStats perTagDayStats(
        bool includeUntagged = false) const;

    // Equity curve — Sprint #104. Cumulative realized P&L over
    // time, one point per fill (sorted by timestamp ASC). Renders
    // as a sparkline / line chart in EquityCurvePanel — answers
    // "am I making money, and is my equity curve smooth or
    // jagged?"
    //
    // Each point: (timestamp_us, cumulative_realized). The curve
    // is monotonic-non-decreasing on per-fill basis but jumps at
    // each fill. The widget can also overlay a running max line
    // (for drawdown shading) — see equityDrawdownSeries().
    struct EquityPoint {
        uint64_t timestamp_us = 0;     // fill timestamp (microseconds)
        double   realized     = 0.0;   // delta from THIS fill
        double   cumulative   = 0.0;   // sum of all realizedDeltas
                                        // from the start of the journal
                                        // through THIS fill (inclusive)
    };
    std::vector<EquityPoint> equityCurve() const;

    // Drawdown series — Sprint #104. Running peak (high water
    // mark) of the equity curve, paired with the current
    // underwater depth. Answers "how deep was my drawdown at
    // each point in time?" — the widget shades the area between
    // equityCurve and this peak in red.
    //
    // Each point: (timestamp_us, running_peak, drawdown) where
    // drawdown = peak - cumulative (>=0 always).
    struct DrawdownPoint {
        uint64_t timestamp_us = 0;
        double   running_peak  = 0.0;  // max cumulative up to this point
        double   drawdown      = 0.0;  // running_peak - cumulative, >=0
    };
    std::vector<DrawdownPoint> equityDrawdownSeries() const;

    // Drawdown recovery events — Sprint #113. A "recovery
    // event" is a maximal excursion below a previous equity
    // high water mark that has since been recovered (peak →
    // trough → new peak). The journal-wide series is the
    // union of every per-segment run, regardless of depth.
    //
    // Field semantics:
    //   start_ts       — first fill that put us under the
    //                    previous peak (DD entry).
    //   trough_ts      — fill with the lowest cumulative
    //                    realized within this DD event.
    //   trough_depth   — peak - trough_cumulative (>= 0).
    //   end_ts         — first fill that put us back at or
    //                    above the previous peak (recovery).
    //   drawdown_us    — end_ts - start_ts (microseconds).
    //   recovery_us    — end_ts - trough_ts (microseconds).
    //
    // An "unrecovered" drawdown (we're still underwater when
    // the journal ends) is NOT in this series — see
    // currentDrawdown() for that. The series is sorted by
    // trough_depth DESC (worst first) so a quick top-3 shows
    // the trader's worst historical pain.
    struct DrawdownEvent {
        uint64_t start_ts     = 0;
        uint64_t trough_ts    = 0;
        uint64_t end_ts       = 0;
        double   peak_before  = 0.0;   // equity just before DD entry
        double   trough_value = 0.0;   // equity at trough
        double   trough_depth = 0.0;   // peak_before - trough_value
        uint64_t drawdown_us  = 0;     // time from entry to recovery
        uint64_t recovery_us  = 0;     // time from trough to recovery
    };
    std::vector<DrawdownEvent> drawdownRecoveries() const;

    // Per-symbol / per-tag drawdown recovery events. Same
    // shape as drawdownRecoveries() but each event's equity
    // curve is built only from fills of that symbol / tag.
    // Helps answer "which symbol drives my drawdowns?".
    std::vector<DrawdownEvent> drawdownRecoveriesBySymbol(
        const std::string& symbol) const;
    std::vector<DrawdownEvent> drawdownRecoveriesByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Current drawdown — Sprint #113. If we're underwater at
    // the journal's last fill, this captures the in-progress
    // drawdown. The end_ts and recovery_us are zero (not yet
    // recovered). Returns a sentinel with trough_depth == 0
    // when no drawdown is in progress.
    DrawdownEvent currentDrawdown() const;

    // Derived drawdown metrics (Sprint #114). All take a
    // DrawdownEvent by value/const-ref and return a scalar.
    // Pure functions — no journal access needed. Use these
    // for "V-shape vs L-shape" interpretation.
    //
    // recoveryRatio(ev)  — recovery_us / drawdown_us.
    //                      < 1.0 → V-shape (recovered faster
    //                      than we fell). = 1.0 → symmetric.
    //                      > 1.0 → L-shape (took longer to
    //                      recover than to fall). Returns
    //                      +inf if drawdown_us == 0.
    // recoverySpeed(ev)  — trough_depth / recovery_us.
    //                      P&L units recovered per microsec.
    //                      Higher = sharper recovery.
    //                      Returns 0 if recovery_us == 0.
    // maxDepth(events)   — max trough_depth across events
    //                      (handy for "what's my worst DD?").
    // avgDepth(events)   — mean trough_depth.
    // avgRecoveryRatio(events) — geometric mean of recoveryRatio
    //                      across events (ratio of ratios,
    //                      not arithmetic mean — avoids skew
    //                      from extreme values).
    static double recoveryRatio(const DrawdownEvent& ev);
    static double recoverySpeed(const DrawdownEvent& ev);
    static double maxDepth(
        const std::vector<DrawdownEvent>& events);
    static double avgDepth(
        const std::vector<DrawdownEvent>& events);
    static double avgRecoveryRatio(
        const std::vector<DrawdownEvent>& events);

    // Monthly returns — Sprint #115. Bucket journal fills by
    // calendar month (year + month) and compute the realized
    // P&L for each bucket. Mirrors perSymbolDayStats() (#95)
    // in shape.
    //
    // Output is sorted by (year, month) ASC. Month is 1-12
    // (1=Jan). The struct is also used by perSymbol- and
    // perTag- monthly variants below.
    struct MonthlyReturn {
        int     year     = 0;     // e.g. 2026
        int     month    = 0;     // 1-12 (Jan=1)
        double  realized = 0.0;
        size_t  count    = 0;
        size_t  wins     = 0;
        size_t  losses   = 0;
        double  winRate  = 0.0;
    };
    std::vector<MonthlyReturn> monthlyReturns() const;
    std::vector<MonthlyReturn> monthlyReturnsBySymbol(
        const std::string& symbol) const;
    std::vector<MonthlyReturn> monthlyReturnsByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Activity window (Sprint #116). First/last fill timestamp
    // + distinct trading day count for the journal-wide view
    // (or filtered to a symbol/tag).
    //
    // activeTradingDays() — number of distinct YYYY-MM-DD
    //   dates that contain at least one fill. Answers
    //   "how many days have I traded?".
    // firstFillUs() / lastFillUs() — microsecond timestamps
    //   of the chronologically-first / last fill. Returns 0
    //   when the journal is empty.
    // Per-symbol / per-tag variants filter before computing.
    size_t activeTradingDays() const;
    size_t activeTradingDaysBySymbol(
        const std::string& symbol) const;
    size_t activeTradingDaysByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    uint64_t firstFillUs() const;
    uint64_t lastFillUs() const;
    uint64_t firstFillUsBySymbol(const std::string& symbol) const;
    uint64_t lastFillUsBySymbol(const std::string& symbol) const;
    uint64_t firstFillUsByTag(const std::string& tag,
                              bool includeUntagged = false) const;
    uint64_t lastFillUsByTag(const std::string& tag,
                             bool includeUntagged = false) const;

    // Trading sessions (Sprint #117). A "session" is a maximal
    // run of fills where consecutive fills are within
    // `gapMinutes` of each other. When the gap exceeds the
    // threshold, a new session starts. Answer "how many
    // sessions have I had, how long do they typically last,
    // and what's my P&L per session?".
    //
    // The default gap of 30 minutes matches the conventional
    // "lunch break" cut-off — fills within 30 min of each
    // other are part of the same trading stint.
    //
    // Per-session metrics:
    //   start_ts / end_ts — first/last fill in the session
    //   fillCount         — number of fills
    //   realized          — sum of realizedDelta
    //   winRate           — W / (W+L), 0 if no round-trips
    //   maxDD             — peak-to-trough drawdown within
    //                       the session (cumulative)
    //   active_us         — end_ts - start_ts (the time
    //                       spanned by the session, not the
    //                       sum of fill durations)
    struct TradingSession {
        uint64_t start_ts  = 0;
        uint64_t end_ts    = 0;
        size_t   fillCount = 0;
        double   realized  = 0.0;
        double   winRate   = 0.0;
        double   maxDD     = 0.0;
        uint64_t active_us = 0;
    };
    std::vector<TradingSession> sessions(
        int gapMinutes = 30) const;
    std::vector<TradingSession> sessionsBySymbol(
        const std::string& symbol,
        int gapMinutes = 30) const;
    std::vector<TradingSession> sessionsByTag(
        const std::string& tag,
        bool includeUntagged = false,
        int gapMinutes = 30) const;

    // Session aggregates (Sprint #117, derived metrics).
    // Pure functions on session vectors — no journal access.
    static double avgRealized(
        const std::vector<TradingSession>& ss);
    static double avgFillCount(
        const std::vector<TradingSession>& ss);
    static uint64_t avgActiveUs(
        const std::vector<TradingSession>& ss);
    static size_t maxFillCount(
        const std::vector<TradingSession>& ss);
    static double totalRealized(
        const std::vector<TradingSession>& ss);

    // Recovery factor (Sprint #119) — net realized divided
    // by max drawdown. > 2.0 is a strong edge (you make 2x
    // your worst DD per cycle), < 1.0 is grinding (DD bigger
    // than net profit). Returns +inf if maxDD == 0 (no
    // drawdown ever).
    //
    // Per-symbol / per-tag variants compute the factor using
    // only that segment's fills. Journal-wide variant uses
    // the whole journal.
    static double recoveryFactor(
        double netRealized, double maxDrawdown);
    double perSymbolRecoveryFactor(
        const std::string& symbol) const;
    double perTagRecoveryFactor(
        const std::string& tag,
        bool includeUntagged = false) const;
    double journalRecoveryFactor() const;

    // Trading frequency (Sprint #120). Per-symbol / per-tag
    // answers "how often do I trade this asset?" — a concentration
    // diagnostic.
    //
    // tradesPerDay()  — average number of round-trips per
    //                   active trading day. >1 means multiple
    //                   trades per session, <0.2 means sparse.
    //                   Uses activeTradingDays as denominator
    //                   (NOT calendar days) so a trader who
    //                   started recently isn't penalized.
    // avgTimeBetweenTrades_us() — average microsecond gap
    //                   between consecutive fills, sorted by
    //                   timestamp ASC. Returns 0 when <2 fills
    //                   exist.
    // Per-symbol / per-tag variants filter then compute.
    double tradesPerDay() const;
    double tradesPerDayBySymbol(const std::string& symbol) const;
    double tradesPerDayByTag(const std::string& tag,
                             bool includeUntagged = false) const;
    uint64_t avgTimeBetweenTrades_us() const;
    uint64_t avgTimeBetweenTrades_usBySymbol(
        const std::string& symbol) const;
    uint64_t avgTimeBetweenTrades_usByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Kelly criterion (Sprint #121). Optimal fraction of
    // capital to risk per trade given the trader's edge.
    //   K = W - (1 - W) / R
    // where W = win rate and R = payoff ratio
    // (avg winner / avg loser).
    //
    // K > 0 means positive edge; K < 0 means no edge.
    // Most traders use "half-Kelly" (K/2) for safety.
    //
    // Returns 0 when there are no winners or no losers
    // (can't compute without both sides of the payoff).
    //
    // Per-symbol / per-tag variants compute W and R from that
    // segment's fills only.
    static double kellyFraction(
        size_t wins, size_t losses,
        double avgWinner, double avgLoser);
    double kellyFraction() const;
    double perSymbolKellyFraction(const std::string& symbol) const;
    double perTagKellyFraction(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Risk of ruin (Sprint #121). Probability of losing
    // `ruinFraction` of capital (default 50%) over N trades
    // given the trader's W/L stats. Uses the gambler's-ruin
    // approximation:
    //   q = 1 - W (loss probability per trade)
    //   p = W     (win probability per trade)
    //   R = payoff ratio (avg winner / avg loser)
    //   b = R
    //   PoR = ((q/p)^(capital_units)) when p != q
    //   capital_units = ruinFraction / (1 + b) * unit_loss
    //
    // For simplicity we use the canonical form:
    //   PoR = ((1-W)/W)^(capital_units)
    // where capital_units = ruinFraction / avg_loss_relative.
    //
    // Returns 0..1; returns 1.0 when ruin is certain (no
    // edge); returns 0.0 when no ruin possible (always wins).
    static double riskOfRuin(
        size_t wins, size_t losses,
        double ruinFraction = 0.5);
    double riskOfRuin(double ruinFraction = 0.5) const;

    // Streak stats — Sprint #105. Track consecutive W or L
    // round-trips. A streak is a maximal run of Ws or Ls; the
    // "current" streak is the run containing the most recent
    // round-trip. Streaks are based on round-trips (realizedDelta
    // != 0), not raw fills.
    //
    // Cross-method invariant: every round-trip appears in exactly
    // one streak, so the sum of all streak lengths equals
    // stats().roundTrips. (Test 97 verifies this.)
    struct StreakStats {
        size_t currentWinStreak   = 0;  // length of the W-run
                                        // containing the most recent
                                        // round-trip (0 if currently
                                        // on a loss or empty)
        size_t currentLossStreak  = 0;  // mirror, for L
        size_t maxWinStreak       = 0;  // longest W-run in history
        size_t maxLossStreak      = 0;  // longest L-run in history
        size_t totalStreaks       = 0;  // count of distinct W+L runs
        size_t totalWinStreaks    = 0;  // count of W runs
        size_t totalLossStreaks   = 0;  // count of L runs
        // Last 20 streaks (newest first) — used by the UI to
        // render a streak-history strip. Each entry: length,
        // isWin.
        struct RecentStreak {
            size_t length = 0;
            bool   isWin  = false;
        };
        std::vector<RecentStreak> recentStreaks;
    };
    StreakStats streakStats() const;

    // Per-symbol / per-tag streak stats (Sprint #122). The
    // same StreakStats struct, but each streak is built only
    // from fills of that symbol/tag. Helps answer "what's my
    // worst losing streak on BTC?" vs "what's my worst on
    // ETH?" — useful when one symbol has much sharper runs
    // than another.
    StreakStats streakStatsBySymbol(const std::string& symbol) const;
    StreakStats streakStatsByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Cumulative win rate over time (Sprint #123). One
    // point per round-trip (skip ties), sorted by timestamp
    // ASC. Each point: (timestamp_us, winRate, count,
    // cumulativeWins, cumulativeLosses, cumulativeRealized).
    //
    // Useful for visualizing "is my edge sharpening or
    // degrading over time?" — plot winRate as a line and you
    // can see convergence (or lack thereof) to the asymptotic
    // win rate.
    //
    // Per-symbol / per-tag variants filter then compute.
    struct WinRatePoint {
        uint64_t timestamp_us        = 0;
        double   winRate             = 0.0;
        size_t   count               = 0;
        size_t   wins                = 0;
        size_t   losses              = 0;
        double   cumulativeRealized  = 0.0;
    };
    std::vector<WinRatePoint> cumulativeWinRate() const;
    std::vector<WinRatePoint> cumulativeWinRateBySymbol(
        const std::string& symbol) const;
    std::vector<WinRatePoint> cumulativeWinRateByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Rolling profit factor (Sprint #124). For each
    // round-trip at index i ≥ N-1, compute PF over the
    // window [i-N+1, i] (the most recent N trades
    // including this one). One point per eligible fill,
    // sorted by timestamp ASC.
    //
    // Each point: (timestamp_us, profitFactor, count,
    // grossWin, grossLoss, winRate). profitFactor is the
    // gross-win / |gross-loss| ratio over the window.
    // Returns +inf when grossLoss == 0 (no losing trade in
    // window — strong edge); returns 0 when grossWin == 0
    // (all losses in window).
    struct RollingPFPoint {
        uint64_t timestamp_us = 0;
        double   profitFactor = 0.0;
        size_t   count        = 0;
        double   grossWin     = 0.0;
        double   grossLoss    = 0.0;   // negative
        double   winRate      = 0.0;
    };
    std::vector<RollingPFPoint> rollingProfitFactor(
        size_t window = 20) const;
    std::vector<RollingPFPoint> rollingProfitFactorBySymbol(
        const std::string& symbol,
        size_t window = 20) const;
    std::vector<RollingPFPoint> rollingProfitFactorByTag(
        const std::string& tag,
        bool includeUntagged = false,
        size_t window = 20) const;

    // Per-symbol summary snapshot (Sprint #125). All key
    // metrics for a single symbol in one struct — saves the
    // UI from making 10+ separate method calls per symbol.
    //
    // Fields: stats + drawdown + recovery factor + Kelly +
    // frequency. NaN-safe: any field whose computation has
    // insufficient data is set to a documented sentinel
    // (0.0 for scalars, empty vector for series).
    struct SymbolSummary {
        std::string symbol;
        // From perSymbolStats.
        double realized         = 0.0;
        size_t roundTripCount   = 0;
        size_t winCount         = 0;
        size_t lossCount        = 0;
        double winRate          = 0.0;
        double avgWinner        = 0.0;
        double avgLoser         = 0.0;
        double profitFactor     = 0.0;
        double expectancy       = 0.0;
        // From perSymbolDrawdown (single entry, looked up).
        double maxDrawdown      = 0.0;
        std::string recoveryDate;
        size_t      recoveryDays = 0;
        double currentDD        = 0.0;
        // From perSymbolRecoveryFactor (Sprint #119).
        double recoveryFactor   = 0.0;
        // From perSymbolKellyFraction (Sprint #121).
        double kellyFraction    = 0.0;
        // From tradesPerDayBySymbol + activeTradingDaysBySymbol
        // (Sprint #116 + #120).
        size_t activeDays       = 0;
        double tradesPerDay     = 0.0;
        uint64_t firstFillUs    = 0;
        uint64_t lastFillUs     = 0;
        // From perSymbolSharpe (Sprint #91).
        double annualizedSharpe = 0.0;
    };
    SymbolSummary symbolSummary(const std::string& symbol) const;

    // Per-tag summary snapshot (Sprint #126). Mirror of
    // symbolSummary() (#125) but keyed by tag. Same shape
    // (TagSummary); one call gives the UI all key per-tag
    // metrics without 10+ separate method calls.
    struct TagSummary {
        std::string tag;
        // From perTagStats.
        double realized         = 0.0;
        size_t roundTripCount   = 0;
        size_t winCount         = 0;
        size_t lossCount        = 0;
        double winRate          = 0.0;
        double avgWinner        = 0.0;
        double avgLoser         = 0.0;
        double profitFactor     = 0.0;
        double expectancy       = 0.0;
        // From perTagDrawdown (single entry, looked up).
        double maxDrawdown      = 0.0;
        std::string recoveryDate;
        size_t      recoveryDays = 0;
        double currentDD        = 0.0;
        // From perTagRecoveryFactor (Sprint #119).
        double recoveryFactor   = 0.0;
        // From perTagKellyFraction (Sprint #121).
        double kellyFraction    = 0.0;
        // From activeTradingDaysByTag + tradesPerDayByTag.
        size_t activeDays       = 0;
        double tradesPerDay     = 0.0;
        uint64_t firstFillUs    = 0;
        uint64_t lastFillUs     = 0;
        // From perTagSharpe (Sprint #91).
        double annualizedSharpe = 0.0;
    };
    TagSummary tagSummary(const std::string& tag,
                          bool includeUntagged = false) const;

    // Daily streak stats (Sprint #127). Bucket fills by
    // local day, compute each day's net realized, classify
    // W/L (ties = skipped), and walk consecutive days into
    // streaks. Same shape as StreakStats but at DAY
    // granularity instead of fill granularity.
    //
    // Answers "did I have a string of green days?" — far
    // more meaningful psychologically than trade-level W/L
    // runs because daily P&L smooths out intra-day noise.
    //
    // Per-symbol / per-tag variants filter then compute.
    struct DailyStreakStats {
        size_t currentWinStreak  = 0;
        size_t currentLossStreak = 0;
        size_t maxWinStreak      = 0;
        size_t maxLossStreak     = 0;
        size_t totalStreaks      = 0;
        size_t totalWinDays      = 0;
        size_t totalLossDays     = 0;
        size_t totalDays         = 0;   // active trading days
        double totalRealized     = 0.0; // sum across all days
    };
    DailyStreakStats dailyStreakStats() const;
    DailyStreakStats dailyStreakStatsBySymbol(
        const std::string& symbol) const;
    DailyStreakStats dailyStreakStatsByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Realized P&L distribution stats (Sprint #128).
    // Returns the percentiles + min/max/mean/stddev of the
    // round-trip realizedDelta distribution. Ties (==0) are
    // excluded (they're not round-trips).
    //
    // Useful for understanding the SHAPE of the P&L
    // distribution: "is my median win > |median loss|?"
    // (it should be), "what's my worst 10% outcome?" (the
    // p10), "how fat are my tails?" (p90-p50 vs p50-p10).
    struct PnLDistribution {
        size_t count     = 0;
        double min       = 0.0;
        double max       = 0.0;
        double mean      = 0.0;
        double stddev    = 0.0;
        double p10       = 0.0;   // 10th percentile
        double p25       = 0.0;   // 25th percentile
        double p50       = 0.0;   // median
        double p75       = 0.0;   // 75th percentile
        double p90       = 0.0;   // 90th percentile
    };
    PnLDistribution pnlDistribution() const;
    PnLDistribution pnlDistributionBySymbol(
        const std::string& symbol) const;
    PnLDistribution pnlDistributionByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Drawdown recovery-time distribution (Sprint #129).
    // For every completed drawdown, bucket its recovery
    // duration into bands and count. Useful for "how long
    // does it usually take me to recover?" — a key piece of
    // risk-of-ruin intuition.
    //
    // Buckets (in microseconds):
    //   <  1 min        same_minute
    //   <  1 hour       under_1h
    //   <  1 day        under_1d
    //   <  1 week       under_1w
    //   <  1 month      under_1mo
    //   >= 1 month      over_1mo
    //
    // Also returns totalDrawdowns (count of all recovered DD
    // events) and avgRecoveryDays (mean recovery time in
    // days, only over recovered events).
    struct DDRecoveryDistribution {
        size_t totalDrawdowns = 0;
        size_t sameMinute     = 0;
        size_t under1h        = 0;
        size_t under1d        = 0;
        size_t under1w        = 0;
        size_t under1mo       = 0;
        size_t over1mo        = 0;
        double avgRecoveryDays = 0.0;
        uint64_t maxRecoveryUs = 0;
    };
    DDRecoveryDistribution ddRecoveryDistribution() const;
    DDRecoveryDistribution ddRecoveryDistributionBySymbol(
        const std::string& symbol) const;
    DDRecoveryDistribution ddRecoveryDistributionByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Drawdown depth distribution (Sprint #130). For every
    // COMPLETED drawdown (recovered), bucket its peak-to-
    // trough depth into ranges. Answers "how severe do my
    // drawdowns usually get?" — complements the recovery-time
    // distribution (#129).
    //
    // Buckets (in absolute depth):
    //   <  50          small   (intraday noise)
    //   <  100         minor
    //   <  500         moderate
    //   <  1000        large
    //   <  5000        severe
    //   >= 5000        catastrophic
    //
    // Also returns avgDepth (mean across all completed DDs)
    // and maxDepth (the worst DD ever).
    struct DDDepthDistribution {
        size_t totalDrawdowns = 0;
        size_t small          = 0;
        size_t minor          = 0;
        size_t moderate       = 0;
        size_t large          = 0;
        size_t severe         = 0;
        size_t catastrophic   = 0;
        double avgDepth       = 0.0;
        double maxDepth       = 0.0;
    };
    DDDepthDistribution ddDepthDistribution() const;
    DDDepthDistribution ddDepthDistributionBySymbol(
        const std::string& symbol) const;
    DDDepthDistribution ddDepthDistributionByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Drawdown duration stats (Sprint #131). For every
    // COMPLETED drawdown, compute the time spent UNDERWATER
    // (from peak → trough). Different from recovery_us
    // (#129): duration is the descent, recovery is the climb.
    //
    // Aggregates over all completed DDs:
    //   - totalDrawdowns, avgDurationDays, maxDurationDays,
    //     totalDurationDays.
    // Per-symbol / per-tag variants.
    struct DDDurationStats {
        size_t  totalDrawdowns  = 0;
        double  avgDurationDays = 0.0;
        double  maxDurationDays = 0.0;
        double  totalDurationDays = 0.0;
    };
    DDDurationStats ddDurationStats() const;
    DDDurationStats ddDurationStatsBySymbol(
        const std::string& symbol) const;
    DDDurationStats ddDurationStatsByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Equity curve annotations (Sprint #132). Significant
    // events on the equity curve that the UI can overlay as
    // labels: max DD start, max DD recovery, every recovered
    // DD (start/end/depth), best single-day gain, worst
    // single-day loss.
    //
    // Sorted by timestamp ASC. Each event has a `kind` enum
    // + a human-readable label + the timestamp. The UI can
    // pick which kinds to display.
    enum class AnnotationKind {
        DDStart,          // peak before a drawdown
        DDEnd,            // recovery back to a previous peak
        MaxDDStart,       // the deepest DD's peak
        MaxDDEnd,         // deepest DD's recovery
        BestDay,          // day with highest positive P&L
        WorstDay,         // day with lowest negative P&L
        EquityHigh,       // new equity high water mark
        EquityLow,        // equity local low (not in DD)
    };
    struct Annotation {
        AnnotationKind kind      = AnnotationKind::DDStart;
        uint64_t       timestamp_us = 0;
        std::string    label;
        double         value = 0.0;  // equity at that point
    };
    std::vector<Annotation> equityAnnotations() const;

    // Per-symbol / per-tag annotation sets (Sprint #133).
    // Same Annotation struct, but only includes events
    // drawn from that symbol's/tag's fill sequence. The
    // MaxDDStart/MaxDDEnd for a symbol is the worst DD
    // experienced by that symbol alone.
    std::vector<Annotation> equityAnnotationsBySymbol(
        const std::string& symbol) const;
    std::vector<Annotation> equityAnnotationsByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Sliding-window Sharpe (Sprint #134). For each
    // round-trip at index i ≥ N-1, compute the Sharpe
    // (mean / stddev) over the window [i-N+1, i] using
    // the round-trip realizedDelta values as returns.
    //
    // Returns empty vector if rt.size() < window. One
    // point per eligible fill, sorted by timestamp ASC.
    struct WindowSharpePoint {
        uint64_t timestamp_us = 0;
        double   sharpe        = 0.0;
        double   mean          = 0.0;
        double   stddev        = 0.0;
        size_t   count         = 0;
    };
    std::vector<WindowSharpePoint> rollingWindowSharpe(
        size_t window = 30) const;
    std::vector<WindowSharpePoint> rollingWindowSharpeBySymbol(
        const std::string& symbol,
        size_t window = 30) const;
    std::vector<WindowSharpePoint> rollingWindowSharpeByTag(
        const std::string& tag,
        bool includeUntagged = false,
        size_t window = 30) const;

    // Composite risk score (Sprint #135). Single 0-100
    // number combining Sharpe, drawdown, win rate, and
    // payoff into one overall edge-quality metric.
    //
    // Sub-scores (each 0-100):
    //   - sharpeScore  : annualized Sharpe normalized.
    //       Sharpe 0 → 50 (neutral), 2 → 100, -1 → 25.
    //   - drawdownScore: inverse max DD normalized.
    //       maxDD 0 → 100, maxDD 1000 → 50, maxDD 10000 → 0.
    //   - winRateScore : win rate × 100 (50% → 50, 75% → 75).
    //   - payoffScore  : min(100, |avgW/avgL| × 50).
    //       |W/L| 1 → 50 (break-even), 2 → 100.
    //
    // Overall = 0.30*sharpe + 0.30*drawdown + 0.20*winRate
    //         + 0.20*payoff.
    //
    // Journal-wide + per-symbol + per-tag variants.
    struct RiskScore {
        double overall      = 0.0;
        double sharpeScore  = 0.0;
        double drawdownScore= 0.0;
        double winRateScore = 0.0;
        double payoffScore  = 0.0;
    };
    RiskScore riskScore() const;
    RiskScore riskScoreBySymbol(const std::string& symbol) const;
    RiskScore riskScoreByTag(const std::string& tag,
                              bool includeUntagged = false) const;

    // Symbol concentration risk (Sprint #136). How much of
    // the journal's total realized P&L comes from each
    // symbol? Concentration = risk: a journal where 80% of
    // profit comes from one symbol has hidden fragility.
    //
    // Returns the symbols sorted by contribution DESC,
    // with absolute and relative share. Also returns the
    // top symbol's share as a quick "how concentrated am I?"
    // number.
    struct SymbolShare {
        std::string symbol;
        double      realized     = 0.0;
        double      share        = 0.0;  // 0..1 of |total|
        double      cumShare     = 0.0;  // running cum share
    };
    std::vector<SymbolShare> symbolConcentration() const;

    // Herfindahl-Hirschman Index (Sprint #137). Single
    // 0-1 number summarizing concentration. = sum of
    // squared shares (where shares are over |realized|).
    //
    // Interpretation:
    //   0.0  — perfectly diversified (infinite symbols)
    //   0.25 — moderate concentration (4 equal symbols)
    //   0.5  — high concentration (2 equal symbols)
    //   1.0  — single-symbol dependency
    //
    // Per-symbol and per-tag variants — same metric on
    // different categorical axes.
    double concentrationHHI() const;
    double concentrationHHIByTag(bool includeUntagged = false) const;

    // Trade-size analysis (Sprint #138). For a given
    // segment (symbol or tag), summarize the distribution
    // of round-trip sizes — useful for "am I sizing BTC
    // differently from ETH?"
    struct TradeSizeStats {
        size_t   roundTripCount = 0;
        double   meanAbs        = 0.0;   // mean |realized|
        double   medianAbs      = 0.0;   // p50 of |realized|
        double   p90Abs         = 0.0;   // p90 of |realized|
        double   maxAbs         = 0.0;   // max |realized|
        double   meanWin        = 0.0;   // mean winner
        double   meanLoss       = 0.0;   // mean |loser|
        double   totalWinSize   = 0.0;   // sum of winners
        double   totalLossSize  = 0.0;   // sum of |losers|
    };
    TradeSizeStats tradeSizeStats() const;
    TradeSizeStats tradeSizeStatsBySymbol(
        const std::string& symbol) const;
    TradeSizeStats tradeSizeStatsByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Equity curve volatility (Sprint #139). Rolling
    // stddev of the equity curve values, sampled at the
    // fill-granularity (one point per round-trip). High
    // values = choppy equity curve; low values = smooth
    // growth.
    //
    // Useful as a "smoothness" diagnostic that complements
    // Sharpe (which is return/risk normalized).
    //
    // Returns empty vector if equity.size() < window.
    struct EquityVolPoint {
        uint64_t timestamp_us = 0;
        double   equityValue   = 0.0;  // cum at this point
        double   rollingStddev = 0.0;  // stddev of last N
        size_t   count         = 0;
    };
    std::vector<EquityVolPoint> equityVolatility(
        size_t window = 30) const;

    // Per-symbol/per-tag equity volatility (Sprint #140).
    // Same shape as equityVolatility() (#139) but applied
    // to the segment's equity curve. Answers "is BTC's
    // equity curve smoother or choppier than ETH's?".
    std::vector<EquityVolPoint> equityVolatilityBySymbol(
        const std::string& symbol,
        size_t window = 30) const;
    std::vector<EquityVolPoint> equityVolatilityByTag(
        const std::string& tag,
        bool includeUntagged = false,
        size_t window = 30) const;

    // Win-rate confidence interval (Sprint #141). Wilson
    // score interval — better-behaved than normal
    // approximation for small samples and edge cases (p=0
    // or p=1).
    //
    // Returns lower/upper bounds on the true win rate at
    // 95% confidence. Sample size n = wins + losses.
    // Zero-realized fills (rounding artefacts) are excluded.
    struct WinRateCI {
        double observed    = 0.0;
        double lower95     = 0.0;
        double upper95     = 0.0;
        size_t wins        = 0;
        size_t losses      = 0;
        size_t total       = 0;
    };
    WinRateCI winRateCI() const;
    WinRateCI winRateCIBySymbol(const std::string& symbol) const;
    WinRateCI winRateCIByTag(const std::string& tag,
                              bool includeUntagged = false) const;

    // Win rate by trade-size bucket (Sprint #142). For each
    // size bucket, compute the win rate. Answers "do my big
    // trades win more often than my small trades?"
    //
    // Buckets (in |realized|):
    //   tiny      <  50
    //   small     <  100
    //   medium    <  500
    //   large     <  1000
    //   huge      <  5000
    //   massive   >= 5000
    struct SizeBucketWR {
        size_t total      = 0;
        size_t wins       = 0;
        size_t losses     = 0;
        double winRate    = 0.0;
        double meanAbs    = 0.0;
    };
    struct WinRateBySize {
        SizeBucketWR tiny;
        SizeBucketWR small;
        SizeBucketWR medium;
        SizeBucketWR large;
        SizeBucketWR huge;
        SizeBucketWR massive;
    };
    WinRateBySize winRateBySize() const;
    WinRateBySize winRateBySizeBySymbol(
        const std::string& symbol) const;
    WinRateBySize winRateBySizeByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Equity curve rate of change (Sprint #143). Rolling
    // linear-regression slope over the last N equity
    // values. Positive slope = accelerating equity, zero
    // = flat, negative = decelerating.
    //
    // Uses simple OLS: slope = (N*Σxy - Σx*Σy) /
    // (N*Σx² - (Σx)²).  Time index i is the x-value.
    //
    // Returns empty vector if equity.size() < window.
    struct EquitySlopePoint {
        uint64_t timestamp_us = 0;
        double   equityValue   = 0.0;
        double   slope         = 0.0;  // per-fill-step
        size_t   count         = 0;
    };
    std::vector<EquitySlopePoint> equityRateOfChange(
        size_t window = 30) const;

    // Per-segment equity rate of change (Sprint #144).
    // Same shape as equityRateOfChange (#143) but applied
    // to the segment's filtered equity curve. Answers
    // "is BTC's edge accelerating or decelerating?"
    std::vector<EquitySlopePoint> equityRateOfChangeBySymbol(
        const std::string& symbol,
        size_t window = 30) const;
    std::vector<EquitySlopePoint> equityRateOfChangeByTag(
        const std::string& tag,
        bool includeUntagged = false,
        size_t window = 30) const;

    // All-symbol summaries (Sprint #145). Returns
    // symbolSummary(symbol) for every symbol in the
    // journal, sorted by realized DESC. One call replaces
    // N per-symbol calls.
    std::vector<SymbolSummary> allSymbolSummaries() const;

    // All-tag summaries (Sprint #145). Same shape as
    // allSymbolSummaries() but for tags. __untagged__
    // included as the synthetic key for empty-tag fills.
    std::vector<TagSummary> allTagSummaries(
        bool includeUntagged = true) const;

    // All-time top trades (Sprint #146). Returns the top N
    // winning round-trips and top N losing round-trips
    // across the entire journal. Useful for "what's my
    // single biggest win/loss?"
    std::vector<BestTrade> topWinners(size_t n = 5) const;
    std::vector<BestTrade> topLosers(size_t n = 5) const;

    // Per-segment top trades (Sprint #147). Same shape as
    // topWinners()/topLosers() (#146) but filtered to a
    // single symbol or tag. Useful for "what were my
    // biggest BTC wins?" or "biggest scalp losses?"
    std::vector<BestTrade> topWinnersBySymbol(
        const std::string& symbol, size_t n = 5) const;
    std::vector<BestTrade> topLosersBySymbol(
        const std::string& symbol, size_t n = 5) const;
    std::vector<BestTrade> topWinnersByTag(
        const std::string& tag, bool includeUntagged = false,
        size_t n = 5) const;
    std::vector<BestTrade> topLosersByTag(
        const std::string& tag, bool includeUntagged = false,
        size_t n = 5) const;

    // Symbol-symbol correlation (Sprint #148). Pearson
    // correlation of per-day realized between two symbols.
    // Returns NaN (via `valid=false`) when either symbol
    // has < 2 fills or fewer than 2 matched days.
    //
    // High positive correlation (≈1.0): symbols move
    // together — no diversification benefit.
    // Negative correlation (<0): symbols move opposite —
    // good for hedging.
    // Near zero (±0.1): independent.
    struct SymbolCorrelation {
        double correlation = 0.0;
        size_t fillsA       = 0;
        size_t fillsB       = 0;
        size_t matchedDays  = 0;
        bool   valid        = false;
    };
    SymbolCorrelation symbolSymbolCorrelation(
        const std::string& symA,
        const std::string& symB) const;

    // All-symbol correlation matrix (Sprint #149).
    // Returns correlations for every (symA, symB) pair
    // where symA < symB alphabetically (avoids duplicates).
    // Useful for a heatmap visualization.
    struct CorrelationMatrixEntry {
        std::string symA;
        std::string symB;
        double      correlation = 0.0;
        size_t      matchedDays = 0;
        bool        valid       = false;
    };
    std::vector<CorrelationMatrixEntry>
    allSymbolCorrelations() const;

    // Fill interval distribution (Sprint #150). For
    // consecutive fills in the journal, compute the time
    // gap between them. Returns a PnLDistribution-style
    // summary (mean, median, p90, max) of the gaps in
    // microseconds.
    //
    // Answers "is my trading fast or slow?" — small mean
    // gap = high-frequency, large gap = patient.
    //
    // No-op (zeros) for fills.size() < 2.
    PnLDistribution fillIntervalStats() const;
    PnLDistribution fillIntervalStatsBySymbol(
        const std::string& symbol) const;
    PnLDistribution fillIntervalStatsByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // All-tag correlation matrix (Sprint #151). Mirror of
    // allSymbolCorrelations() (#149) for tags. Pairwise
    // correlation between every distinct tag in the
    // journal. Useful for strategy diversification
    // analysis.
    std::vector<CorrelationMatrixEntry>
    allTagCorrelations(bool includeUntagged = true) const;

    // Weekly win rate (Sprint #152). For each ISO week
    // (year + week-of-year) with at least one round-trip,
    // returns the win rate and trade count. Answers
    // "did I have a winning week 23 vs losing week 24?"
    struct WeeklyWinRate {
        int     year    = 0;
        int     week    = 0;  // ISO week 1-53
        size_t  total   = 0;
        size_t  wins    = 0;
        double  winRate = 0.0;
        double  realized = 0.0;
    };
    std::vector<WeeklyWinRate> weeklyWinRate() const;

    // Per-segment weekly win rate (Sprint #153). Same
    // shape as weeklyWinRate() (#152) but filtered to a
    // single symbol or tag.
    std::vector<WeeklyWinRate> weeklyWinRateBySymbol(
        const std::string& symbol) const;
    std::vector<WeeklyWinRate> weeklyWinRateByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Performance snapshot (Sprint #154). All key metrics
    // for the last N calendar days of trading (or last N
    // fills if byFillCount=true). Useful for "what's my
    // recent edge like?" without needing to call 20+
    // individual methods.
    struct PerformanceSnapshot {
        size_t   totalFills  = 0;
        size_t   wins        = 0;
        size_t   losses      = 0;
        double   realized    = 0.0;
        double   winRate     = 0.0;
        double   profitFactor = 0.0;
        double   sharpe      = 0.0;   // sample stddev Sharpe
        double   maxDD       = 0.0;   // max peak-trough within window
        double   activeDays  = 0.0;  // distinct days in window
        uint64_t startUs     = 0;
        uint64_t endUs       = 0;
    };
    PerformanceSnapshot recentPerformance(
        size_t lastDays = 30,
        bool byFillCount = false,
        size_t fillCount = 30) const;

    // Per-segment recent performance (Sprint #155). Same
    // shape as recentPerformance (#154) but filtered to
    // a single symbol or tag.
    PerformanceSnapshot recentPerformanceBySymbol(
        const std::string& symbol,
        size_t lastDays = 30,
        bool byFillCount = false,
        size_t fillCount = 30) const;
    PerformanceSnapshot recentPerformanceByTag(
        const std::string& tag,
        bool includeUntagged = false,
        size_t lastDays = 30,
        bool byFillCount = false,
        size_t fillCount = 30) const;

    // Per-segment drawdown summary (Sprint #156).
    // Aggregates drawdown statistics for a single symbol
    // or tag in one struct: count, mean depth, max depth,
    // mean duration (in days), longest duration (days),
    // mean recovery ratio (L/V shape indicator).
    struct SegmentDrawdownStats {
        size_t  count           = 0;
        double  meanDepth       = 0.0;
        double  maxDepth        = 0.0;
        double  meanDrawdownDays = 0.0;
        double  maxDrawdownDays  = 0.0;
        double  meanRecoveryRatio = 0.0;  // rec / desc
    };
    SegmentDrawdownStats segmentDrawdownStatsBySymbol(
        const std::string& symbol) const;
    SegmentDrawdownStats segmentDrawdownStatsByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Cumulative gross-profit / gross-loss series
    // (Sprint #157). One point per round-trip. Tracks
    // running totals of gross wins and |gross losses|.
    // Useful for "how much of my edge came from wins vs
    // losses paid?"
    struct GrossPoint {
        uint64_t timestamp_us = 0;
        double   cumGrossWin  = 0.0;
        double   cumGrossLoss = 0.0;
        double   netRealized  = 0.0;
        size_t   count        = 0;
    };
    std::vector<GrossPoint> cumulativeGrossSeries() const;

    // Per-segment cumulative gross series (Sprint #158).
    // Same shape as cumulativeGrossSeries (#157) but
    // filtered to a single symbol or tag.
    std::vector<GrossPoint> cumulativeGrossSeriesBySymbol(
        const std::string& symbol) const;
    std::vector<GrossPoint> cumulativeGrossSeriesByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // All-segment recent performance (Sprint #159).
    // Returns PerformanceSnapshot for every symbol in the
    // journal, sorted by realized DESC. One call replaces
    // N recentPerformanceBySymbol() calls.
    std::vector<PerformanceSnapshot>
    allRecentPerformance(size_t lastDays = 30,
                         bool byFillCount = false,
                         size_t fillCount = 30) const;
    // Per-tag version (Sprint #159). __untagged__
    // included as synthetic key for empty-tag fills.
    std::vector<PerformanceSnapshot>
    allRecentPerformanceByTag(size_t lastDays = 30,
                              bool byFillCount = false,
                              size_t fillCount = 30) const;

    // Monthly drawdown heatmap (Sprint #160). For each
    // (year, month) with at least one trading day, returns
    // the max DD depth observed during that month. Useful
    // for "when do my drawdowns happen?" — heatmap viz.
    struct MonthlyMaxDD {
        int    year    = 0;
        int    month   = 0;   // 1-12
        double maxDD   = 0.0;
        size_t days    = 0;
    };
    std::vector<MonthlyMaxDD> monthlyMaxDrawdown() const;

    // Per-segment monthly max DD (Sprint #162). Same
    // shape as monthlyMaxDrawdown (#160) but filtered to
    // a single symbol or tag.
    std::vector<MonthlyMaxDD> monthlyMaxDrawdownBySymbol(
        const std::string& symbol) const;
    std::vector<MonthlyMaxDD> monthlyMaxDrawdownByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Top DD-prone symbols (Sprint #163). Bulk helper:
    // returns symbols sorted by their max DD depth DESC.
    // Useful for "which symbols give me the most pain?"
    // panel section.
    struct DDSymbolEntry {
        std::string symbol;
        double      maxDD        = 0.0;
        double      recoveryFactor = 0.0;
        size_t      ddCount      = 0;
    };
    std::vector<DDSymbolEntry>
    topDDProneSymbols(size_t n = 10) const;
    std::vector<DDSymbolEntry>
    topDDProneTags(size_t n = 10,
                   bool includeUntagged = true) const;

    // All-segment DD stats (Sprint #164). Bulk variant of
    // segmentDrawdownStatsBySymbol/ByTag (#156). Returns
    // SegmentDrawdownStats for every symbol/tag in one
    // call, sorted by maxDepth DESC.
    std::vector<SegmentDrawdownStats>
    allSegmentDrawdownStats() const;
    std::vector<SegmentDrawdownStats>
    allSegmentDrawdownStatsByTag(
        bool includeUntagged = true) const;

    // All DD events chronological (Sprint #165). Returns
    // the combined list of all DD events from every
    // symbol/tag in chronological order. Each event is
    // tagged with the segment that produced it (added as
    // a string field in DrawdownEvent or via a wrapper).
    struct DrawdownEventExt : DrawdownEvent {
        std::string segment;  // symbol or tag name
    };
    std::vector<DrawdownEventExt>
    allDrawdownRecoveries() const;
    std::vector<DrawdownEventExt>
    allDrawdownRecoveriesByTag(
        bool includeUntagged = true) const;

    // DD contribution per segment (Sprint #166). For each
    // segment, fraction of total journal-wide max DD that
    // it contributed. Returns sorted DESC by
    // contribution share.
    struct DDContribution {
        std::string segment;
        double      segmentMaxDD  = 0.0;
        double      contribution  = 0.0;  // share [0,1]
    };
    std::vector<DDContribution> ddContributionBySymbol() const;
    std::vector<DDContribution> ddContributionByTag(
        bool includeUntagged = true) const;

    // Profit contribution per segment (Sprint #167). Mirror
    // of ddContributionBySymbol (#166) but using realized
    // P&L. Answers "which segment drives my profit?"
    struct ProfitContribution {
        std::string segment;
        double      segmentRealized = 0.0;
        double      contribution     = 0.0;
    };
    std::vector<ProfitContribution>
    profitContributionBySymbol() const;
    std::vector<ProfitContribution>
    profitContributionByTag(
        bool includeUntagged = true) const;

    // Risk-Efficiency per segment (Sprint #168). For each
    // segment, ratio of profit contribution to DD
    // contribution. Higher = more profit per unit of pain.
    // Returns sorted DESC by efficiency score.
    struct RiskEfficiency {
        std::string segment;
        double      profitShare     = 0.0;
        double      ddShare         = 0.0;
        double      efficiency      = 0.0;
        // raw = profitShare / ddShare when both > 0.
        // 0.0 if either is 0 (or they have opposite signs).
    };
    std::vector<RiskEfficiency> riskEfficiencyBySymbol() const;
    std::vector<RiskEfficiency> riskEfficiencyByTag(
        bool includeUntagged = true) const;

    // Composite risk-adjusted metrics bundle (Sprint #169).
    // Single method that returns Sharpe, Sortino, Calmar
    // in one struct. The UI panel header renders all three
    // in a single row, so this saves multiple method calls.
    struct RiskAdjustedBundle {
        double sharpe  = 0.0;   // mean / stddev of returns
        double sortino = 0.0;   // mean / downside dev
        double calmar  = 0.0;   // annual return / max DD
        double omega   = 0.0;   // prob gain / prob loss
        size_t returns = 0;     // count of round-trips used
    };
    RiskAdjustedBundle riskAdjustedBundle() const;
    RiskAdjustedBundle riskAdjustedBundleBySymbol(
        const std::string& symbol) const;
    RiskAdjustedBundle riskAdjustedBundleByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // All-segment risk-adjusted bundle (Sprint #170).
    // Bulk variant of riskAdjustedBundle (#169).
    struct SegmentBundleEntry {
        std::string     segment;
        RiskAdjustedBundle bundle;
    };
    std::vector<SegmentBundleEntry>
    allSegmentRiskAdjustedBundle() const;
    std::vector<SegmentBundleEntry>
    allSegmentRiskAdjustedBundleByTag(
        bool includeUntagged = true) const;

    // Best/worst sessions of all time (Sprint #171).
    // Top N trading sessions by realized P&L (ASC for worst,
    // DESC for best). Answers "what's my best afternoon
    // ever?" and "what's my worst morning ever?"
    std::vector<TradingSession>
    topSessions(size_t n = 5,
                size_t gapMinutes = 30) const;
    std::vector<TradingSession>
    worstSessions(size_t n = 5,
                  size_t gapMinutes = 30) const;

    // Per-symbol/per-tag top/worst sessions (Sprint #172).
    // Mirror of #171 filtered to a single segment.
    std::vector<TradingSession>
    topSessionsBySymbol(const std::string& symbol,
                        size_t n = 5,
                        size_t gapMinutes = 30) const;
    std::vector<TradingSession>
    worstSessionsBySymbol(const std::string& symbol,
                          size_t n = 5,
                          size_t gapMinutes = 30) const;
    std::vector<TradingSession>
    topSessionsByTag(const std::string& tag,
                      bool includeUntagged = false,
                      size_t n = 5,
                      size_t gapMinutes = 30) const;
    std::vector<TradingSession>
    worstSessionsByTag(const std::string& tag,
                        bool includeUntagged = false,
                        size_t n = 5,
                        size_t gapMinutes = 30) const;

    // Day-of-week × hour heatmap (Sprint #173). For each
    // (weekday 0-6, hour 0-23) combination with at least one
    // round-trip, returns total realized and count. Useful
    // for "when am I profitable?" — 7×24 grid visualization.
    struct HeatmapCell {
        int    weekday = 0;  // 0=Sun, 6=Sat
        int    hour    = 0;  // 0-23
        double realized = 0.0;
        size_t count   = 0;
    };
    std::vector<HeatmapCell> weekdayHourPnL() const;

    // Per-segment weekday-hour heatmap (Sprint #174).
    // Same shape as weekdayHourPnL (#173) but filtered
    // to a single symbol or tag.
    struct SegmentHeatmapCell {
        std::string segment;
        int         weekday = 0;
        int         hour    = 0;
        double      realized = 0.0;
        size_t      count   = 0;
    };
    std::vector<SegmentHeatmapCell>
    weekdayHourPnLBySymbol(const std::string& symbol) const;
    std::vector<SegmentHeatmapCell>
    weekdayHourPnLByTag(const std::string& tag,
                         bool includeUntagged = false) const;

    // Top N most-traded segments (Sprint #175). Bulk sort
    // by fill count DESC. Answers "where do I spend my
    // trading time?"
    struct VolumeEntry {
        std::string segment;
        size_t      totalFills = 0;
        double      realized   = 0.0;
        double      shares     = 0.0;  // share of journal fills
    };
    std::vector<VolumeEntry>
    topMostTradedSymbols(size_t n = 0) const;  // 0 = all
    std::vector<VolumeEntry>
    topMostTradedTags(size_t n = 0,
                       bool includeUntagged = true) const;

    // Journal metadata (Sprint #176). High-level summary
    // of the journal's existence — first/last fill, span,
    // active days, totals.
    struct JournalMeta {
        uint64_t firstFillUs   = 0;
        uint64_t lastFillUs    = 0;
        double   spanDays      = 0.0;
        size_t   activeDays    = 0;
        size_t   totalFills    = 0;
        size_t   totalSymbols  = 0;
        size_t   totalTags     = 0;
        double   totalRealized = 0.0;
        double   maxDD         = 0.0;
        double   sharpe        = 0.0;
    };
    JournalMeta journalMetadata() const;

    // Forward decl for dailyPnLSeriesBySymbol/Tag (Sprint #177).
    // DailyPnL is defined further down; we declare its
    // existence here so the method signatures compile.
    struct DailyPnL;
    std::vector<DailyPnL> dailyPnLSeriesBySymbol(
        const std::string& symbol) const;
    std::vector<DailyPnL> dailyPnLSeriesByTag(
        const std::string& tag,
        bool includeUntagged = false) const;

    // Symbol leaderboard (Sprint #161). For each symbol,
    // compute a key performance metric and sort symbols
    // by it DESC. Single method that returns the
    // leaderboard view.
    enum class LeaderboardMetric {
        Realized,        // total net P&L
        Sharpe,          // annual Sharpe ratio
        WinRate,         // win rate %
        ProfitFactor,    // grossWin / |grossLoss|
        RecoveryFactor,  // net / maxDD
        RiskScore        // composite 0-100 score (#135)
    };
    struct LeaderboardEntry {
        std::string symbol;
        double      metricValue = 0.0;
        size_t      totalFills = 0;
        double      realized   = 0.0;
    };
    std::vector<LeaderboardEntry>
    symbolLeaderboard(LeaderboardMetric metric =
                      LeaderboardMetric::Realized) const;
    // Per-tag version.
    std::vector<LeaderboardEntry>
    tagLeaderboard(LeaderboardMetric metric =
                    LeaderboardMetric::Realized,
                    bool includeUntagged = true) const;

    // Day-of-week stats — Sprint #106. For each (symbol,
    // weekday) bucket with at least one round-trip, the
    // aggregated stats. Answers "do I lose money on Mondays
    // for SOL specifically?" — the trader wants calendar-aware
    // analytics that survive across months.
    //
    // weekday: 0=Sunday, 1=Monday, ..., 6=Saturday (matches
    // struct tm.tm_wday from localtime_r).
    struct DayOfWeekBucket {
        size_t roundTrips = 0;
        size_t wins       = 0;
        size_t losses     = 0;
        double realized   = 0.0;
    };
    struct PerSymbolDayOfWeekStats {
        std::vector<std::string> symbols;       // rows (sorted ASC)
        // Indexed as grid[symIdx * 7 + weekday].
        std::vector<DayOfWeekBucket> grid;
    };
    PerSymbolDayOfWeekStats perSymbolDayOfWeekStats() const;

    // Per-tag day-of-week mirror. Honors includeUntagged.
    struct PerTagDayOfWeekStats {
        std::vector<std::string> tags;
        std::vector<DayOfWeekBucket> grid;
    };
    PerTagDayOfWeekStats perTagDayOfWeekStats(
        bool includeUntagged = false) const;

    // Daily P&L time series (Sprint #109). One entry per
    // calendar day with at least one round-trip. Sorted by
    // date ASC. Used as the input for rolling Sharpe + other
    // time-series analytics.
    //
    // Each point: (date, realized, roundTrips). realized is
    // the sum of realizedDelta for fills on that day; roundTrips
    // is the count of round-trip fills (|realizedDelta|>1e-9).
    //
    // Cross-method invariant:
    //   Σ realized == stats().netRealized (over the full history)
    //   Σ roundTrips == stats().roundTripCount
    // (Test 99 verifies this.)
    struct DailyPnL {
        std::string date;        // YYYY-MM-DD
        double      realized     = 0.0;
        size_t      roundTrips   = 0;
    };
    std::vector<DailyPnL> dailyPnLSeries() const;

    // Per-symbol daily P&L series (Sprint #109). Returns a
    // map: symbol → ordered list of (date, realized,
    // roundTrips). Used by rollingSharpeBySymbol() + future
    // per-symbol Sharpe time-series widgets.
    std::map<std::string, std::vector<DailyPnL>>
        perSymbolDailyPnL() const;

    // Per-tag daily P&L series (Sprint #109). Honors
    // includeUntagged (rolls untagged fills into "__untagged__"
    // when true; skips them when false).
    std::map<std::string, std::vector<DailyPnL>>
        perTagDailyPnL(bool includeUntagged = false) const;

    // Rolling Sharpe (Sprint #109). Daily Sharpe ratio on a
    // rolling window of size `windowDays` (default 30).
    // Returns one point per day where the rolling window is
    // complete (i.e. at least windowDays days of history are
    // available up to that day).
    //
    // Sharpe = mean(window daily returns) / stddev(window
    // daily returns) * sqrt(252) (annualized trading-day
    // convention).
    //
    // Each point: (date, sharpe). Days with no fills get
    // realized=0 in the input series (not skipped) — this
    // keeps the rolling window aligned with calendar days.
    struct RollingSharpePoint {
        std::string date;        // YYYY-MM-DD (last day of window)
        double      sharpe       = 0.0;   // 0 if stddev==0
    };
    std::vector<RollingSharpePoint>
        rollingSharpe(size_t windowDays = 30) const;

    // Per-symbol rolling Sharpe (Sprint #109). Returns one
    // RollingSharpePoint series per symbol that has enough
    // history (>= windowDays of fills). Symbols with fewer
    // than windowDays of history get an empty vector.
    std::map<std::string, std::vector<RollingSharpePoint>>
        rollingSharpeBySymbol(size_t windowDays = 30) const;

    // Hour-of-day stats — Sprint #106. Same shape as
    // day-of-week but bucketed by hour 0..23 (local time).
    // Answers "do I always lose at 14:00?".
    //
    // Grid indexed as grid[symIdx * 24 + hour].
    struct HourOfDayBucket {
        size_t roundTrips = 0;
        size_t wins       = 0;
        size_t losses     = 0;
        double realized   = 0.0;
    };
    struct PerSymbolHourOfDayStats {
        std::vector<std::string> symbols;
        std::vector<HourOfDayBucket> grid;
    };
    PerSymbolHourOfDayStats perSymbolHourOfDayStats() const;
    struct PerTagHourOfDayStats {
        std::vector<std::string> tags;
        std::vector<HourOfDayBucket> grid;
    };
    PerTagHourOfDayStats perTagHourOfDayStats(
        bool includeUntagged = false) const;

    // Per-tag Sortino (Sprint #99). Per-tag mirror.
    struct PerTagSortino {
        std::string tag;
        double dailySortino       = 0.0;
        double annualizedSortino  = 0.0;
        double meanDailyReturn    = 0.0;
        double downsideDeviation  = 0.0;
        size_t sampleSize         = 0;
    };
    std::vector<PerTagSortino> perTagSortino(
        bool includeUntagged = false) const;

    // Per-symbol risk-adjusted return (Sprint #91). For each
    // symbol, Sharpe on the daily series — same algorithm as
    // sharpe() (#84) but applied to the symbol's own daily series.
    // Sorted by annualized Sharpe DESCENDING so the symbol with
    // the best risk-adjusted return surfaces first.
    //
    // A symbol with fewer than 2 distinct trading days gets
    // dailySharpe=0, annualizedSharpe=0 (no division by zero).
    // meanDailyReturn is still meaningful on a single day; the
    // stddev is just 0.
    struct PerSymbolSharpe {
        std::string symbol;
        double dailySharpe       = 0.0;
        double annualizedSharpe  = 0.0;
        double meanDailyReturn   = 0.0;
        double stddevDailyReturn = 0.0;
        size_t sampleSize        = 0;
    };
    std::vector<PerSymbolSharpe> perSymbolSharpe() const;

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
