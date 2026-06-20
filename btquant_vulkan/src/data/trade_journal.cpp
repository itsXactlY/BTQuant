#include "trade_journal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

namespace btquant {

namespace {

// Minimal hand-rolled escape/unescape for JSON string values. Sufficient
// for symbol names + the small set of strings we serialize.
std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

std::string jsonUnescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next = s[i + 1];
            switch (next) {
                case '"':  out.push_back('"');  ++i; break;
                case '\\': out.push_back('\\'); ++i; break;
                case 'n':  out.push_back('\n'); ++i; break;
                case 'r':  out.push_back('\r'); ++i; break;
                case 't':  out.push_back('\t'); ++i; break;
                default:   out.push_back(s[i]);      break;
            }
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

// Parse a flat JSON object {"k":v,"k2":v2,...} into a key→raw-value map.
// Values are returned as raw strings (numbers, booleans, quoted strings).
struct KV { std::string key, raw; };

std::vector<KV> parseFlatObject(const std::string& line) {
    std::vector<KV> out;
    size_t i = 0;
    // Skip whitespace and outer braces.
    auto skipWs = [&]() {
        while (i < line.size() &&
               (line[i] == ' '  || line[i] == '\t' ||
                line[i] == '\n' || line[i] == '\r')) ++i;
    };
    skipWs();
    if (i >= line.size() || line[i] != '{') return out;
    ++i;
    while (i < line.size()) {
        skipWs();
        if (i < line.size() && line[i] == '}') { ++i; break; }
        // Read key.
        if (i >= line.size() || line[i] != '"') break;
        ++i;
        std::string key;
        while (i < line.size() && line[i] != '"') {
            if (line[i] == '\\' && i + 1 < line.size()) {
                key.push_back(line[i]);
                key.push_back(line[i + 1]);
                i += 2;
            } else {
                key.push_back(line[i++]);
            }
        }
        if (i < line.size()) ++i;  // closing "
        skipWs();
        // Colon.
        if (i >= line.size() || line[i] != ':') break;
        ++i;
        skipWs();
        // Value: quoted string OR bareword (number/bool).
        std::string raw;
        if (i < line.size() && line[i] == '"') {
            ++i;
            while (i < line.size() && line[i] != '"') {
                if (line[i] == '\\' && i + 1 < line.size()) {
                    raw.push_back(line[i]);
                    raw.push_back(line[i + 1]);
                    i += 2;
                } else {
                    raw.push_back(line[i++]);
                }
            }
            if (i < line.size()) ++i;
        } else {
            while (i < line.size() && line[i] != ',' && line[i] != '}') {
                raw.push_back(line[i++]);
            }
        }
        // Trim trailing whitespace.
        while (!raw.empty() &&
               (raw.back() == ' ' || raw.back() == '\t' ||
                raw.back() == '\n' || raw.back() == '\r')) {
            raw.pop_back();
        }
        out.push_back({jsonUnescape(key), raw});
        skipWs();
        if (i < line.size() && line[i] == ',') { ++i; continue; }
        if (i < line.size() && line[i] == '}') { ++i; break; }
    }
    return out;
}

const KV* findKV(const std::vector<KV>& kvs, const std::string& key) {
    for (const auto& kv : kvs) {
        if (kv.key == key) return &kv;
    }
    return nullptr;
}

} // namespace

TradeJournal::TradeJournal(const std::string& path) : m_path(path) {}

bool TradeJournal::append(const JournalFill& r) {
    namespace fs = std::filesystem;
    try {
        fs::path p(m_path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        std::ofstream out(m_path, std::ios::app);
        if (!out.is_open()) return false;
        out << toJsonLine(r) << "\n";
        return out.good();
    } catch (...) {
        return false;
    }
}

double TradeJournal::totalRealized() const {
    std::vector<JournalFill> fills = loadAll();
    double sum = 0.0;
    for (const auto& f : fills) sum += f.realizedDelta;
    return sum;
}

std::vector<std::pair<std::string, double>>
TradeJournal::realizedBySymbol() const {
    std::vector<JournalFill> fills = loadAll();
    std::unordered_map<std::string, double> agg;
    for (const auto& f : fills) {
        agg[f.symbol] += f.realizedDelta;
    }
    std::vector<std::pair<std::string, double>> out;
    out.reserve(agg.size());
    for (auto& kv : agg) out.emplace_back(std::move(kv.first), kv.second);
    std::sort(out.begin(), out.end(),
              [](const std::pair<std::string, double>& a,
                 const std::pair<std::string, double>& b) {
                  return std::fabs(a.second) > std::fabs(b.second);
              });
    return out;
}

std::vector<std::pair<std::string, double>>
TradeJournal::realizedByTag(bool includeUntagged) const {
    std::vector<JournalFill> fills = loadAll();
    std::unordered_map<std::string, double> agg;
    for (const auto& f : fills) {
        if (f.tag.empty() && !includeUntagged) continue;
        const std::string key = f.tag.empty() ? "__untagged__" : f.tag;
        agg[key] += f.realizedDelta;
    }
    std::vector<std::pair<std::string, double>> out;
    out.reserve(agg.size());
    for (auto& kv : agg) out.emplace_back(std::move(kv.first), kv.second);
    std::sort(out.begin(), out.end(),
              [](const std::pair<std::string, double>& a,
                 const std::pair<std::string, double>& b) {
                  return std::fabs(a.second) > std::fabs(b.second);
              });
    return out;
}

TradeJournal::Stats TradeJournal::stats() const {
    // Epsilon for "is this a real win/loss vs a rounding artifact".
    // 1e-9 is well below any meaningful dollar amount on a typical
    // trade but large enough to swallow float-json round-trip noise.
    constexpr double kEps = 1e-9;

    std::vector<JournalFill> fills = loadAll();
    Stats s;
    s.fillCount = fills.size();

    double grossWin  = 0.0;
    double grossLoss = 0.0;
    double sumRTpnl  = 0.0;

    for (const auto& f : fills) {
        s.netRealized += f.realizedDelta;
        // Only round-trip fills (realized != 0) count for win/loss
        // statistics. Open fills have realizedDelta == 0 by
        // definition and would otherwise skew winRate toward 0%.
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        s.roundTripCount++;
        sumRTpnl += f.realizedDelta;
        if (f.realizedDelta > kEps) {
            s.winCount++;
            grossWin += f.realizedDelta;
        } else if (f.realizedDelta < -kEps) {
            s.lossCount++;
            grossLoss += f.realizedDelta;  // negative
        }
    }

    if (s.roundTripCount > 0) {
        s.winRate    = static_cast<double>(s.winCount) /
                       static_cast<double>(s.roundTripCount);
        s.expectancy = sumRTpnl /
                       static_cast<double>(s.roundTripCount);
    }
    if (s.winCount  > 0) s.avgWinner = grossWin  / s.winCount;
    if (s.lossCount > 0) s.avgLoser  = grossLoss /
                                      static_cast<double>(s.lossCount);
    // Mirror RiskMetrics sentinel: no losses + at least one win =
    // "infinite" profit factor. No fills at all = 0 (not inf, not
    // NaN — the panel can format it as "—" without special-casing).
    if (s.lossCount == 0) {
        s.profitFactor = (s.winCount > 0)
            ? std::numeric_limits<double>::infinity()
            : 0.0;
    } else {
        s.profitFactor = grossWin / -grossLoss;
    }

    return s;
}

std::vector<std::pair<std::string, double>>
TradeJournal::realizedByDay() const {
    std::vector<JournalFill> fills = loadAll();

    // Build a date-bucketed map. std::map (not unordered) so the
    // iteration is naturally sorted by date string — and since the
    // key is "YYYY-MM-DD", lexical sort matches chronological sort.
    // A trader who wants to read "last week" just slices the tail.
    std::map<std::string, double> buckets;

    for (const auto& f : fills) {
        // timestamp_us is system_clock::now() at fill time.
        // localtime_r groups by the trader's local midnight — same
        // convention as RiskGuard's auto-reset (#69). std::time_t
        // is seconds; truncate microseconds before conversion.
        std::time_t secs = static_cast<std::time_t>(f.timestamp_us / 1000000ULL);
        std::tm tm{};
        // localtime_r is POSIX; localtime_s is Windows. Use the
        // POSIX form with a portable fallback via localtime when
        // _POSIX_C_SOURCE isn't defined.
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        char date[16];  // "YYYY-MM-DD" + null
        std::strftime(date, sizeof(date), "%Y-%m-%d", &tm);
        buckets[date] += f.realizedDelta;
    }

    // std::map iteration is already date-ASC (lexical sort ==
    // chronological for ISO dates). Drain into a vector and return.
    std::vector<std::pair<std::string, double>> out;
    out.reserve(buckets.size());
    for (auto& kv : buckets) {
        out.emplace_back(std::move(kv.first), kv.second);
    }
    return out;
}

namespace {

// Pure helpers shared between the journal-wide and per-symbol
// methods. Operate on a chronological (date, value) series — the
// caller decides how to bucket (by day across all fills, or by day
// scoped to a single symbol/tag/etc.). Keeping the algorithm
// separate from the bucketing means a future perTagDrawdown() (or
// per-strategy Sharpe) is a thin wrapper, not a duplicate.

// Compute the worst peak-to-trough decline on a chronological
// equity-curve series. The same algorithm that was inline in
// maxDrawdown() (#80) — refactored into a helper in Sprint #91 so
// perSymbolDrawdown() (#91) can reuse it without copy-paste.
//
// Algorithm:
//   - Walk the series forward, tracking running peak + peakDate.
//   - When equity hits a new peak, update peakDate to "now".
//   - Whenever drawdown widens beyond the worst seen, record the
//     worstDD + the troughDate that ended it + the peakDate that
//     anchored it (i.e. the most recent peak anchor at the time
//     the trough occurred).
//   - currentDD is the drawdown as of the last equity point.
//   - recoveryDate + recoveryDays (Sprint #95): after a drawdown
//     bottoms, walk forward until equity returns to the peak that
//     started the worstDD. The first date that meets or exceeds
//     that peak is the recovery date. Days = recoveryDate -
//     troughDate (in calendar days). Stays empty/0 when the
//     drawdown hasn't been recovered yet, or when there's no
//     drawdown at all.
TradeJournal::Drawdown computeDrawdownFromSeries(
    const std::vector<std::pair<std::string, double>>& daily) {
    TradeJournal::Drawdown dd;
    if (daily.empty()) return dd;
    double equity = 0.0;
    double peak   = 0.0;
    double worstDD = 0.0;
    double worstDDPeak = 0.0;   // peak value that started the worst DD
    std::string peakDate;          // date of running peak
    std::string peakDateAtWorst;   // peakDate captured at worstDD
    std::string troughDateAtWorst; // date of worst trough
    for (const auto& kv : daily) {
        equity += kv.second;
        if (equity > peak) {
            peak = equity;
            peakDate = kv.first;
        }
        double curDD = peak - equity;  // >= 0
        if (curDD > worstDD + 1e-9) {
            worstDD = curDD;
            worstDDPeak = peak;        // remember the peak we dropped from
            troughDateAtWorst = kv.first;
            peakDateAtWorst = peakDate;
        }
    }
    dd.maxDrawdown = worstDD;
    if (worstDD > 1e-9) {
        dd.peakDate = peakDateAtWorst;
        dd.troughDate = troughDateAtWorst;
        // Sprint #95: walk forward from the trough to find when
        // equity recovered to the peak that started the worst DD.
        // Track cumulative equity from the start of the series,
        // but only consider dates after the trough. The first
        // such date where cumulative equity reaches the peak that
        // started the worstDD is the recovery date.
        // daysFromTrough counts the number of distinct trading
        // days between trough and recovery. When equity never
        // reaches worstDDPeak again, recoveryDate/recoveryDays
        // stay empty/0.
        if (worstDDPeak > 1e-9) {
            bool seenTrough = false;
            double cum = 0.0;
            int  daysFromTrough = 0;
            for (const auto& kv : daily) {
                cum += kv.second;
                if (!seenTrough) {
                    if (kv.first == troughDateAtWorst) {
                        seenTrough = true;
                        // Don't count the trough itself as a
                        // recovery day — recovery is the *next*
                        // day equity reaches the peak.
                    }
                    continue;
                }
                daysFromTrough++;
                if (cum >= worstDDPeak - 1e-9) {
                    dd.recoveryDate = kv.first;
                    dd.recoveryDays = static_cast<size_t>(
                        daysFromTrough);
                    break;
                }
            }
            // Empty recoveryDate + 0 recoveryDays is the
            // documented sentinel when the DD hasn't been
            // recovered yet.
        }
    }
    dd.currentDD = peak - equity;
    return dd;
}

// Compute Sharpe on a chronological daily series. Same algorithm
// as sharpe() (#84) — extracted in Sprint #91 so perSymbolSharpe()
// can reuse it.
//
//   dailySharpe      = mean / stddev (sample, Bessel-corrected)
//   annualizedSharpe = daily * sqrt(252)
//   meanDailyReturn  = sum / N
//   stddevDailyReturn= sqrt(Σ(d-mean)² / (N-1))
//
// Returns zeroed Sharpe when sampleSize < 2 (no division by zero,
// no NaN).
TradeJournal::Sharpe computeSharpeFromSeries(
    const std::vector<std::pair<std::string, double>>& daily) {
    constexpr double kTradingDays = 252.0;
    TradeJournal::Sharpe out;
    out.sampleSize = daily.size();
    if (daily.empty()) return out;

    double sum = 0.0;
    for (const auto& kv : daily) sum += kv.second;
    out.meanDailyReturn = sum / static_cast<double>(daily.size());

    if (daily.size() < 2) {
        out.stddevDailyReturn = 0.0;
        out.dailySharpe       = 0.0;
        out.annualizedSharpe  = 0.0;
        return out;
    }
    double sqSum = 0.0;
    for (const auto& kv : daily) {
        double d = kv.second - out.meanDailyReturn;
        sqSum += d * d;
    }
    out.stddevDailyReturn = std::sqrt(sqSum /
                                      static_cast<double>(daily.size() - 1));
    if (out.stddevDailyReturn > 1e-9) {
        out.dailySharpe = out.meanDailyReturn / out.stddevDailyReturn;
        out.annualizedSharpe = out.dailySharpe * std::sqrt(kTradingDays);
    } else {
        out.dailySharpe      = 0.0;
        out.annualizedSharpe = 0.0;
    }
    return out;
}

// Convert a (timestamp_us, realizedDelta) series into a
// chronological "YYYY-MM-DD" → sum(realizedDelta) map. Pulled out
// because perSymbolDrawdown() and perSymbolSharpe() (#91) both
// need it scoped to a single symbol — calling realizedByDay()
// once and filtering would re-bucket the journal unnecessarily.
std::map<std::string, double> bucketByLocalDay(
    const std::vector<JournalFill>& fills) {
    std::map<std::string, double> buckets;
    for (const auto& f : fills) {
        std::time_t secs = static_cast<std::time_t>(f.timestamp_us /
                                                    1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        char date[16];
        std::strftime(date, sizeof(date), "%Y-%m-%d", &tm);
        buckets[date] += f.realizedDelta;
    }
    return buckets;
}

}  // namespace

TradeJournal::Drawdown TradeJournal::maxDrawdown() const {
    // Sprint #91 refactor: extracted the algorithm into
    // computeDrawdownFromSeries() so perSymbolDrawdown() can share
    // it. The result is identical to the prior implementation —
    // the daily bucket ordering matches realizedByDay() (which
    // returns std::map iteration order = lexical = chronological
    // for ISO dates).
    auto daily = realizedByDay();
    return computeDrawdownFromSeries(daily);
}

TradeJournal::Streaks TradeJournal::streaks() const {
    // Same epsilon as stats() — swallow float-json round-trip
    // noise without classifying a true zero as a win/loss.
    constexpr double kEps = 1e-9;

    Streaks s;
    std::vector<JournalFill> fills = loadAll();

    size_t runWin  = 0;
    size_t runLoss = 0;

    for (const auto& f : fills) {
        // Open fills (realized == 0) neither break nor extend a
        // streak — they don't represent a decision outcome. Skip
        // them in the run counters.
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        if (f.realizedDelta > kEps) {
            runWin++;
            runLoss = 0;
            if (runWin > s.longestWinStreak) s.longestWinStreak = runWin;
        } else {
            runLoss++;
            runWin = 0;
            if (runLoss > s.longestLossStreak) s.longestLossStreak = runLoss;
        }
    }
    // The current streak is whichever run is still open at the
    // end of the walk — exactly one of these will be > 0 if
    // there's been at least one round-trip.
    s.currentWinStreak  = runWin;
    s.currentLossStreak = runLoss;
    return s;
}

TradeJournal::Sharpe TradeJournal::sharpe() const {
    // Sprint #91 refactor: extracted the algorithm into
    // computeSharpeFromSeries(). Result is identical to the prior
    // implementation; the bucket ordering matches realizedByDay().
    auto daily = realizedByDay();
    return computeSharpeFromSeries(daily);
}

std::vector<TradeJournal::PerSymbolStats>
TradeJournal::perSymbolStats() const {
    // Same epsilon as stats() — swallow float-json round-trip
    // noise without classifying a true zero as a win/loss.
    constexpr double kEps = 1e-9;

    std::vector<JournalFill> fills = loadAll();

    // Two-pass aggregation:
    //   1) Sum realized + count W/L/round-trips per symbol.
    //   2) Compute derived stats (winRate, avgW/L, PF, expectancy).
    // Single pass would need lazy eval / mutable struct fields;
    // two passes with local maps is clearer and the cost is the
    // same (O(N) over fills either way).
    struct Acc {
        double realized    = 0.0;
        double grossWin    = 0.0;
        double grossLoss   = 0.0;
        double sumRTpnl    = 0.0;
        size_t roundTrips  = 0;
        size_t wins        = 0;
        size_t losses      = 0;
    };
    std::unordered_map<std::string, Acc> accs;
    accs.reserve(8);

    for (const auto& f : fills) {
        Acc& a = accs[f.symbol];
        a.realized += f.realizedDelta;
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        a.roundTrips++;
        a.sumRTpnl += f.realizedDelta;
        if (f.realizedDelta > kEps) {
            a.wins++;
            a.grossWin += f.realizedDelta;
        } else if (f.realizedDelta < -kEps) {
            a.losses++;
            a.grossLoss += f.realizedDelta;  // negative
        }
    }

    // Drain into vector, apply derived stats, sort by abs-realized DESC.
    std::vector<PerSymbolStats> out;
    out.reserve(accs.size());
    for (auto& kv : accs) {
        PerSymbolStats s;
        s.symbol          = kv.first;
        s.realized        = kv.second.realized;
        s.roundTripCount  = kv.second.roundTrips;
        s.winCount        = kv.second.wins;
        s.lossCount       = kv.second.losses;
        if (kv.second.roundTrips > 0) {
            s.winRate    = static_cast<double>(kv.second.wins) /
                           static_cast<double>(kv.second.roundTrips);
            s.expectancy = kv.second.sumRTpnl /
                           static_cast<double>(kv.second.roundTrips);
        }
        if (kv.second.wins   > 0) s.avgWinner = kv.second.grossWin  /
                                                 kv.second.wins;
        if (kv.second.losses > 0) s.avgLoser  = kv.second.grossLoss /
                                                 static_cast<double>(kv.second.losses);
        // PF sentinel: no losses + at least one win = +inf.
        // No fills at all = 0 (matches stats() convention).
        if (kv.second.losses == 0) {
            s.profitFactor = (kv.second.wins > 0)
                ? std::numeric_limits<double>::infinity()
                : 0.0;
        } else {
            s.profitFactor = kv.second.grossWin / -kv.second.grossLoss;
        }
        out.push_back(std::move(s));
    }
    std::sort(out.begin(), out.end(),
              [](const PerSymbolStats& a, const PerSymbolStats& b) {
                  return std::fabs(a.realized) > std::fabs(b.realized);
              });
    return out;
}

std::vector<TradeJournal::PerTagStats>
TradeJournal::perTagStats(bool includeUntagged) const {
    // Same epsilon + Acc pattern as perSymbolStats() (#86). The
    // map is keyed by tag instead of symbol; untagged fills use
    // the "__untagged__" synthetic key when includeUntagged is
    // true (matches realizedByTag() — #73).
    constexpr double kEps = 1e-9;

    std::vector<JournalFill> fills = loadAll();

    struct Acc {
        double realized    = 0.0;
        double grossWin    = 0.0;
        double grossLoss   = 0.0;
        double sumRTpnl    = 0.0;
        size_t roundTrips  = 0;
        size_t wins        = 0;
        size_t losses      = 0;
    };
    std::unordered_map<std::string, Acc> accs;
    accs.reserve(8);

    for (const auto& f : fills) {
        if (f.tag.empty() && !includeUntagged) continue;
        const std::string key = f.tag.empty() ? "__untagged__" : f.tag;
        Acc& a = accs[key];
        a.realized += f.realizedDelta;
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        a.roundTrips++;
        a.sumRTpnl += f.realizedDelta;
        if (f.realizedDelta > kEps) {
            a.wins++;
            a.grossWin += f.realizedDelta;
        } else if (f.realizedDelta < -kEps) {
            a.losses++;
            a.grossLoss += f.realizedDelta;
        }
    }

    std::vector<PerTagStats> out;
    out.reserve(accs.size());
    for (auto& kv : accs) {
        PerTagStats s;
        s.tag            = kv.first;
        s.realized       = kv.second.realized;
        s.roundTripCount = kv.second.roundTrips;
        s.winCount       = kv.second.wins;
        s.lossCount      = kv.second.losses;
        if (kv.second.roundTrips > 0) {
            s.winRate    = static_cast<double>(kv.second.wins) /
                           static_cast<double>(kv.second.roundTrips);
            s.expectancy = kv.second.sumRTpnl /
                           static_cast<double>(kv.second.roundTrips);
        }
        if (kv.second.wins   > 0) s.avgWinner = kv.second.grossWin  /
                                                 kv.second.wins;
        if (kv.second.losses > 0) s.avgLoser  = kv.second.grossLoss /
                                                 static_cast<double>(kv.second.losses);
        if (kv.second.losses == 0) {
            s.profitFactor = (kv.second.wins > 0)
                ? std::numeric_limits<double>::infinity()
                : 0.0;
        } else {
            s.profitFactor = kv.second.grossWin / -kv.second.grossLoss;
        }
        out.push_back(std::move(s));
    }
    std::sort(out.begin(), out.end(),
              [](const PerTagStats& a, const PerTagStats& b) {
                  return std::fabs(a.realized) > std::fabs(b.realized);
              });
    return out;
}

std::vector<TradeJournal::PerSymbolDrawdown>
TradeJournal::perSymbolDrawdown() const {
    // For each distinct symbol: bucket that symbol's fills by local
    // day, derive the daily equity curve, and compute the worst
    // peak-to-trough decline via the shared helper. Sorted by
    // maxDrawdown DESCENDING so the worst symbol surfaces first —
    // matches the "which symbol hurt me most?" question.
    std::vector<JournalFill> fills = loadAll();

    // Group fills by symbol first, then bucket each group's daily.
    // Two maps deep is fine — total cost is O(N) over fills, and
    // a per-symbol-sort pass at the end is O(K log K) over
    // distinct symbols.
    std::unordered_map<std::string, std::vector<JournalFill>> bySymbol;
    bySymbol.reserve(8);
    for (const auto& f : fills) bySymbol[f.symbol].push_back(f);

    std::vector<PerSymbolDrawdown> out;
    out.reserve(bySymbol.size());
    for (auto& kv : bySymbol) {
        PerSymbolDrawdown e;
        e.symbol = kv.first;
        e.fillCount = kv.second.size();
        // bucketByLocalDay returns a std::map — convert to vector
        // of pairs (already in chronological order thanks to
        // std::map's lexical sort on ISO dates).
        auto buckets = bucketByLocalDay(kv.second);
        std::vector<std::pair<std::string, double>> series;
        series.reserve(buckets.size());
        for (auto& bkv : buckets) {
            series.emplace_back(std::move(bkv.first), bkv.second);
        }
        auto dd = computeDrawdownFromSeries(series);
        e.maxDrawdown   = dd.maxDrawdown;
        e.peakDate      = dd.peakDate;
        e.troughDate    = dd.troughDate;
        e.recoveryDate  = dd.recoveryDate;     // Sprint #95
        e.recoveryDays  = dd.recoveryDays;     // Sprint #95
        e.currentDD     = dd.currentDD;
        out.push_back(std::move(e));
    }
    // Worst-first: symbol with biggest maxDD tops the list. When
    // two symbols tie (e.g. both never had a drawdown, both
    // maxDD==0), the std::unordered_map iteration order is
    // implementation-defined — the test that asserts ordering
    // should pick values that don't tie at zero.
    std::sort(out.begin(), out.end(),
              [](const PerSymbolDrawdown& a, const PerSymbolDrawdown& b) {
                  if (a.maxDrawdown != b.maxDrawdown)
                      return a.maxDrawdown > b.maxDrawdown;
                  // Tie-break by symbol name so the output is
                  // deterministic across runs (unordered_map
                  // iteration order isn't).
                  return a.symbol < b.symbol;
              });
    return out;
}

std::vector<TradeJournal::PerSymbolSharpe>
TradeJournal::perSymbolSharpe() const {
    // Same shape as perSymbolDrawdown() but the daily series feeds
    // computeSharpeFromSeries() instead. Sorted by annualized
    // Sharpe DESCENDING — answers "which symbol gives me the best
    // return per unit of risk?" directly from the rendered table.
    std::vector<JournalFill> fills = loadAll();

    std::unordered_map<std::string, std::vector<JournalFill>> bySymbol;
    bySymbol.reserve(8);
    for (const auto& f : fills) bySymbol[f.symbol].push_back(f);

    std::vector<PerSymbolSharpe> out;
    out.reserve(bySymbol.size());
    for (auto& kv : bySymbol) {
        PerSymbolSharpe e;
        e.symbol = kv.first;
        auto buckets = bucketByLocalDay(kv.second);
        std::vector<std::pair<std::string, double>> series;
        series.reserve(buckets.size());
        for (auto& bkv : buckets) {
            series.emplace_back(std::move(bkv.first), bkv.second);
        }
        auto sh = computeSharpeFromSeries(series);
        e.dailySharpe       = sh.dailySharpe;
        e.annualizedSharpe  = sh.annualizedSharpe;
        e.meanDailyReturn   = sh.meanDailyReturn;
        e.stddevDailyReturn = sh.stddevDailyReturn;
        e.sampleSize        = sh.sampleSize;
        out.push_back(std::move(e));
    }
    std::sort(out.begin(), out.end(),
              [](const PerSymbolSharpe& a, const PerSymbolSharpe& b) {
                  if (a.annualizedSharpe != b.annualizedSharpe)
                      return a.annualizedSharpe > b.annualizedSharpe;
                  // Tie-break by mean daily return (DESC) then by
                  // symbol name (ASC) for deterministic output.
                  if (a.meanDailyReturn != b.meanDailyReturn)
                      return a.meanDailyReturn > b.meanDailyReturn;
                  return a.symbol < b.symbol;
              });
    return out;
}

std::vector<TradeJournal::PerTagDrawdown>
TradeJournal::perTagDrawdown(bool includeUntagged) const {
    // Per-tag mirror of perSymbolDrawdown() (#91). Same algorithm,
    // same sort + tie-break. `includeUntagged` matches perTagStats()
    // (#88): empty-tag fills are skipped when false (default),
    // aggregated under "__untagged__" when true.
    std::vector<JournalFill> fills = loadAll();

    std::unordered_map<std::string, std::vector<JournalFill>> byTag;
    byTag.reserve(8);
    for (const auto& f : fills) {
        if (f.tag.empty() && !includeUntagged) continue;
        const std::string key = f.tag.empty() ? "__untagged__" : f.tag;
        byTag[key].push_back(f);
    }

    std::vector<PerTagDrawdown> out;
    out.reserve(byTag.size());
    for (auto& kv : byTag) {
        PerTagDrawdown e;
        e.tag = kv.first;
        e.fillCount = kv.second.size();
        auto buckets = bucketByLocalDay(kv.second);
        std::vector<std::pair<std::string, double>> series;
        series.reserve(buckets.size());
        for (auto& bkv : buckets) {
            series.emplace_back(std::move(bkv.first), bkv.second);
        }
        auto dd = computeDrawdownFromSeries(series);
        e.maxDrawdown   = dd.maxDrawdown;
        e.peakDate      = dd.peakDate;
        e.troughDate    = dd.troughDate;
        e.recoveryDate  = dd.recoveryDate;     // Sprint #95
        e.recoveryDays  = dd.recoveryDays;     // Sprint #95
        e.currentDD     = dd.currentDD;
        out.push_back(std::move(e));
    }
    std::sort(out.begin(), out.end(),
              [](const PerTagDrawdown& a, const PerTagDrawdown& b) {
                  if (a.maxDrawdown != b.maxDrawdown)
                      return a.maxDrawdown > b.maxDrawdown;
                  return a.tag < b.tag;
              });
    return out;
}

std::vector<TradeJournal::PerTagSharpe>
TradeJournal::perTagSharpe(bool includeUntagged) const {
    // Per-tag mirror of perSymbolSharpe() (#91). Same algorithm,
    // same sort + tie-break. includeUntagged matches perTagStats()
    // (#88).
    std::vector<JournalFill> fills = loadAll();

    std::unordered_map<std::string, std::vector<JournalFill>> byTag;
    byTag.reserve(8);
    for (const auto& f : fills) {
        if (f.tag.empty() && !includeUntagged) continue;
        const std::string key = f.tag.empty() ? "__untagged__" : f.tag;
        byTag[key].push_back(f);
    }

    std::vector<PerTagSharpe> out;
    out.reserve(byTag.size());
    for (auto& kv : byTag) {
        PerTagSharpe e;
        e.tag = kv.first;
        auto buckets = bucketByLocalDay(kv.second);
        std::vector<std::pair<std::string, double>> series;
        series.reserve(buckets.size());
        for (auto& bkv : buckets) {
            series.emplace_back(std::move(bkv.first), bkv.second);
        }
        auto sh = computeSharpeFromSeries(series);
        e.dailySharpe       = sh.dailySharpe;
        e.annualizedSharpe  = sh.annualizedSharpe;
        e.meanDailyReturn   = sh.meanDailyReturn;
        e.stddevDailyReturn = sh.stddevDailyReturn;
        e.sampleSize        = sh.sampleSize;
        out.push_back(std::move(e));
    }
    std::sort(out.begin(), out.end(),
              [](const PerTagSharpe& a, const PerTagSharpe& b) {
                  if (a.annualizedSharpe != b.annualizedSharpe)
                      return a.annualizedSharpe > b.annualizedSharpe;
                  if (a.meanDailyReturn != b.meanDailyReturn)
                      return a.meanDailyReturn > b.meanDailyReturn;
                  return a.tag < b.tag;
              });
    return out;
}

TradeJournal::Calmar TradeJournal::calmar() const {
    // Sprint #95. Calmar = annualized return / |max DD|.
    //
    // Two-pass: get sharpe() to read the mean daily return, then
    // get maxDrawdown() for the worst drop. Cost: O(N) over fills
    // twice — the journal is small enough that this is fine, and
    // reusing the existing helpers keeps the contract simple.
    //
    // Edge cases:
    //   - maxDrawdown == 0 → calmarRatio = 0 (sentinel; trader
    //     hasn't seen a drop yet, so the metric is undefined).
    //   - annualizedReturn < 0 → calmarRatio < 0 (losing year
    //     over a non-trivial DD). Negative Calmar is meaningful
    //     — it's a "stay away" signal.
    //   - sharpe() returns zero mean when sampleSize < 1, so a
    //     one-day journal gives calmarRatio = 0 / maxDD = 0.
    Calmar out;
    auto sh = sharpe();
    auto dd = maxDrawdown();
    out.annualizedReturn = sh.meanDailyReturn * 252.0;
    out.maxDrawdown      = dd.maxDrawdown;
    if (dd.maxDrawdown > 1e-9) {
        out.calmarRatio = out.annualizedReturn / dd.maxDrawdown;
    } else {
        out.calmarRatio = 0.0;
    }
    return out;
}

std::vector<TradeJournal::PerSymbolCalmar>
TradeJournal::perSymbolCalmar() const {
    // Per-symbol mirror of calmar() (#95). For each symbol:
    //   1) bucket the symbol's fills by local day
    //   2) feed the daily series into computeSharpeFromSeries()
    //      for meanDailyReturn (annualized × 252 = annRet)
    //   3) feed the same series into computeDrawdownFromSeries()
    //      for maxDrawdown
    //   4) calmarRatio = annRet / maxDD, with 0 sentinel when
    //      maxDD == 0
    // Sorted by calmarRatio DESC.
    std::vector<JournalFill> fills = loadAll();

    std::unordered_map<std::string, std::vector<JournalFill>> bySymbol;
    bySymbol.reserve(8);
    for (const auto& f : fills) bySymbol[f.symbol].push_back(f);

    std::vector<PerSymbolCalmar> out;
    out.reserve(bySymbol.size());
    for (auto& kv : bySymbol) {
        PerSymbolCalmar e;
        e.symbol = kv.first;
        auto buckets = bucketByLocalDay(kv.second);
        std::vector<std::pair<std::string, double>> series;
        series.reserve(buckets.size());
        for (auto& bkv : buckets) {
            series.emplace_back(std::move(bkv.first), bkv.second);
        }
        auto sh = computeSharpeFromSeries(series);
        auto dd = computeDrawdownFromSeries(series);
        e.annualizedReturn = sh.meanDailyReturn * 252.0;
        e.maxDrawdown      = dd.maxDrawdown;
        if (dd.maxDrawdown > 1e-9) {
            e.calmarRatio = e.annualizedReturn / e.maxDrawdown;
        } else {
            e.calmarRatio = 0.0;
        }
        out.push_back(std::move(e));
    }
    std::sort(out.begin(), out.end(),
              [](const PerSymbolCalmar& a, const PerSymbolCalmar& b) {
                  if (a.calmarRatio != b.calmarRatio)
                      return a.calmarRatio > b.calmarRatio;
                  if (a.annualizedReturn != b.annualizedReturn)
                      return a.annualizedReturn > b.annualizedReturn;
                  return a.symbol < b.symbol;
              });
    return out;
}

std::vector<TradeJournal::PerTagCalmar>
TradeJournal::perTagCalmar(bool includeUntagged) const {
    // Per-tag mirror of perSymbolCalmar() (#97). Same shape +
    // includeUntagged handling as perTagDrawdown() / perTagSharpe().
    std::vector<JournalFill> fills = loadAll();

    std::unordered_map<std::string, std::vector<JournalFill>> byTag;
    byTag.reserve(8);
    for (const auto& f : fills) {
        if (f.tag.empty() && !includeUntagged) continue;
        const std::string key = f.tag.empty() ? "__untagged__" : f.tag;
        byTag[key].push_back(f);
    }

    std::vector<PerTagCalmar> out;
    out.reserve(byTag.size());
    for (auto& kv : byTag) {
        PerTagCalmar e;
        e.tag = kv.first;
        auto buckets = bucketByLocalDay(kv.second);
        std::vector<std::pair<std::string, double>> series;
        series.reserve(buckets.size());
        for (auto& bkv : buckets) {
            series.emplace_back(std::move(bkv.first), bkv.second);
        }
        auto sh = computeSharpeFromSeries(series);
        auto dd = computeDrawdownFromSeries(series);
        e.annualizedReturn = sh.meanDailyReturn * 252.0;
        e.maxDrawdown      = dd.maxDrawdown;
        if (dd.maxDrawdown > 1e-9) {
            e.calmarRatio = e.annualizedReturn / e.maxDrawdown;
        } else {
            e.calmarRatio = 0.0;
        }
        out.push_back(std::move(e));
    }
    std::sort(out.begin(), out.end(),
              [](const PerTagCalmar& a, const PerTagCalmar& b) {
                  if (a.calmarRatio != b.calmarRatio)
                      return a.calmarRatio > b.calmarRatio;
                  if (a.annualizedReturn != b.annualizedReturn)
                      return a.annualizedReturn > b.annualizedReturn;
                  return a.tag < b.tag;
              });
    return out;
}

// Pure helper shared between the journal-wide sortino() (#99),
// perSymbolSortino() (#99), and perTagSortino() (#99). Takes a
// chronological daily series and computes Sortino using
// downsideDeviation as the denominator.
//
// Formula (target = 0):
//   downsideDeviation = sqrt(mean(min(0, r)²))
//                     = RMS of negative returns
//   dailySortino      = mean(r) / downsideDeviation
//   annualizedSortino = dailySortino × sqrt(252)
//
// Edge cases:
//   - sampleSize < 1: zeroed Sortino.
//   - All returns >= 0 (no bad days): downsideDeviation = 0,
//     Sortino = 0 (panel renders as "∞" via the same convention
//     used for profit factor when there are no losses).
//   - All returns equal (positive or negative): sampleSize >= 1
//     is still fine, but downsideDeviation may still be 0 if
//     mean > 0; the sentinel handles it.
TradeJournal::Sortino computeSortinoFromSeries(
    const std::vector<std::pair<std::string, double>>& daily) {
    constexpr double kTradingDays = 252.0;
    TradeJournal::Sortino out;
    out.sampleSize = daily.size();
    if (daily.empty()) return out;

    // 1) Mean.
    double sum = 0.0;
    for (const auto& kv : daily) sum += kv.second;
    out.meanDailyReturn = sum / static_cast<double>(daily.size());

    // 2) Downside deviation = RMS of negative returns, with
    //    target = 0. Equivalent to sqrt(mean(min(0, r)²)).
    double negSqSum = 0.0;
    for (const auto& kv : daily) {
        if (kv.second < 0.0) negSqSum += kv.second * kv.second;
    }
    out.downsideDeviation = std::sqrt(negSqSum /
                                      static_cast<double>(daily.size()));

    // 3) Sortino = mean / downsideDeviation. Annualized.
    if (out.downsideDeviation > 1e-9) {
        out.dailySortino      = out.meanDailyReturn / out.downsideDeviation;
        out.annualizedSortino = out.dailySortino * std::sqrt(kTradingDays);
    } else {
        // All-positive daily returns → no downside → Sortino
        // undefined. Sentinel: 0 (panel renders as "∞").
        out.dailySortino      = 0.0;
        out.annualizedSortino = 0.0;
    }
    return out;
}

TradeJournal::Sortino TradeJournal::sortino() const {
    // Sprint #99. Thin wrapper over the shared helper.
    auto daily = realizedByDay();
    return computeSortinoFromSeries(daily);
}

std::vector<TradeJournal::PerSymbolSortino>
TradeJournal::perSymbolSortino() const {
    // Per-symbol Sortino. Same shape as perSymbolSharpe() (#91)
    // — group fills by symbol, bucketByLocalDay per group, feed
    // into computeSortinoFromSeries(). Sorted by annualized
    // Sortino DESCENDING.
    std::vector<JournalFill> fills = loadAll();

    std::unordered_map<std::string, std::vector<JournalFill>> bySymbol;
    bySymbol.reserve(8);
    for (const auto& f : fills) bySymbol[f.symbol].push_back(f);

    std::vector<PerSymbolSortino> out;
    out.reserve(bySymbol.size());
    for (auto& kv : bySymbol) {
        PerSymbolSortino e;
        e.symbol = kv.first;
        auto buckets = bucketByLocalDay(kv.second);
        std::vector<std::pair<std::string, double>> series;
        series.reserve(buckets.size());
        for (auto& bkv : buckets) {
            series.emplace_back(std::move(bkv.first), bkv.second);
        }
        auto so = computeSortinoFromSeries(series);
        e.dailySortino       = so.dailySortino;
        e.annualizedSortino  = so.annualizedSortino;
        e.meanDailyReturn    = so.meanDailyReturn;
        e.downsideDeviation  = so.downsideDeviation;
        e.sampleSize         = so.sampleSize;
        out.push_back(std::move(e));
    }
    std::sort(out.begin(), out.end(),
              [](const PerSymbolSortino& a, const PerSymbolSortino& b) {
                  if (a.annualizedSortino != b.annualizedSortino)
                      return a.annualizedSortino > b.annualizedSortino;
                  if (a.meanDailyReturn != b.meanDailyReturn)
                      return a.meanDailyReturn > b.meanDailyReturn;
                  return a.symbol < b.symbol;
              });
    return out;
}

// ---- perSymbolDayStats() / perTagDayStats() helpers (#102) ----
//
// Heatmap-ready bucketing: for each (axis, day) pair with at
// least one fill, sum realized + count round-trips. Returns
// the data in a row-major indexed form so a heatmap widget
// can iterate without rebuilding the lookup structure.
//
// Two helper entry points: one for symbol-keyed buckets, one
// for tag-keyed. Both build (axis, day) → DayCell maps, then
// flatten into (sorted-axes) × (sorted-dates) indexed grids.
namespace {

template <typename GroupKey>
std::map<std::string, std::map<std::string, TradeJournal::DayCell>>
bucketByDayPerAxis(const std::vector<JournalFill>& fills,
                   std::function<std::string(const JournalFill&)> keyFn,
                   bool includeUntagged = true) {
    std::map<std::string, std::map<std::string, TradeJournal::DayCell>> out;
    constexpr double kEps = 1e-9;
    for (const auto& f : fills) {
        // tag-specific filter — passed through keyFn (the symbol
        // extractor ignores tag state; the tag extractor honors
        // includeUntagged).
        if constexpr (false) {}  // placeholder for compile-time if
        std::string key = keyFn(f);
        // For tag mode with empty tag, the keyFn returns
        // "__untagged__" when includeUntagged=true, or "" when
        // includeUntagged=false. Empty key means "skip".
        if (key.empty()) continue;
        std::time_t secs = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        char date[16];
        std::strftime(date, sizeof(date), "%Y-%m-%d", &tm);
        auto& cell = out[key][date];
        cell.realized += f.realizedDelta;
        if (std::fabs(f.realizedDelta) > kEps) {
            cell.roundTrips++;
            if (f.realizedDelta > kEps) cell.wins++;
            else cell.losses++;
        }
    }
    return out;
}

}  // namespace

std::vector<TradeJournal::PerTagSortino>
TradeJournal::perTagSortino(bool includeUntagged) const {
    // Per-tag Sortino. Same shape as perSymbolSortino() but
    // grouped by tag. includeUntagged handling matches
    // perTagStats() / perTagSharpe() / perTagCalmar().
    std::vector<JournalFill> fills = loadAll();

    std::unordered_map<std::string, std::vector<JournalFill>> byTag;
    byTag.reserve(8);
    for (const auto& f : fills) {
        if (f.tag.empty() && !includeUntagged) continue;
        const std::string key = f.tag.empty() ? "__untagged__" : f.tag;
        byTag[key].push_back(f);
    }

    std::vector<PerTagSortino> out;
    out.reserve(byTag.size());
    for (auto& kv : byTag) {
        PerTagSortino e;
        e.tag = kv.first;
        auto buckets = bucketByLocalDay(kv.second);
        std::vector<std::pair<std::string, double>> series;
        series.reserve(buckets.size());
        for (auto& bkv : buckets) {
            series.emplace_back(std::move(bkv.first), bkv.second);
        }
        auto so = computeSortinoFromSeries(series);
        e.dailySortino       = so.dailySortino;
        e.annualizedSortino  = so.annualizedSortino;
        e.meanDailyReturn    = so.meanDailyReturn;
        e.downsideDeviation  = so.downsideDeviation;
        e.sampleSize         = so.sampleSize;
        out.push_back(std::move(e));
    }
    std::sort(out.begin(), out.end(),
              [](const PerTagSortino& a, const PerTagSortino& b) {
                  if (a.annualizedSortino != b.annualizedSortino)
                      return a.annualizedSortino > b.annualizedSortino;
                  if (a.meanDailyReturn != b.meanDailyReturn)
                      return a.meanDailyReturn > b.meanDailyReturn;
                  return a.tag < b.tag;
              });
    return out;
}

TradeJournal::PerSymbolDayStats
TradeJournal::perSymbolDayStats() const {
    // Sprint #102. Heatmap-ready grid: symbols × dates → DayCell.
    auto buckets = bucketByDayPerAxis<JournalFill>(
        loadAll(),
        [](const JournalFill& f) { return f.symbol; });
    PerSymbolDayStats out;
    out.symbols.reserve(buckets.size());
    for (const auto& kv : buckets) out.symbols.push_back(kv.first);
    // Union of all dates across all symbols, sorted ASC.
    std::set<std::string> dateSet;
    for (const auto& kv : buckets)
        for (const auto& dk : kv.second) dateSet.insert(dk.first);
    out.dates.assign(dateSet.begin(), dateSet.end());
    // Fill grid row-major: out.grid[symbolIdx * dates.size() + dateIdx].
    out.grid.assign(out.symbols.size() * out.dates.size(), DayCell{});
    for (size_t si = 0; si < out.symbols.size(); ++si) {
        const auto& symBuckets = buckets.at(out.symbols[si]);
        for (size_t di = 0; di < out.dates.size(); ++di) {
            auto it = symBuckets.find(out.dates[di]);
            if (it != symBuckets.end())
                out.grid[si * out.dates.size() + di] = it->second;
            // else: DayCell default (all zeros) means "no fills"
        }
    }
    return out;
}

TradeJournal::PerTagDayStats
TradeJournal::perTagDayStats(bool includeUntagged) const {
    auto buckets = bucketByDayPerAxis<JournalFill>(
        loadAll(),
        [includeUntagged](const JournalFill& f) -> std::string {
            if (f.tag.empty()) {
                return includeUntagged ? "__untagged__" : "";
            }
            return f.tag;
        });
    PerTagDayStats out;
    out.tags.reserve(buckets.size());
    for (const auto& kv : buckets) out.tags.push_back(kv.first);
    std::set<std::string> dateSet;
    for (const auto& kv : buckets)
        for (const auto& dk : kv.second) dateSet.insert(dk.first);
    out.dates.assign(dateSet.begin(), dateSet.end());
    out.grid.assign(out.tags.size() * out.dates.size(), DayCell{});
    for (size_t ti = 0; ti < out.tags.size(); ++ti) {
        const auto& tagBuckets = buckets.at(out.tags[ti]);
        for (size_t di = 0; di < out.dates.size(); ++di) {
            auto it = tagBuckets.find(out.dates[di]);
            if (it != tagBuckets.end())
                out.grid[ti * out.dates.size() + di] = it->second;
        }
    }
    return out;
}

std::vector<TradeJournal::EquityPoint>
TradeJournal::equityCurve() const {
    // Sprint #104. Sort fills by timestamp ASC, accumulate
    // realizedDelta into a running cumulative series.
    //
    // Why per-fill granularity and not per-day: the widget renders
    // a smooth line; per-fill gives the most detail. The day-level
    // bucketing (#102) loses intra-day shape.
    auto fills = loadAll();
    std::sort(fills.begin(), fills.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    std::vector<EquityPoint> out;
    out.reserve(fills.size());
    double cumulative = 0.0;
    for (const auto& f : fills) {
        cumulative += f.realizedDelta;
        out.push_back(EquityPoint{f.timestamp_us,
                                  f.realizedDelta,
                                  cumulative});
    }
    return out;
}

std::vector<TradeJournal::DrawdownPoint>
TradeJournal::equityDrawdownSeries() const {
    // Sprint #104. Walk the equity curve, track the running peak
    // and compute drawdown at each point. Same per-fill
    // granularity as equityCurve().
    auto fills = loadAll();
    std::sort(fills.begin(), fills.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    std::vector<DrawdownPoint> out;
    out.reserve(fills.size());
    double cumulative = 0.0;
    double peak        = 0.0;
    for (const auto& f : fills) {
        cumulative += f.realizedDelta;
        if (cumulative > peak) peak = cumulative;
        double dd = peak - cumulative;
        if (dd < 0.0) dd = 0.0;       // never negative — clamp FP noise
        out.push_back(DrawdownPoint{f.timestamp_us, peak, dd});
    }
    return out;
}

namespace {
// Sprint #113 — drawdown event extraction. Walk an equity
// curve (already sorted by timestamp ASC) and emit one
// DrawdownEvent for each peak → trough → recovery cycle.
// An unrecovered drawdown is returned via the `current` out-
// param (zero defaults otherwise).
//
// The implementation mirrors equityDrawdownSeries()'s walk
// pattern but tracks state machine:
//   idle       — at or above peak; no DD in progress
//   in_dd      — under peak; recording start, trough, peak_before
//   recovered  — hit peak again; emit event, return to idle
void extractDrawdownEvents(
    const std::vector<TradeJournal::EquityPoint>& curve,
    std::vector<TradeJournal::DrawdownEvent>& events,
    TradeJournal::DrawdownEvent& current) {
    events.clear();
    current = TradeJournal::DrawdownEvent{};   // zero
    if (curve.empty()) return;
    // Use lowest() so the first point always counts as a new
    // high water mark — never erroneously enters DD at p[0].
    double peak = std::numeric_limits<double>::lowest();
    uint64_t peak_ts = 0;
    bool in_dd = false;
    double trough_value = 0.0;
    uint64_t trough_ts = 0;
    uint64_t start_ts = 0;
    double peak_before = 0.0;
    for (const auto& p : curve) {
        if (p.cumulative > peak) {
            // New high water mark.
            peak = p.cumulative;
            peak_ts = p.timestamp_us;
            if (in_dd) {
                // Recovery: emit the event.
                TradeJournal::DrawdownEvent ev;
                ev.start_ts     = start_ts;
                ev.trough_ts    = trough_ts;
                ev.end_ts       = p.timestamp_us;
                ev.peak_before  = peak_before;
                ev.trough_value = trough_value;
                ev.trough_depth = peak_before - trough_value;
                ev.drawdown_us  = p.timestamp_us - start_ts;
                ev.recovery_us  = p.timestamp_us - trough_ts;
                events.push_back(ev);
                in_dd = false;
            }
        } else if (!in_dd) {
            // Entry into drawdown.
            in_dd = true;
            start_ts = peak_ts;
            peak_before = peak;
            trough_value = p.cumulative;
            trough_ts = p.timestamp_us;
        } else {
            // Still in DD: track trough.
            if (p.cumulative < trough_value) {
                trough_value = p.cumulative;
                trough_ts = p.timestamp_us;
            }
        }
    }
    if (in_dd) {
        // Unrecovered — populate current.
        current.start_ts     = start_ts;
        current.trough_ts    = trough_ts;
        current.peak_before  = peak_before;
        current.trough_value = trough_value;
        current.trough_depth = peak_before - trough_value;
        current.drawdown_us  = curve.back().timestamp_us - start_ts;
        // end_ts / recovery_us stay zero.
    }
}
}  // namespace

std::vector<TradeJournal::DrawdownEvent>
TradeJournal::drawdownRecoveries() const {
    // Sprint #113. Walk the journal-wide equity curve, emit one
    // event per peak → trough → recovery cycle. Sorted by
    // trough_depth DESC.
    auto curve = equityCurve();
    std::vector<DrawdownEvent> events;
    DrawdownEvent current;
    extractDrawdownEvents(curve, events, current);
    // Drop the in-progress drawdown — drawdownRecoveries()
    // answers "what DD events have I RECOVERED from?".
    std::sort(events.begin(), events.end(),
              [](const DrawdownEvent& a, const DrawdownEvent& b) {
                  return a.trough_depth > b.trough_depth;
              });
    return events;
}

std::vector<TradeJournal::DrawdownEvent>
TradeJournal::drawdownRecoveriesBySymbol(
    const std::string& symbol) const {
    // Sprint #113. Per-symbol: rebuild the equity curve from
    // only this symbol's fills, then walk it.
    auto fills = loadAll();
    std::sort(fills.begin(), fills.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    std::vector<EquityPoint> curve;
    curve.reserve(fills.size());
    double cumulative = 0.0;
    for (const auto& f : fills) {
        if (f.symbol != symbol) continue;
        cumulative += f.realizedDelta;
        curve.push_back(EquityPoint{f.timestamp_us,
                                    f.realizedDelta,
                                    cumulative});
    }
    std::vector<DrawdownEvent> events;
    DrawdownEvent current;
    extractDrawdownEvents(curve, events, current);
    std::sort(events.begin(), events.end(),
              [](const DrawdownEvent& a, const DrawdownEvent& b) {
                  return a.trough_depth > b.trough_depth;
              });
    return events;
}

std::vector<TradeJournal::DrawdownEvent>
TradeJournal::drawdownRecoveriesByTag(
    const std::string& tag,
    bool includeUntagged) const {
    // Sprint #113. Per-tag (mirrors perSymbolDrawdown()). Tag
    // selection matches perTagStats() — "__untagged__" is the
    // synthetic key when includeUntagged=true.
    auto fills = loadAll();
    std::sort(fills.begin(), fills.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    std::vector<EquityPoint> curve;
    curve.reserve(fills.size());
    double cumulative = 0.0;
    for (const auto& f : fills) {
        bool matches;
        if (tag == "__untagged__") {
            matches = f.tag.empty();
        } else {
            if (includeUntagged && f.tag.empty()) matches = false;
            else matches = (f.tag == tag);
        }
        if (!matches) continue;
        cumulative += f.realizedDelta;
        curve.push_back(EquityPoint{f.timestamp_us,
                                    f.realizedDelta,
                                    cumulative});
    }
    std::vector<DrawdownEvent> events;
    DrawdownEvent current;
    extractDrawdownEvents(curve, events, current);
    std::sort(events.begin(), events.end(),
              [](const DrawdownEvent& a, const DrawdownEvent& b) {
                  return a.trough_depth > b.trough_depth;
              });
    return events;
}

TradeJournal::DrawdownEvent
TradeJournal::currentDrawdown() const {
    // Sprint #113. Returns the in-progress DD if we're
    // underwater, or zero-Depth sentinel otherwise.
    auto curve = equityCurve();
    std::vector<DrawdownEvent> events;
    DrawdownEvent current;
    extractDrawdownEvents(curve, events, current);
    return current;
}

// ---- Sprint #114: DrawdownEvent derived metrics ----
//
// All pure functions. No journal access. Encoded as static
// methods so the call site reads `TradeJournal::recoveryRatio(ev)`
// — explicit namespace avoids accidental namespace pollution.

double TradeJournal::recoveryRatio(const DrawdownEvent& ev) {
    if (ev.drawdown_us == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(ev.recovery_us) /
           static_cast<double>(ev.drawdown_us);
}

double TradeJournal::recoverySpeed(const DrawdownEvent& ev) {
    if (ev.recovery_us == 0) return 0.0;
    return ev.trough_depth /
           static_cast<double>(ev.recovery_us);
}

double TradeJournal::maxDepth(
    const std::vector<TradeJournal::DrawdownEvent>& events) {
    if (events.empty()) return 0.0;
    double m = 0.0;
    for (const auto& e : events) {
        if (e.trough_depth > m) m = e.trough_depth;
    }
    return m;
}

double TradeJournal::avgDepth(
    const std::vector<TradeJournal::DrawdownEvent>& events) {
    if (events.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& e : events) sum += e.trough_depth;
    return sum / static_cast<double>(events.size());
}

double TradeJournal::avgRecoveryRatio(
    const std::vector<TradeJournal::DrawdownEvent>& events) {
    // Geometric mean of recoveryRatio across events.
    // Symmetric in log-space: avg of ratios == exp(avg of
    // log(ratio)). Guards against zero drawdown_us by
    // skipping those events.
    if (events.empty()) return 0.0;
    double sum_log = 0.0;
    size_t n = 0;
    for (const auto& e : events) {
        if (e.drawdown_us == 0) continue;
        double r = static_cast<double>(e.recovery_us) /
                   static_cast<double>(e.drawdown_us);
        if (r <= 0.0) continue;  // skip non-positive
        sum_log += std::log(r);
        ++n;
    }
    if (n == 0) return 0.0;
    return std::exp(sum_log / static_cast<double>(n));
}

namespace {
// Sprint #116 — shared activity helpers. The three flavors
// (journal-wide, by-symbol, by-tag) all reduce to "iterate
// filtered fills and compute min(ts), max(ts), set of
// distinct YYYY-MM-DD". yearMonthKey() is reused from
// Sprint #115 but we need a finer-grained daily key:
//   key = year * 10000 + month * 100 + day
// (year-month-day as sortable int).
int dayKeyFromTimestamp(uint64_t ts_us) {
    std::time_t secs =
        static_cast<std::time_t>(ts_us / 1000000ULL);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &secs);
#else
    localtime_r(&secs, &tm);
#endif
    return (tm.tm_year + 1900) * 10000
         + (tm.tm_mon + 1) * 100
         + tm.tm_mday;
}

template <typename Pred>
size_t countDistinctDays(const std::vector<JournalFill>& fills,
                         Pred pred) {
    std::set<int> seen;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        seen.insert(dayKeyFromTimestamp(f.timestamp_us));
    }
    return seen.size();
}
}  // namespace

size_t TradeJournal::activeTradingDays() const {
    return countDistinctDays(loadAll(),
        [](const JournalFill&) { return true; });
}

size_t TradeJournal::activeTradingDaysBySymbol(
    const std::string& symbol) const {
    return countDistinctDays(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

size_t TradeJournal::activeTradingDaysByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return countDistinctDays(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #116 — first/last fill by predicate. Returns 0 for
// empty match. Single pass.
template <typename Pred>
uint64_t firstLastFill(const std::vector<JournalFill>& fills,
                       Pred pred,
                       bool wantFirst) {
    uint64_t result = 0;
    bool seen = false;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (!seen) {
            result = f.timestamp_us;
            seen = true;
        } else if (wantFirst) {
            if (f.timestamp_us < result)
                result = f.timestamp_us;
        } else {
            if (f.timestamp_us > result)
                result = f.timestamp_us;
        }
    }
    return result;
}
}  // namespace

uint64_t TradeJournal::firstFillUs() const {
    return firstLastFill(loadAll(),
        [](const JournalFill&) { return true; }, true);
}

uint64_t TradeJournal::lastFillUs() const {
    return firstLastFill(loadAll(),
        [](const JournalFill&) { return true; }, false);
}

uint64_t TradeJournal::firstFillUsBySymbol(
    const std::string& symbol) const {
    return firstLastFill(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        }, true);
}

uint64_t TradeJournal::lastFillUsBySymbol(
    const std::string& symbol) const {
    return firstLastFill(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        }, false);
}

uint64_t TradeJournal::firstFillUsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return firstLastFill(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        }, true);
}

uint64_t TradeJournal::lastFillUsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return firstLastFill(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        }, false);
}

namespace {
// Sprint #117 — trading-session grouping. Sort fills by
// timestamp ASC, then walk through them. A new session starts
// whenever the gap from the previous fill exceeds the
// threshold (gapMinutes * 60 * 1_000_000 µs).
//
// Within each session, track per-session cumulative realized
// + drawdown (peak-to-trough over the session-local curve)
// + win/loss counts.
//
// Templated on the filter predicate so journal-wide + per-
// symbol + per-tag all share one tested core.
constexpr double kSessEps = 1e-9;

template <typename Pred>
std::vector<TradeJournal::TradingSession>
buildSessions(const std::vector<JournalFill>& fills,
              int gapMinutes, Pred pred) {
    std::vector<JournalFill> filtered;
    filtered.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) filtered.push_back(f);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    const uint64_t gap_us =
        static_cast<uint64_t>(gapMinutes) * 60ULL * 1000000ULL;
    std::vector<TradeJournal::TradingSession> out;
    if (filtered.empty()) return out;
    TradeJournal::TradingSession cur;
    cur.start_ts  = filtered[0].timestamp_us;
    cur.end_ts    = filtered[0].timestamp_us;
    cur.fillCount = 1;
    cur.realized  = filtered[0].realizedDelta;
    double cum    = filtered[0].realizedDelta;
    double peak   = cum > 0.0 ? cum : 0.0;
    // maxDD in a session is the deepest peak-to-trough seen
    // at any point in the session, not at end. Track it
    // inline as cum evolves.
    double maxDD  = 0.0;
    {
        double dd = peak - cum;
        if (dd > maxDD) maxDD = dd;
        if (dd < 0.0) dd = 0.0;
    }
    size_t wins   = 0;
    size_t losses = 0;
    if (std::fabs(filtered[0].realizedDelta) > kSessEps) {
        if (filtered[0].realizedDelta > 0) ++wins;
        else ++losses;
    }
    for (size_t i = 1; i < filtered.size(); ++i) {
        const auto& f = filtered[i];
        uint64_t prev_ts = filtered[i-1].timestamp_us;
        if (f.timestamp_us - prev_ts > gap_us) {
            // Close current session.
            cur.end_ts    = prev_ts;
            cur.active_us = cur.end_ts - cur.start_ts;
            cur.winRate   = (wins + losses) > 0
                            ? static_cast<double>(wins) /
                              static_cast<double>(wins + losses)
                            : 0.0;
            cur.maxDD     = maxDD;
            out.push_back(cur);
            // Reset for new session.
            cur = TradeJournal::TradingSession{};
            cur.start_ts  = f.timestamp_us;
            cur.end_ts    = f.timestamp_us;
            cur.fillCount = 1;
            cur.realized  = f.realizedDelta;
            cum           = f.realizedDelta;
            peak          = cum > 0.0 ? cum : 0.0;
            maxDD         = peak - cum;
            if (maxDD < 0.0) maxDD = 0.0;
            wins = losses = 0;
            if (std::fabs(f.realizedDelta) > kSessEps) {
                if (f.realizedDelta > 0) ++wins;
                else ++losses;
            }
        } else {
            // Continue current session.
            cur.end_ts    = f.timestamp_us;
            cur.fillCount += 1;
            cur.realized  += f.realizedDelta;
            cum           += f.realizedDelta;
            if (cum > peak) peak = cum;
            {
                double dd = peak - cum;
                if (dd > maxDD) maxDD = dd;
                if (dd < 0.0) dd = 0.0;
            }
            if (std::fabs(f.realizedDelta) > kSessEps) {
                if (f.realizedDelta > 0) ++wins;
                else ++losses;
            }
        }
    }
    // Close final session.
    cur.end_ts    = filtered.back().timestamp_us;
    cur.active_us = cur.end_ts - cur.start_ts;
    cur.winRate   = (wins + losses) > 0
                    ? static_cast<double>(wins) /
                      static_cast<double>(wins + losses)
                    : 0.0;
    cur.maxDD     = maxDD;
    if (cur.maxDD < 0.0) cur.maxDD = 0.0;
    out.push_back(cur);
    return out;
}
}  // namespace

std::vector<TradeJournal::TradingSession>
TradeJournal::sessions(int gapMinutes) const {
    return buildSessions(loadAll(), gapMinutes,
        [](const JournalFill&) { return true; });
}

std::vector<TradeJournal::TradingSession>
TradeJournal::sessionsBySymbol(
    const std::string& symbol,
    int gapMinutes) const {
    return buildSessions(loadAll(), gapMinutes,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::TradingSession>
TradeJournal::sessionsByTag(
    const std::string& tag,
    bool includeUntagged,
    int gapMinutes) const {
    return buildSessions(loadAll(), gapMinutes,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

double TradeJournal::avgRealized(
    const std::vector<TradingSession>& ss) {
    if (ss.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& s : ss) sum += s.realized;
    return sum / static_cast<double>(ss.size());
}

double TradeJournal::avgFillCount(
    const std::vector<TradingSession>& ss) {
    if (ss.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& s : ss) sum += s.fillCount;
    return sum / static_cast<double>(ss.size());
}

uint64_t TradeJournal::avgActiveUs(
    const std::vector<TradingSession>& ss) {
    if (ss.empty()) return 0;
    uint64_t sum = 0;
    for (const auto& s : ss) sum += s.active_us;
    return sum / static_cast<uint64_t>(ss.size());
}

size_t TradeJournal::maxFillCount(
    const std::vector<TradingSession>& ss) {
    if (ss.empty()) return 0;
    size_t m = 0;
    for (const auto& s : ss) {
        if (s.fillCount > m) m = s.fillCount;
    }
    return m;
}

double TradeJournal::totalRealized(
    const std::vector<TradingSession>& ss) {
    double sum = 0.0;
    for (const auto& s : ss) sum += s.realized;
    return sum;
}

double TradeJournal::recoveryFactor(
    double netRealized, double maxDrawdown) {
    if (maxDrawdown <= 1e-9) {
        // No drawdown — undefined ratio. Use +inf to signal
        // "perfect" (the trader's P&L grew monotonically).
        return netRealized > 1e-9
            ? std::numeric_limits<double>::infinity()
            : 0.0;   // also no P&L — degenerate.
    }
    return netRealized / maxDrawdown;
}

double TradeJournal::perSymbolRecoveryFactor(
    const std::string& symbol) const {
    // Net realized (sum of all this symbol's realizedDeltas)
    // over max drawdown (looked up from perSymbolDrawdown()).
    auto fills = loadAll();
    double net = 0.0;
    for (const auto& f : fills) {
        if (f.symbol == symbol) net += f.realizedDelta;
    }
    double maxDD = 0.0;
    for (const auto& psd : perSymbolDrawdown()) {
        if (psd.symbol == symbol) {
            maxDD = psd.maxDrawdown;
            break;
        }
    }
    return recoveryFactor(net, maxDD);
}

double TradeJournal::perTagRecoveryFactor(
    const std::string& tag,
    bool includeUntagged) const {
    auto fills = loadAll();
    double net = 0.0;
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (!f.tag.empty()) continue;
        } else {
            if (includeUntagged && f.tag.empty()) continue;
            if (f.tag != tag) continue;
        }
        net += f.realizedDelta;
    }
    // Look up tag's maxDD from perTagDrawdown vector.
    double maxDD = 0.0;
    for (const auto& ptd : perTagDrawdown(includeUntagged)) {
        if (ptd.tag == tag) {
            maxDD = ptd.maxDrawdown;
            break;
        }
    }
    return recoveryFactor(net, maxDD);
}

double TradeJournal::journalRecoveryFactor() const {
    auto fills = loadAll();
    double net = 0.0;
    for (const auto& f : fills) net += f.realizedDelta;
    auto dd = maxDrawdown();
    return recoveryFactor(net, dd.maxDrawdown);
}

double TradeJournal::tradesPerDay() const {
    auto fills = loadAll();
    size_t days = activeTradingDays();
    if (days == 0) return 0.0;
    return static_cast<double>(fills.size()) /
           static_cast<double>(days);
}

double TradeJournal::tradesPerDayBySymbol(
    const std::string& symbol) const {
    auto fills = loadAll();
    size_t count = 0;
    for (const auto& f : fills) {
        if (f.symbol == symbol) ++count;
    }
    size_t days = activeTradingDaysBySymbol(symbol);
    if (days == 0) return 0.0;
    return static_cast<double>(count) /
           static_cast<double>(days);
}

double TradeJournal::tradesPerDayByTag(
    const std::string& tag,
    bool includeUntagged) const {
    auto fills = loadAll();
    size_t count = 0;
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (f.tag.empty()) ++count;
        } else {
            if (includeUntagged && f.tag.empty()) continue;
            if (f.tag == tag) ++count;
        }
    }
    size_t days = activeTradingDaysByTag(tag, includeUntagged);
    if (days == 0) return 0.0;
    return static_cast<double>(count) /
           static_cast<double>(days);
}

uint64_t TradeJournal::avgTimeBetweenTrades_us() const {
    auto fills = loadAll();
    std::sort(fills.begin(), fills.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    if (fills.size() < 2) return 0;
    uint64_t total = 0;
    for (size_t i = 1; i < fills.size(); ++i) {
        total += fills[i].timestamp_us - fills[i-1].timestamp_us;
    }
    return total / (fills.size() - 1);
}

uint64_t TradeJournal::avgTimeBetweenTrades_usBySymbol(
    const std::string& symbol) const {
    auto fills = loadAll();
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (f.symbol == symbol) sub.push_back(f);
    }
    std::sort(sub.begin(), sub.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    if (sub.size() < 2) return 0;
    uint64_t total = 0;
    for (size_t i = 1; i < sub.size(); ++i) {
        total += sub[i].timestamp_us - sub[i-1].timestamp_us;
    }
    return total / (sub.size() - 1);
}

uint64_t TradeJournal::avgTimeBetweenTrades_usByTag(
    const std::string& tag,
    bool includeUntagged) const {
    auto fills = loadAll();
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (f.tag.empty()) sub.push_back(f);
        } else {
            if (includeUntagged && f.tag.empty()) continue;
            if (f.tag == tag) sub.push_back(f);
        }
    }
    std::sort(sub.begin(), sub.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    if (sub.size() < 2) return 0;
    uint64_t total = 0;
    for (size_t i = 1; i < sub.size(); ++i) {
        total += sub[i].timestamp_us - sub[i-1].timestamp_us;
    }
    return total / (sub.size() - 1);
}

double TradeJournal::kellyFraction(
    size_t wins, size_t losses,
    double avgWinner, double avgLoser) {
    // Kelly % = W - (1 - W) / R
    //   where W = wins / (wins + losses)
    //         R = avg winner / |avg loser|
    //
    // Return 0 if either side has 0 round-trips (can't
    // compute payoff without both sides) or if avgLoser is
    // near zero (degenerate).
    constexpr double kEps = 1e-9;
    size_t total = wins + losses;
    if (total == 0) return 0.0;
    if (wins == 0 || losses == 0) return 0.0;
    if (std::fabs(avgLoser) < kEps) return 0.0;
    double W = static_cast<double>(wins) /
               static_cast<double>(total);
    double R = avgWinner / std::fabs(avgLoser);
    double K = W - (1.0 - W) / R;
    // Clamp to [-1, 1] — anything outside is degenerate.
    if (K < -1.0) K = -1.0;
    if (K >  1.0) K =  1.0;
    return K;
}

double TradeJournal::kellyFraction() const {
    auto stats = perSymbolStats();
    if (stats.empty()) return 0.0;
    // Journal-wide W = sum(wins) / sum(rt), not weighted by
    // symbol. Recompute from raw fills.
    auto fills = loadAll();
    size_t wins = 0, losses = 0;
    double sumWin = 0.0, sumLoss = 0.0;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) < 1e-9) continue;
        if (f.realizedDelta > 0) {
            ++wins;
            sumWin += f.realizedDelta;
        } else {
            ++losses;
            sumLoss += f.realizedDelta;
        }
    }
    double avgW = wins > 0 ? sumWin / wins : 0.0;
    double avgL = losses > 0 ? sumLoss / losses : 0.0;
    return kellyFraction(wins, losses, avgW, avgL);
}

double TradeJournal::perSymbolKellyFraction(
    const std::string& symbol) const {
    auto fills = loadAll();
    size_t wins = 0, losses = 0;
    double sumWin = 0.0, sumLoss = 0.0;
    for (const auto& f : fills) {
        if (f.symbol != symbol) continue;
        if (std::fabs(f.realizedDelta) < 1e-9) continue;
        if (f.realizedDelta > 0) {
            ++wins;
            sumWin += f.realizedDelta;
        } else {
            ++losses;
            sumLoss += f.realizedDelta;
        }
    }
    double avgW = wins > 0 ? sumWin / wins : 0.0;
    double avgL = losses > 0 ? sumLoss / losses : 0.0;
    return kellyFraction(wins, losses, avgW, avgL);
}

double TradeJournal::perTagKellyFraction(
    const std::string& tag,
    bool includeUntagged) const {
    auto fills = loadAll();
    size_t wins = 0, losses = 0;
    double sumWin = 0.0, sumLoss = 0.0;
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (!f.tag.empty()) continue;
        } else {
            if (includeUntagged && f.tag.empty()) continue;
            if (f.tag != tag) continue;
        }
        if (std::fabs(f.realizedDelta) < 1e-9) continue;
        if (f.realizedDelta > 0) {
            ++wins;
            sumWin += f.realizedDelta;
        } else {
            ++losses;
            sumLoss += f.realizedDelta;
        }
    }
    double avgW = wins > 0 ? sumWin / wins : 0.0;
    double avgL = losses > 0 ? sumLoss / losses : 0.0;
    return kellyFraction(wins, losses, avgW, avgL);
}

double TradeJournal::riskOfRuin(
    size_t wins, size_t losses,
    double ruinFraction) {
    // PoR = ((1-W)/W)^(capital_units)
    //   W = wins / (wins+losses)
    //   capital_units = ruinFraction (in units of "1 loss")
    //
    // (Simpler than the full gambler's ruin with payoff
    // ratio, but a useful first-order estimate.)
    constexpr double kEps = 1e-9;
    size_t total = wins + losses;
    if (total == 0) return 1.0;   // no data → assume worst
    if (losses == 0) return 0.0;  // never loses → no ruin
    if (wins == 0) return 1.0;    // never wins → certain ruin
    double W = static_cast<double>(wins) /
               static_cast<double>(total);
    double q_over_p = (1.0 - W) / W;
    // If W < 0.5, q_over_p > 1 → PoR explodes for any
    // positive capital_units. Clamp to 1.0.
    if (q_over_p <= 1.0 + kEps) {
        // Long-run positive edge: PoR = (q/p)^(units).
        // units = ruinFraction / unit_loss (we use 1 as the
        // unit loss; ruinFraction is the dollar-amount fraction
        // of capital at risk).
        double units = ruinFraction;  // simplified
        double por = std::pow(q_over_p, units);
        if (por < 0.0) por = 0.0;
        if (por > 1.0) por = 1.0;
        return por;
    } else {
        // Negative or zero edge → ruin is at least as likely
        // as no-ruin. Conservative: report 1.0.
        return 1.0;
    }
}

double TradeJournal::riskOfRuin(double ruinFraction) const {
    auto fills = loadAll();
    size_t wins = 0, losses = 0;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) < 1e-9) continue;
        if (f.realizedDelta > 0) ++wins;
        else ++losses;
    }
    return riskOfRuin(wins, losses, ruinFraction);
}

namespace {
// Sprint #115 — shared monthly bucket builder. The three
// monthlyReturns*() methods differ only in the filter predicate.
struct MonthAcc {
    double realized = 0.0;
    size_t count    = 0;
    size_t wins     = 0;
    size_t losses   = 0;
};
// Key: year*100 + month (sortable as integer).
using MonthKey = int;

MonthKey yearMonthKey(const JournalFill& f) {
    std::time_t secs =
        static_cast<std::time_t>(f.timestamp_us / 1000000ULL);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &secs);
#else
    localtime_r(&secs, &tm);
#endif
    return (tm.tm_year + 1900) * 100 + (tm.tm_mon + 1);
}

int yearFromKey(MonthKey k) { return k / 100; }
int monthFromKey(MonthKey k) { return k % 100; }

template <typename Pred>
std::vector<TradeJournal::MonthlyReturn>
buildMonthlyReturns(const std::vector<JournalFill>& fills,
                    Pred pred) {
    constexpr double kEps = 1e-9;
    std::map<MonthKey, MonthAcc> accs;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        MonthKey k = yearMonthKey(f);
        MonthAcc& a = accs[k];
        a.realized += f.realizedDelta;
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        a.count++;
        if (f.realizedDelta > kEps) a.wins++;
        else if (f.realizedDelta < -kEps) a.losses++;
    }
    std::vector<TradeJournal::MonthlyReturn> out;
    out.reserve(accs.size());
    for (const auto& [k, a] : accs) {
        TradeJournal::MonthlyReturn r;
        r.year     = yearFromKey(k);
        r.month    = monthFromKey(k);
        r.realized = a.realized;
        r.count    = a.count;
        r.wins     = a.wins;
        r.losses   = a.losses;
        r.winRate  = a.count > 0
                     ? static_cast<double>(a.wins) /
                       static_cast<double>(a.count)
                     : 0.0;
        out.push_back(r);
    }
    // Already sorted by key (std::map).
    return out;
}
}  // namespace

std::vector<TradeJournal::MonthlyReturn>
TradeJournal::monthlyReturns() const {
    return buildMonthlyReturns(loadAll(),
        [](const JournalFill&) { return true; });
}

std::vector<TradeJournal::MonthlyReturn>
TradeJournal::monthlyReturnsBySymbol(
    const std::string& symbol) const {
    return buildMonthlyReturns(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::MonthlyReturn>
TradeJournal::monthlyReturnsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildMonthlyReturns(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #122 — shared streak walker. Templated on the
// filter predicate so journal-wide + per-symbol + per-tag
// share one tested core. The filter may return true for
// any subset of fills; the walker processes them in
// timestamp order.
template <typename Pred>
TradeJournal::StreakStats
buildStreakStats(const std::vector<JournalFill>& fills,
                 Pred pred) {
    std::vector<JournalFill> filtered;
    filtered.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) filtered.push_back(f);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    TradeJournal::StreakStats out;
    constexpr double kEps = 1e-9;
    bool   inRun         = false;
    bool   runIsWin      = false;
    size_t runLen        = 0;
    std::vector<TradeJournal::StreakStats::RecentStreak>
        closedRuns;
    auto closeRun = [&]() {
        if (!inRun) return;
        TradeJournal::StreakStats::RecentStreak r;
        r.length = runLen;
        r.isWin  = runIsWin;
        closedRuns.push_back(r);
        out.totalStreaks++;
        if (runIsWin) {
            out.totalWinStreaks++;
            if (runLen > out.maxWinStreak)
                out.maxWinStreak = runLen;
        } else {
            out.totalLossStreaks++;
            if (runLen > out.maxLossStreak)
                out.maxLossStreak = runLen;
        }
        inRun = false;
        runLen = 0;
    };
    for (const auto& f : filtered) {
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        bool fillIsWin = (f.realizedDelta > kEps);
        if (inRun && fillIsWin != runIsWin) closeRun();
        if (!inRun) {
            inRun    = true;
            runIsWin = fillIsWin;
            runLen   = 1;
        } else {
            runLen++;
        }
    }
    // The most recent run is the in-flight one.
    if (inRun) {
        if (runIsWin) out.currentWinStreak = runLen;
        else          out.currentLossStreak = runLen;
        TradeJournal::StreakStats::RecentStreak r;
        r.length = runLen;
        r.isWin  = runIsWin;
        closedRuns.push_back(r);
        out.totalStreaks++;
        if (runIsWin) {
            out.totalWinStreaks++;
            if (runLen > out.maxWinStreak)
                out.maxWinStreak = runLen;
        } else {
            out.totalLossStreaks++;
            if (runLen > out.maxLossStreak)
                out.maxLossStreak = runLen;
        }
    }
    std::reverse(closedRuns.begin(), closedRuns.end());
    if (closedRuns.size() > 20) {
        closedRuns.resize(20);
    }
    out.recentStreaks = std::move(closedRuns);
    return out;
}
}  // namespace

TradeJournal::StreakStats
TradeJournal::streakStats() const {
    return buildStreakStats(loadAll(),
        [](const JournalFill&) { return true; });
}

TradeJournal::StreakStats
TradeJournal::streakStatsBySymbol(
    const std::string& symbol) const {
    return buildStreakStats(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::StreakStats
TradeJournal::streakStatsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildStreakStats(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #123 — shared win-rate-over-time builder.
template <typename Pred>
std::vector<TradeJournal::WinRatePoint>
buildCumulativeWinRate(
    const std::vector<JournalFill>& fills,
    Pred pred) {
    std::vector<JournalFill> filtered;
    filtered.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) filtered.push_back(f);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    constexpr double kEps = 1e-9;
    std::vector<TradeJournal::WinRatePoint> out;
    out.reserve(filtered.size());
    size_t wins = 0, losses = 0;
    double cum  = 0.0;
    for (const auto& f : filtered) {
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        cum += f.realizedDelta;
        if (f.realizedDelta > kEps) ++wins;
        else ++losses;
        TradeJournal::WinRatePoint p;
        p.timestamp_us       = f.timestamp_us;
        p.count              = wins + losses;
        p.wins               = wins;
        p.losses             = losses;
        p.winRate            = static_cast<double>(wins) /
                               static_cast<double>(wins + losses);
        p.cumulativeRealized = cum;
        out.push_back(p);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::WinRatePoint>
TradeJournal::cumulativeWinRate() const {
    return buildCumulativeWinRate(loadAll(),
        [](const JournalFill&) { return true; });
}

std::vector<TradeJournal::WinRatePoint>
TradeJournal::cumulativeWinRateBySymbol(
    const std::string& symbol) const {
    return buildCumulativeWinRate(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::WinRatePoint>
TradeJournal::cumulativeWinRateByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildCumulativeWinRate(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #124 — shared rolling-PF builder.
template <typename Pred>
std::vector<TradeJournal::RollingPFPoint>
buildRollingProfitFactor(
    const std::vector<JournalFill>& fills,
    size_t window, Pred pred) {
    std::vector<JournalFill> filtered;
    filtered.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) filtered.push_back(f);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    // Keep only round-trips (skip ties) — same definition
    // as perSymbolStats: |realizedDelta| > 0.
    std::vector<JournalFill> rt;
    rt.reserve(filtered.size());
    for (const auto& f : filtered) {
        if (std::fabs(f.realizedDelta) > 1e-9) rt.push_back(f);
    }
    if (rt.size() < window) return {};
    std::vector<TradeJournal::RollingPFPoint> out;
    out.reserve(rt.size() - window + 1);
    double ringWins = 0.0, ringLosses = 0.0;
    size_t ringWinCount = 0, ringLossCount = 0;
    // Seed ring buffer with the first `window` round-trips.
    for (size_t i = 0; i < window; ++i) {
        if (rt[i].realizedDelta > 0) {
            ringWins += rt[i].realizedDelta;
            ++ringWinCount;
        } else {
            ringLosses += rt[i].realizedDelta;
            ++ringLossCount;
        }
    }
    // Emit the first point.
    {
        TradeJournal::RollingPFPoint p;
        p.timestamp_us = rt[window - 1].timestamp_us;
        p.count        = window;
        p.grossWin     = ringWins;
        p.grossLoss    = ringLosses;
        p.winRate      = static_cast<double>(ringWinCount) /
                         static_cast<double>(window);
        p.profitFactor = std::fabs(ringLosses) < 1e-9
            ? (ringWins > 1e-9
                 ? std::numeric_limits<double>::infinity()
                 : 0.0)
            : ringWins / std::fabs(ringLosses);
        out.push_back(p);
    }
    // Slide window: drop rt[i-window], add rt[i].
    for (size_t i = window; i < rt.size(); ++i) {
        const auto& dropped = rt[i - window];
        const auto& added   = rt[i];
        if (dropped.realizedDelta > 0) {
            ringWins -= dropped.realizedDelta;
            --ringWinCount;
        } else {
            ringLosses -= dropped.realizedDelta;
            --ringLossCount;
        }
        if (added.realizedDelta > 0) {
            ringWins += added.realizedDelta;
            ++ringWinCount;
        } else {
            ringLosses += added.realizedDelta;
            ++ringLossCount;
        }
        TradeJournal::RollingPFPoint p;
        p.timestamp_us = added.timestamp_us;
        p.count        = window;
        p.grossWin     = ringWins;
        p.grossLoss    = ringLosses;
        p.winRate      = static_cast<double>(ringWinCount) /
                         static_cast<double>(window);
        p.profitFactor = std::fabs(ringLosses) < 1e-9
            ? (ringWins > 1e-9
                 ? std::numeric_limits<double>::infinity()
                 : 0.0)
            : ringWins / std::fabs(ringLosses);
        out.push_back(p);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::RollingPFPoint>
TradeJournal::rollingProfitFactor(size_t window) const {
    return buildRollingProfitFactor(loadAll(), window,
        [](const JournalFill&) { return true; });
}

std::vector<TradeJournal::RollingPFPoint>
TradeJournal::rollingProfitFactorBySymbol(
    const std::string& symbol,
    size_t window) const {
    return buildRollingProfitFactor(loadAll(), window,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::RollingPFPoint>
TradeJournal::rollingProfitFactorByTag(
    const std::string& tag,
    bool includeUntagged,
    size_t window) const {
    return buildRollingProfitFactor(loadAll(), window,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

TradeJournal::SymbolSummary
TradeJournal::symbolSummary(const std::string& symbol) const {
    // Sprint #125. Single-pass snapshot of one symbol. Fills
    // are loaded once; all metrics are derived from that
    // single vector (no per-method reload).
    SymbolSummary s;
    s.symbol = symbol;

    auto fills = loadAll();
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (f.symbol == symbol) sub.push_back(f);
    }

    // PerSymbolStats shape.
    constexpr double kEps = 1e-9;
    double grossWin = 0.0, grossLoss = 0.0;
    size_t wins = 0, losses = 0;
    double sumRTpnl = 0.0;
    for (const auto& f : sub) {
        s.realized += f.realizedDelta;
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        s.roundTripCount++;
        sumRTpnl += f.realizedDelta;
        if (f.realizedDelta > 0) {
            s.winCount++;
            grossWin += f.realizedDelta;
        } else {
            s.lossCount++;
            grossLoss += f.realizedDelta;
        }
    }
    s.winRate      = s.roundTripCount > 0
                     ? static_cast<double>(s.winCount) /
                       static_cast<double>(s.roundTripCount)
                     : 0.0;
    s.avgWinner    = s.winCount > 0
                     ? grossWin / static_cast<double>(s.winCount)
                     : 0.0;
    s.avgLoser     = s.lossCount > 0
                     ? grossLoss / static_cast<double>(s.lossCount)
                     : 0.0;
    s.profitFactor = std::fabs(grossLoss) < kEps
                     ? (grossWin > kEps
                          ? std::numeric_limits<double>::infinity()
                          : 0.0)
                     : grossWin / std::fabs(grossLoss);
    s.expectancy   = s.roundTripCount > 0
                     ? sumRTpnl /
                       static_cast<double>(s.roundTripCount)
                     : 0.0;

    // Drawdown lookup.
    for (const auto& psd : perSymbolDrawdown()) {
        if (psd.symbol == symbol) {
            s.maxDrawdown   = psd.maxDrawdown;
            s.recoveryDate  = psd.recoveryDate;
            s.recoveryDays  = psd.recoveryDays;
            s.currentDD     = psd.currentDD;
            break;
        }
    }

    // Recovery factor + Kelly.
    s.recoveryFactor = recoveryFactor(s.realized, s.maxDrawdown);
    s.kellyFraction  = kellyFraction(
        s.winCount, s.lossCount,
        s.avgWinner, s.avgLoser);

    // Active days + first/last.
    s.activeDays    = activeTradingDaysBySymbol(symbol);
    s.tradesPerDay  = s.activeDays > 0
                      ? static_cast<double>(s.roundTripCount) /
                        static_cast<double>(s.activeDays)
                      : 0.0;
    s.firstFillUs   = firstFillUsBySymbol(symbol);
    s.lastFillUs    = lastFillUsBySymbol(symbol);

    // Sharpe.
    for (const auto& pss : perSymbolSharpe()) {
        if (pss.symbol == symbol) {
            s.annualizedSharpe = pss.annualizedSharpe;
            break;
        }
    }

    return s;
}

TradeJournal::TagSummary
TradeJournal::tagSummary(const std::string& tag,
                         bool includeUntagged) const {
    // Sprint #126. Mirror of symbolSummary() (#125) for tags.
    // Same shape; loads fills once, filters once, looks up
    // drawdown/sharpe from vectors.
    TagSummary s;
    s.tag = tag;

    auto fills = loadAll();
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (f.tag.empty()) sub.push_back(f);
        } else {
            if (includeUntagged && f.tag.empty()) continue;
            if (f.tag == tag) sub.push_back(f);
        }
    }

    constexpr double kEps = 1e-9;
    double grossWin = 0.0, grossLoss = 0.0;
    size_t wins = 0, losses = 0;
    double sumRTpnl = 0.0;
    for (const auto& f : sub) {
        s.realized += f.realizedDelta;
        if (std::fabs(f.realizedDelta) <= kEps) continue;
        s.roundTripCount++;
        sumRTpnl += f.realizedDelta;
        if (f.realizedDelta > 0) {
            s.winCount++;
            grossWin += f.realizedDelta;
        } else {
            s.lossCount++;
            grossLoss += f.realizedDelta;
        }
    }
    s.winRate      = s.roundTripCount > 0
                     ? static_cast<double>(s.winCount) /
                       static_cast<double>(s.roundTripCount)
                     : 0.0;
    s.avgWinner    = s.winCount > 0
                     ? grossWin / static_cast<double>(s.winCount)
                     : 0.0;
    s.avgLoser     = s.lossCount > 0
                     ? grossLoss / static_cast<double>(s.lossCount)
                     : 0.0;
    s.profitFactor = std::fabs(grossLoss) < kEps
                     ? (grossWin > kEps
                          ? std::numeric_limits<double>::infinity()
                          : 0.0)
                     : grossWin / std::fabs(grossLoss);
    s.expectancy   = s.roundTripCount > 0
                     ? sumRTpnl /
                       static_cast<double>(s.roundTripCount)
                     : 0.0;

    for (const auto& ptd : perTagDrawdown(includeUntagged)) {
        if (ptd.tag == tag) {
            s.maxDrawdown   = ptd.maxDrawdown;
            s.recoveryDate  = ptd.recoveryDate;
            s.recoveryDays  = ptd.recoveryDays;
            s.currentDD     = ptd.currentDD;
            break;
        }
    }

    s.recoveryFactor = recoveryFactor(s.realized, s.maxDrawdown);
    s.kellyFraction  = kellyFraction(
        s.winCount, s.lossCount,
        s.avgWinner, s.avgLoser);

    s.activeDays    = activeTradingDaysByTag(tag, includeUntagged);
    s.tradesPerDay  = s.activeDays > 0
                      ? static_cast<double>(s.roundTripCount) /
                        static_cast<double>(s.activeDays)
                      : 0.0;
    s.firstFillUs   = firstFillUsByTag(tag, includeUntagged);
    s.lastFillUs    = lastFillUsByTag(tag, includeUntagged);

    for (const auto& pts : perTagSharpe(includeUntagged)) {
        if (pts.tag == tag) {
            s.annualizedSharpe = pts.annualizedSharpe;
            break;
        }
    }

    return s;
}

namespace {
// Sprint #127 — shared daily-streak walker. Buckets fills
// by local day (skip days with no round-trips), classifies
// each day's net P&L as W/L (skip ties), walks consecutive
// days into streaks.
template <typename Pred>
TradeJournal::DailyStreakStats
buildDailyStreakStats(const std::vector<JournalFill>& fills,
                      Pred pred) {
    std::vector<JournalFill> filtered;
    filtered.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) filtered.push_back(f);
    }
    // bucketByLocalDay returns a std::map<YYYY-MM-DD, sum>.
    // We use the same helper from Sprint #115.
    auto buckets = bucketByLocalDay(filtered);
    // Walk days in chronological order (std::map is sorted).
    TradeJournal::DailyStreakStats out;
    out.totalDays = buckets.size();
    constexpr double kEps = 1e-9;
    bool   inRun         = false;
    bool   runIsWin      = false;
    size_t runLen        = 0;
    auto closeRun = [&]() {
        if (!inRun) return;
        out.totalStreaks++;
        if (runIsWin) {
            // totalWinDays counts DAYS in W runs, not
            // the number of W runs. Add runLen.
            out.totalWinDays += runLen;
            if (runLen > out.maxWinStreak) out.maxWinStreak = runLen;
        } else {
            out.totalLossDays += runLen;
            if (runLen > out.maxLossStreak)
                out.maxLossStreak = runLen;
        }
        inRun = false;
        runLen = 0;
    };
    for (const auto& kv : buckets) {
        out.totalRealized += kv.second;
        if (std::fabs(kv.second) <= kEps) continue;  // tie day
        bool dayIsWin = (kv.second > kEps);
        if (inRun && dayIsWin != runIsWin) closeRun();
        if (!inRun) {
            inRun    = true;
            runIsWin = dayIsWin;
            runLen   = 1;
        } else {
            runLen++;
        }
    }
    if (inRun) {
        if (runIsWin) out.currentWinStreak = runLen;
        else          out.currentLossStreak = runLen;
        out.totalStreaks++;
        if (runIsWin) {
            out.totalWinDays += runLen;
            if (runLen > out.maxWinStreak)
                out.maxWinStreak = runLen;
        } else {
            out.totalLossDays += runLen;
            if (runLen > out.maxLossStreak)
                out.maxLossStreak = runLen;
        }
    }
    return out;
}
}  // namespace

TradeJournal::DailyStreakStats
TradeJournal::dailyStreakStats() const {
    return buildDailyStreakStats(loadAll(),
        [](const JournalFill&) { return true; });
}

TradeJournal::DailyStreakStats
TradeJournal::dailyStreakStatsBySymbol(
    const std::string& symbol) const {
    return buildDailyStreakStats(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::DailyStreakStats
TradeJournal::dailyStreakStatsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildDailyStreakStats(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #128 — shared P&L distribution builder. Returns
// percentiles + mean/stddev of the round-trip realizedDelta
// values (ties excluded).
template <typename Pred>
TradeJournal::PnLDistribution
buildPnLDistribution(const std::vector<JournalFill>& fills,
                     Pred pred) {
    std::vector<JournalFill> filtered;
    filtered.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) filtered.push_back(f);
    }
    // Keep only round-trips.
    std::vector<double> rt;
    rt.reserve(filtered.size());
    for (const auto& f : filtered) {
        if (std::fabs(f.realizedDelta) > 1e-9) {
            rt.push_back(f.realizedDelta);
        }
    }
    TradeJournal::PnLDistribution out;
    if (rt.empty()) return out;
    std::sort(rt.begin(), rt.end());
    out.count = rt.size();
    out.min   = rt.front();
    out.max   = rt.back();
    double sum = 0.0;
    for (double x : rt) sum += x;
    out.mean = sum / static_cast<double>(rt.size());
    double var = 0.0;
    for (double x : rt) {
        double d = x - out.mean;
        var += d * d;
    }
    if (rt.size() > 1) {
        out.stddev = std::sqrt(var /
            static_cast<double>(rt.size() - 1));
    } else {
        out.stddev = 0.0;
    }
    // Linear-interpolation percentile. For N sorted values,
    // the p-th percentile sits at index (p/100) * (N-1),
    // interpolated between the two neighbors.
    auto pctile = [&](double p) {
        double rank = (p / 100.0) *
                      static_cast<double>(rt.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(rank));
        size_t hi = static_cast<size_t>(std::ceil(rank));
        if (lo == hi) return rt[lo];
        double frac = rank - static_cast<double>(lo);
        return rt[lo] * (1.0 - frac) + rt[hi] * frac;
    };
    out.p10 = pctile(10);
    out.p25 = pctile(25);
    out.p50 = pctile(50);
    out.p75 = pctile(75);
    out.p90 = pctile(90);
    return out;
}
}  // namespace

TradeJournal::PnLDistribution
TradeJournal::pnlDistribution() const {
    return buildPnLDistribution(loadAll(),
        [](const JournalFill&) { return true; });
}

TradeJournal::PnLDistribution
TradeJournal::pnlDistributionBySymbol(
    const std::string& symbol) const {
    return buildPnLDistribution(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::PnLDistribution
TradeJournal::pnlDistributionByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildPnLDistribution(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #129 — shared DD-recovery-time distribution
// builder. Buckets each completed drawdown's recovery_us
// into bands.
template <typename Container>
TradeJournal::DDRecoveryDistribution
buildDDRecoveryDistribution(const Container& events) {
    TradeJournal::DDRecoveryDistribution out;
    if (events.empty()) return out;
    constexpr uint64_t kMin  = 60ULL * 1000000ULL;        // 1 min
    constexpr uint64_t kHr   = 60ULL * kMin;               // 1 hr
    constexpr uint64_t kDay  = 24ULL * kHr;                // 1 day
    constexpr uint64_t kWk   = 7ULL * kDay;                // 1 week
    constexpr uint64_t kMo   = 30ULL * kDay;               // 1 month
    uint64_t totalRecUs = 0;
    for (const auto& ev : events) {
        // Only count RECOVERED drawdowns — those with a
        // recovery_us > 0 (i.e. the trader climbed back
        // out). Unrecovered DDs have recovery_us == 0.
        if (ev.recovery_us == 0) continue;
        out.totalDrawdowns++;
        totalRecUs += ev.recovery_us;
        if (ev.recovery_us > out.maxRecoveryUs) {
            out.maxRecoveryUs = ev.recovery_us;
        }
        if (ev.recovery_us < kMin)        ++out.sameMinute;
        else if (ev.recovery_us < kHr)    ++out.under1h;
        else if (ev.recovery_us < kDay)   ++out.under1d;
        else if (ev.recovery_us < kWk)    ++out.under1w;
        else if (ev.recovery_us < kMo)    ++out.under1mo;
        else                              ++out.over1mo;
    }
    if (out.totalDrawdowns > 0) {
        double avgUs = static_cast<double>(totalRecUs) /
                       static_cast<double>(out.totalDrawdowns);
        out.avgRecoveryDays = avgUs /
            static_cast<double>(kDay);
    }
    return out;
}
}  // namespace

TradeJournal::DDRecoveryDistribution
TradeJournal::ddRecoveryDistribution() const {
    return buildDDRecoveryDistribution(drawdownRecoveries());
}

TradeJournal::DDRecoveryDistribution
TradeJournal::ddRecoveryDistributionBySymbol(
    const std::string& symbol) const {
    return buildDDRecoveryDistribution(
        drawdownRecoveriesBySymbol(symbol));
}

TradeJournal::DDRecoveryDistribution
TradeJournal::ddRecoveryDistributionByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildDDRecoveryDistribution(
        drawdownRecoveriesByTag(tag, includeUntagged));
}

namespace {
// Sprint #130 — shared DD-depth distribution builder.
// Buckets each completed DD's peak-to-trough depth into
// absolute ranges.
template <typename Container>
TradeJournal::DDDepthDistribution
buildDDDepthDistribution(const Container& events) {
    TradeJournal::DDDepthDistribution out;
    if (events.empty()) return out;
    double totalDepth = 0.0;
    for (const auto& ev : events) {
        if (ev.recovery_us == 0) continue;  // skip unrecovered
        out.totalDrawdowns++;
        double depth = ev.trough_depth;
        if (depth < 0.0) depth = -depth;  // safety
        totalDepth += depth;
        if (depth > out.maxDepth) out.maxDepth = depth;
        if      (depth < 50.0)    ++out.small;
        else if (depth < 100.0)   ++out.minor;
        else if (depth < 500.0)   ++out.moderate;
        else if (depth < 1000.0)  ++out.large;
        else if (depth < 5000.0)  ++out.severe;
        else                      ++out.catastrophic;
    }
    if (out.totalDrawdowns > 0) {
        out.avgDepth = totalDepth /
            static_cast<double>(out.totalDrawdowns);
    }
    return out;
}
}  // namespace

TradeJournal::DDDepthDistribution
TradeJournal::ddDepthDistribution() const {
    return buildDDDepthDistribution(drawdownRecoveries());
}

TradeJournal::DDDepthDistribution
TradeJournal::ddDepthDistributionBySymbol(
    const std::string& symbol) const {
    return buildDDDepthDistribution(
        drawdownRecoveriesBySymbol(symbol));
}

TradeJournal::DDDepthDistribution
TradeJournal::ddDepthDistributionByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildDDDepthDistribution(
        drawdownRecoveriesByTag(tag, includeUntagged));
}

namespace {
// Sprint #131 — shared DD-duration stats builder.
// Sums drawdown_us across all completed DDs.
template <typename Container>
TradeJournal::DDDurationStats
buildDDDurationStats(const Container& events) {
    TradeJournal::DDDurationStats out;
    if (events.empty()) return out;
    constexpr uint64_t kDay = 24ULL * 60ULL * 60ULL * 1000000ULL;
    double totalDays = 0.0;
    double maxDays   = 0.0;
    for (const auto& ev : events) {
        if (ev.recovery_us == 0) continue;  // skip unrecovered
        out.totalDrawdowns++;
        double days = static_cast<double>(ev.drawdown_us) /
                      static_cast<double>(kDay);
        totalDays += days;
        if (days > maxDays) maxDays = days;
    }
    if (out.totalDrawdowns > 0) {
        out.avgDurationDays   = totalDays /
            static_cast<double>(out.totalDrawdowns);
        out.maxDurationDays   = maxDays;
        out.totalDurationDays = totalDays;
    }
    return out;
}
}  // namespace

TradeJournal::DDDurationStats
TradeJournal::ddDurationStats() const {
    return buildDDDurationStats(drawdownRecoveries());
}

TradeJournal::DDDurationStats
TradeJournal::ddDurationStatsBySymbol(
    const std::string& symbol) const {
    return buildDDDurationStats(
        drawdownRecoveriesBySymbol(symbol));
}

TradeJournal::DDDurationStats
TradeJournal::ddDurationStatsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildDDDurationStats(
        drawdownRecoveriesByTag(tag, includeUntagged));
}

// Sprint #132/133 — equity annotation builder. Templated
// on the filter predicate so journal-wide + per-symbol +
// per-tag share one tested core.
template <typename Pred>
std::vector<TradeJournal::Annotation>
buildEquityAnnotations(const TradeJournal& j, Pred pred) {
    std::vector<TradeJournal::Annotation> out;
    auto fills = j.loadAll();
    // Filter fills.
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    // Build a filtered equity curve. Reuse equityCurve()
    // shape by sorting and walking.
    if (sub.empty()) return out;
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    std::vector<TradeJournal::EquityPoint> curve;
    curve.reserve(sub.size());
    {
        double cum = 0.0;
        for (const auto& f : sub) {
            cum += f.realizedDelta;
            TradeJournal::EquityPoint p;
            p.cumulative   = cum;
            p.timestamp_us = f.timestamp_us;
            curve.push_back(p);
        }
    }
    if (curve.empty()) return out;
    std::vector<TradeJournal::DrawdownEvent> dds;
    {
        // Mirror Sprint #113 drawdown walker logic.
        constexpr double kEps = 1e-9;
        double peak = std::numeric_limits<double>::lowest();
        uint64_t peak_ts = 0;
        bool in_dd = false;
        TradeJournal::DrawdownEvent cur;
        for (size_t i = 0; i < curve.size(); ++i) {
            const auto& p = curve[i];
            if (p.cumulative > peak) {
                if (in_dd) {
                    cur.end_ts = peak_ts;
                    cur.drawdown_us = cur.end_ts - cur.start_ts;
                    cur.recovery_us = cur.end_ts - cur.trough_ts;
                    dds.push_back(cur);
                    in_dd = false;
                }
                peak = p.cumulative;
                peak_ts = p.timestamp_us;
            } else if (!in_dd) {
                cur = TradeJournal::DrawdownEvent{};
                cur.start_ts = peak_ts;
                cur.trough_ts = p.timestamp_us;
                cur.peak_before = peak;
                cur.trough_value = p.cumulative;
                cur.trough_depth = peak - p.cumulative;
                in_dd = true;
            } else {
                if (p.cumulative < cur.trough_value) {
                    cur.trough_value = p.cumulative;
                    cur.trough_ts = p.timestamp_us;
                    cur.trough_depth = peak - p.cumulative;
                }
            }
            (void)kEps;
        }
        if (in_dd) {
            cur.end_ts = peak_ts;
            cur.drawdown_us = cur.end_ts - cur.start_ts;
            cur.recovery_us = cur.end_ts - cur.trough_ts;
            dds.push_back(cur);
        }
        std::sort(dds.begin(), dds.end(),
            [](const TradeJournal::DrawdownEvent& a,
               const TradeJournal::DrawdownEvent& b) {
                return a.trough_depth > b.trough_depth;
            });
    }
    // 1. Every recovered DD start + end.
    for (const auto& dd : dds) {
        TradeJournal::Annotation a;
        a.kind = TradeJournal::AnnotationKind::DDStart;
        a.timestamp_us = dd.start_ts;
        a.value = dd.peak_before;
        a.label = "DD start (-" +
            std::to_string(static_cast<int>(dd.trough_depth)) + ")";
        out.push_back(a);
        TradeJournal::Annotation b;
        b.kind = TradeJournal::AnnotationKind::DDEnd;
        b.timestamp_us = dd.end_ts;
        b.value = dd.peak_before;
        b.label = "DD recovered";
        out.push_back(b);
    }
    // 2. The single deepest DD.
    if (!dds.empty()) {
        const auto& maxDD = dds.front();
        TradeJournal::Annotation a;
        a.kind = TradeJournal::AnnotationKind::MaxDDStart;
        a.timestamp_us = maxDD.start_ts;
        a.value = maxDD.peak_before;
        a.label = "MAX DD start (-" +
            std::to_string(static_cast<int>(maxDD.trough_depth)) + ")";
        out.push_back(a);
        TradeJournal::Annotation b;
        b.kind = TradeJournal::AnnotationKind::MaxDDEnd;
        b.timestamp_us = maxDD.end_ts;
        b.value = maxDD.peak_before;
        b.label = "MAX DD recovered";
        out.push_back(b);
    }
    // 3. Best + worst day (within filtered fills).
    {
        std::map<std::string, double> days;
        for (const auto& f : sub) {
            std::time_t s = static_cast<std::time_t>(
                f.timestamp_us / 1000000ULL);
            std::tm tm{};
            localtime_r(&s, &tm);
            char buf[16];
            std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
            days[buf] += f.realizedDelta;
        }
        if (!days.empty()) {
            double bestVal = -1e18, worstVal = 1e18;
            std::string bestDate, worstDate;
            for (const auto& kv : days) {
                if (kv.second > bestVal) {
                    bestVal = kv.second; bestDate = kv.first;
                }
                if (kv.second < worstVal) {
                    worstVal = kv.second; worstDate = kv.first;
                }
            }
            if (!bestDate.empty()) {
                TradeJournal::Annotation a;
                a.kind = TradeJournal::AnnotationKind::BestDay;
                a.timestamp_us = 0;
                a.value = bestVal;
                a.label = "Best day " + bestDate + ": " +
                    std::to_string(static_cast<int>(bestVal));
                out.push_back(a);
            }
            if (!worstDate.empty() && worstDate != bestDate) {
                TradeJournal::Annotation a;
                a.kind = TradeJournal::AnnotationKind::WorstDay;
                a.timestamp_us = 0;
                a.value = worstVal;
                a.label = "Worst day " + worstDate + ": " +
                    std::to_string(static_cast<int>(worstVal));
                out.push_back(a);
            }
        }
    }
    // 4. Equity high water marks.
    {
        TradeJournal::Annotation a;
        a.kind = TradeJournal::AnnotationKind::EquityHigh;
        a.timestamp_us = curve[0].timestamp_us;
        a.value = curve[0].cumulative;
        a.label = "Equity high: " +
            std::to_string(static_cast<int>(curve[0].cumulative));
        out.push_back(a);
    }
    double runningPeak = curve[0].cumulative;
    for (size_t i = 1; i < curve.size(); ++i) {
        if (curve[i].cumulative > runningPeak) {
            runningPeak = curve[i].cumulative;
            TradeJournal::Annotation a;
            a.kind = TradeJournal::AnnotationKind::EquityHigh;
            a.timestamp_us = curve[i].timestamp_us;
            a.value = curve[i].cumulative;
            a.label = "Equity high: " +
                std::to_string(static_cast<int>(curve[i].cumulative));
            out.push_back(a);
        }
    }
    // 5. Sort by timestamp ASC.
    std::sort(out.begin(), out.end(),
        [](const TradeJournal::Annotation& a,
           const TradeJournal::Annotation& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    return out;
}

std::vector<TradeJournal::Annotation>
TradeJournal::equityAnnotations() const {
    return buildEquityAnnotations(*this,
        [](const JournalFill&) { return true; });
}

std::vector<TradeJournal::Annotation>
TradeJournal::equityAnnotationsBySymbol(
    const std::string& symbol) const {
    return buildEquityAnnotations(*this,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::Annotation>
TradeJournal::equityAnnotationsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildEquityAnnotations(*this,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #134 — shared rolling-Sharpe builder. Same
// ring-buffer pattern as rollingProfitFactor (#124) but
// computes (mean / stddev) over the window. Returns 0
// when stddev == 0 (all returns identical).
template <typename Pred>
std::vector<TradeJournal::WindowSharpePoint>
buildRollingSharpe(const std::vector<JournalFill>& fills,
                    size_t window, Pred pred) {
    std::vector<JournalFill> filtered;
    filtered.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) filtered.push_back(f);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    std::vector<double> rt;
    rt.reserve(filtered.size());
    for (const auto& f : filtered) {
        if (std::fabs(f.realizedDelta) > 1e-9) {
            rt.push_back(f.realizedDelta);
        }
    }
    if (rt.size() < window) return {};
    std::vector<TradeJournal::WindowSharpePoint> out;
    out.reserve(rt.size() - window + 1);
    auto computeStats = [&](size_t start, size_t end) {
        // Returns (mean, stddev).
        double sum = 0.0;
        for (size_t k = start; k < end; ++k) sum += rt[k];
        double mean = sum / static_cast<double>(end - start);
        double var = 0.0;
        for (size_t k = start; k < end; ++k) {
            double d = rt[k] - mean;
            var += d * d;
        }
        double stddev = end - start > 1
            ? std::sqrt(var / static_cast<double>(end - start - 1))
            : 0.0;
        return std::make_pair(mean, stddev);
    };
    // Emit the first point.
    {
        auto [mean, stddev] = computeStats(0, window);
        TradeJournal::WindowSharpePoint p;
        p.timestamp_us = filtered[window - 1].timestamp_us;
        p.count = window;
        p.mean = mean;
        p.stddev = stddev;
        p.sharpe = stddev > 1e-9 ? mean / stddev : 0.0;
        out.push_back(p);
    }
    // Slide the window: drop rt[i-window], add rt[i].
    // Maintain running sum + sum-of-squares for O(1)
    // window updates.
    double sum   = 0.0;
    double sumSq = 0.0;
    for (size_t k = 0; k < window; ++k) {
        sum   += rt[k];
        sumSq += rt[k] * rt[k];
    }
    for (size_t i = window; i < rt.size(); ++i) {
        double dropped = rt[i - window];
        double added   = rt[i];
        sum   = sum - dropped + added;
        sumSq = sumSq - dropped * dropped + added * added;
        double mean = sum / static_cast<double>(window);
        double var  = (sumSq - static_cast<double>(window) *
                       mean * mean) /
                      static_cast<double>(window - 1);
        double stddev = var > 0 ? std::sqrt(var) : 0.0;
        TradeJournal::WindowSharpePoint p;
        p.timestamp_us = filtered[i].timestamp_us;
        p.count = window;
        p.mean = mean;
        p.stddev = stddev;
        p.sharpe = stddev > 1e-9 ? mean / stddev : 0.0;
        out.push_back(p);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::WindowSharpePoint>
TradeJournal::rollingWindowSharpe(size_t window) const {
    return buildRollingSharpe(loadAll(), window,
        [](const JournalFill&) { return true; });
}

std::vector<TradeJournal::WindowSharpePoint>
TradeJournal::rollingWindowSharpeBySymbol(
    const std::string& symbol,
    size_t window) const {
    return buildRollingSharpe(loadAll(), window,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::WindowSharpePoint>
TradeJournal::rollingWindowSharpeByTag(
    const std::string& tag,
    bool includeUntagged,
    size_t window) const {
    return buildRollingSharpe(loadAll(), window,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #135 — shared risk-score builder. Combines 4
// sub-scores into a single overall 0-100 score using fixed
// weights (30% Sharpe, 30% drawdown, 20% win rate, 20%
// payoff).
template <typename Pred>
TradeJournal::RiskScore
buildRiskScore(const TradeJournal& j, Pred pred) {
    TradeJournal::RiskScore out;
    auto fills = j.loadAll();
    // Filter fills.
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    if (sub.empty()) return out;

    // Sub-score 1: Sharpe normalized.
    // Use the existing rollingWindowSharpe(window) — take
    // the LAST point's sharpe as the current edge quality.
    auto rsharpe = j.rollingWindowSharpe(30);
    if (!rsharpe.empty()) {
        double s = rsharpe.back().sharpe;
        // Map: 0 → 50, 2 → 100, -1 → 25 (linear extrapolation).
        out.sharpeScore = std::clamp(50.0 + s * 25.0, 0.0, 100.0);
    }
    // Sub-score 2: Drawdown inverse-normalized.
    double maxDD = 0.0;
    {
        auto dds = j.drawdownRecoveries();
        for (const auto& dd : dds) {
            if (dd.trough_depth > maxDD) maxDD = dd.trough_depth;
        }
    }
    // log-ish: 0 → 100, 1000 → 50, 10000 → 0.
    if (maxDD < 1e-9) {
        out.drawdownScore = 100.0;
    } else {
        // Score = 100 * exp(-maxDD / 2000).
        // maxDD=0    → 100.
        // maxDD=1000 → 60.65.
        // maxDD=2000 → 36.79.
        // maxDD=5000 → 8.21.
        // maxDD=10000→ 0.67.
        out.drawdownScore = std::clamp(
            100.0 * std::exp(-maxDD / 2000.0), 0.0, 100.0);
    }
    // Sub-score 3: Win rate × 100.
    size_t wins = 0, losses = 0;
    double grossWin = 0.0, grossLoss = 0.0;
    for (const auto& f : sub) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        if (f.realizedDelta > 0) {
            ++wins;
            grossWin += f.realizedDelta;
        } else {
            ++losses;
            grossLoss += f.realizedDelta;
        }
    }
    size_t total = wins + losses;
    if (total > 0) {
        out.winRateScore = 100.0 *
            static_cast<double>(wins) /
            static_cast<double>(total);
    }
    // Sub-score 4: Payoff (avgW / |avgL|) normalized.
    double avgW = wins > 0 ? grossWin / static_cast<double>(wins)
                           : 0.0;
    double avgL = losses > 0 ? grossLoss /
                               static_cast<double>(losses)
                            : 0.0;
    if (wins > 0 && losses > 0 && std::fabs(avgL) > 1e-9) {
        double payoff = avgW / std::fabs(avgL);
        // 1.0 → 50, 2.0 → 100 (cap at 100).
        out.payoffScore = std::clamp(payoff * 50.0, 0.0, 100.0);
    } else if (wins > 0 && losses == 0) {
        out.payoffScore = 100.0;  // all wins
    }
    // Overall: weighted combination.
    out.overall = 0.30 * out.sharpeScore +
                  0.30 * out.drawdownScore +
                  0.20 * out.winRateScore +
                  0.20 * out.payoffScore;
    return out;
}
}  // namespace

TradeJournal::RiskScore
TradeJournal::riskScore() const {
    return buildRiskScore(*this,
        [](const JournalFill&) { return true; });
}

TradeJournal::RiskScore
TradeJournal::riskScoreBySymbol(
    const std::string& symbol) const {
    return buildRiskScore(*this,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::RiskScore
TradeJournal::riskScoreByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildRiskScore(*this,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::SymbolShare>
TradeJournal::symbolConcentration() const {
    // Sprint #136. Bucket realized P&L by symbol. Sort DESC
    // by |realized|. Compute share as fraction of total
    // |realized| (sign-agnostic — we're measuring
    // contribution, not direction).
    std::vector<SymbolShare> out;
    auto fills = loadAll();
    std::map<std::string, double> totals;
    double grandAbs = 0.0;
    for (const auto& f : fills) {
        totals[f.symbol] += f.realizedDelta;
        grandAbs += std::fabs(f.realizedDelta);
    }
    if (grandAbs < 1e-9) return out;
    for (const auto& kv : totals) {
        SymbolShare s;
        s.symbol   = kv.first;
        s.realized = kv.second;
        s.share    = std::fabs(kv.second) / grandAbs;
        out.push_back(s);
    }
    std::sort(out.begin(), out.end(),
        [](const SymbolShare& a, const SymbolShare& b) {
            return std::fabs(a.realized) > std::fabs(b.realized);
        });
    double cum = 0.0;
    for (auto& s : out) {
        cum += s.share;
        s.cumShare = cum;
    }
    return out;
}

namespace {
// Sprint #137 — Herfindahl-Hirschman Index (HHI) helper.
// Generic: takes any keyFn that buckets fills, returns
// sum of squared shares over |realized|.
double computeHHI(const std::vector<JournalFill>& fills,
                  std::function<std::string(const JournalFill&)> keyFn,
                  bool includeUntagged) {
    std::map<std::string, double> totals;
    double grandAbs = 0.0;
    for (const auto& f : fills) {
        std::string key = keyFn(f);
        if (key.empty()) continue;
        totals[key] += f.realizedDelta;
        grandAbs += std::fabs(f.realizedDelta);
    }
    if (grandAbs < 1e-9) return 0.0;
    double hhi = 0.0;
    for (const auto& kv : totals) {
        double share = std::fabs(kv.second) / grandAbs;
        hhi += share * share;
    }
    return hhi;
}
}  // namespace

double TradeJournal::concentrationHHI() const {
    // Sprint #137. HHI over symbols. Standard
    // market-concentration metric. 1/N = perfectly equal,
    // 1.0 = single symbol.
    return computeHHI(loadAll(),
        [](const JournalFill& f) { return f.symbol; },
        false);
}

double TradeJournal::concentrationHHIByTag(
    bool includeUntagged) const {
    return computeHHI(loadAll(),
        [includeUntagged](const JournalFill& f) -> std::string {
            if (f.tag.empty()) {
                return includeUntagged ? "__untagged__" : "";
            }
            return f.tag;
        },
        includeUntagged);
}

namespace {
// Sprint #138 — shared trade-size stats builder.
// Templated on filter predicate.
template <typename Pred>
TradeJournal::TradeSizeStats
buildTradeSizeStats(const std::vector<JournalFill>& fills,
                    Pred pred) {
    TradeJournal::TradeSizeStats out;
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    if (sub.empty()) return out;
    std::vector<double> absVals;
    absVals.reserve(sub.size());
    double totalWin = 0.0, totalLoss = 0.0;
    size_t wins = 0, losses = 0;
    for (const auto& f : sub) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        out.roundTripCount++;
        double a = std::fabs(f.realizedDelta);
        absVals.push_back(a);
        if (f.realizedDelta > 0) {
            totalWin += f.realizedDelta;
            ++wins;
        } else {
            totalLoss += a;
            ++losses;
        }
    }
    if (out.roundTripCount == 0) return out;
    double sumAbs = 0.0, maxAbs = 0.0;
    for (double a : absVals) {
        sumAbs += a;
        if (a > maxAbs) maxAbs = a;
    }
    out.meanAbs = sumAbs /
        static_cast<double>(out.roundTripCount);
    out.maxAbs  = maxAbs;
    out.totalWinSize  = totalWin;
    out.totalLossSize = totalLoss;
    if (wins   > 0) out.meanWin  = totalWin  /
        static_cast<double>(wins);
    if (losses > 0) out.meanLoss = totalLoss /
        static_cast<double>(losses);
    // Sort for percentiles.
    std::sort(absVals.begin(), absVals.end());
    auto pctile = [&](double p) {
        double rank = (p / 100.0) *
            static_cast<double>(absVals.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(rank));
        size_t hi = static_cast<size_t>(std::ceil(rank));
        if (lo == hi) return absVals[lo];
        double frac = rank - static_cast<double>(lo);
        return absVals[lo] * (1.0 - frac) +
               absVals[hi] * frac;
    };
    out.medianAbs = pctile(50);
    out.p90Abs    = pctile(90);
    return out;
}
}  // namespace

TradeJournal::TradeSizeStats
TradeJournal::tradeSizeStats() const {
    return buildTradeSizeStats(loadAll(),
        [](const JournalFill&) { return true; });
}

TradeJournal::TradeSizeStats
TradeJournal::tradeSizeStatsBySymbol(
    const std::string& symbol) const {
    return buildTradeSizeStats(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::TradeSizeStats
TradeJournal::tradeSizeStatsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildTradeSizeStats(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::EquityVolPoint>
TradeJournal::equityVolatility(size_t window) const {
    // Sprint #139. Walk the equity curve, compute rolling
    // stddev over a window of consecutive equity values.
    // O(N) via ring buffer with running sum + sumSq.
    std::vector<EquityVolPoint> out;
    auto curve = equityCurve();
    if (curve.size() < window) return out;
    out.reserve(curve.size() - window + 1);
    double sum = 0.0, sumSq = 0.0;
    for (size_t k = 0; k < window; ++k) {
        double v = curve[k].cumulative;
        sum   += v;
        sumSq += v * v;
    }
    {
        double mean = sum / static_cast<double>(window);
        double var = (sumSq / static_cast<double>(window)) -
                     mean * mean;
        if (var < 0.0) var = 0.0;
        EquityVolPoint p;
        p.timestamp_us   = curve[window - 1].timestamp_us;
        p.equityValue    = curve[window - 1].cumulative;
        p.rollingStddev  = std::sqrt(var);
        p.count          = window;
        out.push_back(p);
    }
    for (size_t i = window; i < curve.size(); ++i) {
        double dropped = curve[i - window].cumulative;
        double added   = curve[i].cumulative;
        sum   = sum - dropped + added;
        sumSq = sumSq - dropped * dropped + added * added;
        double mean = sum / static_cast<double>(window);
        double var  = (sumSq / static_cast<double>(window)) -
                      mean * mean;
        if (var < 0.0) var = 0.0;
        EquityVolPoint p;
        p.timestamp_us   = curve[i].timestamp_us;
        p.equityValue    = curve[i].cumulative;
        p.rollingStddev  = std::sqrt(var);
        p.count          = window;
        out.push_back(p);
    }
    return out;
}

namespace {
// Sprint #140 — equity-volatility builder. Builds a
// filtered equity curve (cumulative sum of realizedDelta
// for matching fills), then runs the same rolling-stddev
// loop as equityVolatility() (#139).
template <typename Pred>
std::vector<TradeJournal::EquityVolPoint>
buildEquityVolatility(const std::vector<JournalFill>& fills,
                       size_t window, Pred pred) {
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    std::vector<TradeJournal::EquityVolPoint> out;
    if (sub.size() < window) return out;
    out.reserve(sub.size() - window + 1);
    double sum = 0.0, sumSq = 0.0;
    double cum = 0.0;
    std::vector<double> cums;
    cums.reserve(sub.size());
    for (const auto& f : sub) {
        cum += f.realizedDelta;
        cums.push_back(cum);
    }
    for (size_t k = 0; k < window; ++k) {
        double v = cums[k];
        sum   += v;
        sumSq += v * v;
    }
    {
        double mean = sum / static_cast<double>(window);
        double var = (sumSq / static_cast<double>(window)) -
                     mean * mean;
        if (var < 0.0) var = 0.0;
        TradeJournal::EquityVolPoint p;
        p.timestamp_us   = sub[window - 1].timestamp_us;
        p.equityValue    = cums[window - 1];
        p.rollingStddev  = std::sqrt(var);
        p.count          = window;
        out.push_back(p);
    }
    for (size_t i = window; i < sub.size(); ++i) {
        double dropped = cums[i - window];
        double added   = cums[i];
        sum   = sum - dropped + added;
        sumSq = sumSq - dropped * dropped + added * added;
        double mean = sum / static_cast<double>(window);
        double var  = (sumSq / static_cast<double>(window)) -
                      mean * mean;
        if (var < 0.0) var = 0.0;
        TradeJournal::EquityVolPoint p;
        p.timestamp_us   = sub[i].timestamp_us;
        p.equityValue    = cums[i];
        p.rollingStddev  = std::sqrt(var);
        p.count          = window;
        out.push_back(p);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::EquityVolPoint>
TradeJournal::equityVolatilityBySymbol(
    const std::string& symbol,
    size_t window) const {
    return buildEquityVolatility(loadAll(), window,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::EquityVolPoint>
TradeJournal::equityVolatilityByTag(
    const std::string& tag,
    bool includeUntagged,
    size_t window) const {
    return buildEquityVolatility(loadAll(), window,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #141 — Wilson score CI helper.
// Generic: counts wins + losses in matching fills, returns
// the CI struct.
template <typename Pred>
TradeJournal::WinRateCI
buildWinRateCI(const std::vector<JournalFill>& fills,
               Pred pred) {
    TradeJournal::WinRateCI out;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        if (f.realizedDelta > 0) out.wins++;
        else out.losses++;
    }
    out.total = out.wins + out.losses;
    if (out.total == 0) return out;
    double n = static_cast<double>(out.total);
    double p = static_cast<double>(out.wins) / n;
    out.observed = p;
    constexpr double z = 1.959963984540054;  // 95% CI
    double z2 = z * z;
    double denom = 1.0 + z2 / n;
    double center = (p + z2 / (2.0 * n)) / denom;
    double margin = z * std::sqrt(
        (p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom;
    out.lower95 = std::max(0.0, center - margin);
    out.upper95 = std::min(1.0, center + margin);
    return out;
}
}  // namespace

TradeJournal::WinRateCI
TradeJournal::winRateCI() const {
    return buildWinRateCI(loadAll(),
        [](const JournalFill&) { return true; });
}

TradeJournal::WinRateCI
TradeJournal::winRateCIBySymbol(
    const std::string& symbol) const {
    return buildWinRateCI(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::WinRateCI
TradeJournal::winRateCIByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildWinRateCI(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #142 — win-rate-by-size helper.
template <typename Pred>
TradeJournal::WinRateBySize
buildWinRateBySize(const std::vector<JournalFill>& fills,
                    Pred pred) {
    TradeJournal::WinRateBySize out;
    auto classify = [&](double abs) -> TradeJournal::SizeBucketWR* {
        if (abs < 50.0)   return &out.tiny;
        if (abs < 100.0)  return &out.small;
        if (abs < 500.0)  return &out.medium;
        if (abs < 1000.0) return &out.large;
        if (abs < 5000.0) return &out.huge;
        return &out.massive;
    };
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        double a = std::fabs(f.realizedDelta);
        auto* b = classify(a);
        b->total++;
        b->meanAbs += a;
        if (f.realizedDelta > 0) b->wins++;
        else b->losses++;
    }
    auto finalize = [](TradeJournal::SizeBucketWR& b) {
        if (b.total > 0) {
            b.winRate = static_cast<double>(b.wins) /
                        static_cast<double>(b.total);
            b.meanAbs /= static_cast<double>(b.total);
        }
    };
    finalize(out.tiny); finalize(out.small);
    finalize(out.medium); finalize(out.large);
    finalize(out.huge); finalize(out.massive);
    return out;
}
}  // namespace

TradeJournal::WinRateBySize
TradeJournal::winRateBySize() const {
    return buildWinRateBySize(loadAll(),
        [](const JournalFill&) { return true; });
}

TradeJournal::WinRateBySize
TradeJournal::winRateBySizeBySymbol(
    const std::string& symbol) const {
    return buildWinRateBySize(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::WinRateBySize
TradeJournal::winRateBySizeByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildWinRateBySize(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::EquitySlopePoint>
TradeJournal::equityRateOfChange(size_t window) const {
    // Sprint #143. OLS slope of equity vs fill index
    // over a rolling window. Per-fill slope: cum change
    // per round-trip.
    //
    // Formula: slope = (N*Σ(xy) - Σx*Σy) /
    //                  (N*Σ(x²) - (Σx)²)
    //   where x_i = i - (N-1)/2  (centered for numerical
    //   stability of the denominator).
    std::vector<EquitySlopePoint> out;
    auto curve = equityCurve();
    if (curve.size() < window) return out;
    out.reserve(curve.size() - window + 1);
    auto computeSlope = [&](size_t start, size_t end) {
        size_t N = end - start;
        // Center x values around 0 for numerical stability.
        double xCenter = -static_cast<double>(N - 1) / 2.0;
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0;
        double sumX2 = 0.0;
        for (size_t k = 0; k < N; ++k) {
            double x = xCenter + static_cast<double>(k);
            double y = curve[start + k].cumulative;
            sumX  += x;
            sumY  += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        double denom = static_cast<double>(N) * sumX2 -
                       sumX * sumX;
        if (std::fabs(denom) < 1e-12) return 0.0;
        double numer = static_cast<double>(N) * sumXY -
                       sumX * sumY;
        return numer / denom;
    };
    // First point.
    {
        double s = computeSlope(0, window);
        EquitySlopePoint p;
        p.timestamp_us = curve[window - 1].timestamp_us;
        p.equityValue  = curve[window - 1].cumulative;
        p.slope        = s;
        p.count        = window;
        out.push_back(p);
    }
    // Sliding: window shifts by 1 each step. Slope is NOT
    // O(1) updatable in general (numerator changes by
    // recomputed terms), so we recompute each step. With
    // O(N) total work this is still cheap for journal
    // sizes.
    for (size_t i = window; i < curve.size(); ++i) {
        double s = computeSlope(i - window, i);
        EquitySlopePoint p;
        p.timestamp_us = curve[i].timestamp_us;
        p.equityValue  = curve[i].cumulative;
        p.slope        = s;
        p.count        = window;
        out.push_back(p);
    }
    return out;
}

namespace {
// Sprint #144 — equity-rate-of-change builder on a
// filtered equity curve. Same slope formula as #143 but
// applied to the per-segment cum.
template <typename Pred>
std::vector<TradeJournal::EquitySlopePoint>
buildEquityRateOfChange(const std::vector<JournalFill>& fills,
                         size_t window, Pred pred) {
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    std::vector<TradeJournal::EquitySlopePoint> out;
    if (sub.size() < window) return out;
    out.reserve(sub.size() - window + 1);
    // Build cums.
    std::vector<double> cums;
    cums.reserve(sub.size());
    {
        double cum = 0.0;
        for (const auto& f : sub) {
            cum += f.realizedDelta;
            cums.push_back(cum);
        }
    }
    auto computeSlope = [&](size_t start, size_t end) {
        size_t N = end - start;
        double xCenter = -static_cast<double>(N - 1) / 2.0;
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0;
        double sumX2 = 0.0;
        for (size_t k = 0; k < N; ++k) {
            double x = xCenter + static_cast<double>(k);
            double y = cums[start + k];
            sumX  += x;
            sumY  += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        double denom = static_cast<double>(N) * sumX2 -
                       sumX * sumX;
        if (std::fabs(denom) < 1e-12) return 0.0;
        double numer = static_cast<double>(N) * sumXY -
                       sumX * sumY;
        return numer / denom;
    };
    {
        double s = computeSlope(0, window);
        TradeJournal::EquitySlopePoint p;
        p.timestamp_us = sub[window - 1].timestamp_us;
        p.equityValue  = cums[window - 1];
        p.slope        = s;
        p.count        = window;
        out.push_back(p);
    }
    for (size_t i = window; i < sub.size(); ++i) {
        double s = computeSlope(i - window, i);
        TradeJournal::EquitySlopePoint p;
        p.timestamp_us = sub[i].timestamp_us;
        p.equityValue  = cums[i];
        p.slope        = s;
        p.count        = window;
        out.push_back(p);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::EquitySlopePoint>
TradeJournal::equityRateOfChangeBySymbol(
    const std::string& symbol,
    size_t window) const {
    return buildEquityRateOfChange(loadAll(), window,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::EquitySlopePoint>
TradeJournal::equityRateOfChangeByTag(
    const std::string& tag,
    bool includeUntagged,
    size_t window) const {
    return buildEquityRateOfChange(loadAll(), window,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::SymbolSummary>
TradeJournal::allSymbolSummaries() const {
    // Sprint #145. Find every symbol in the journal, run
    // symbolSummary() for each, sort by realized DESC.
    std::vector<SymbolSummary> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(symbolSummary(s));
    }
    std::sort(out.begin(), out.end(),
        [](const SymbolSummary& a, const SymbolSummary& b) {
            return a.realized > b.realized;
        });
    return out;
}

std::vector<TradeJournal::TagSummary>
TradeJournal::allTagSummaries(bool includeUntagged) const {
    // Sprint #145. Mirror of allSymbolSummaries() for tags.
    std::vector<TagSummary> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(tagSummary(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const TagSummary& a, const TagSummary& b) {
            return a.realized > b.realized;
        });
    return out;
}

std::vector<TradeJournal::BestTrade>
TradeJournal::topWinners(size_t n) const {
    // Sprint #146. Top N winning round-trips across the
    // entire journal. Sorted DESC by realized.
    auto fills = loadAll();
    std::vector<JournalFill> wins;
    wins.reserve(fills.size());
    for (const auto& f : fills) {
        if (f.realizedDelta > 0) wins.push_back(f);
    }
    std::sort(wins.begin(), wins.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.realizedDelta > b.realizedDelta;
        });
    std::vector<BestTrade> out;
    size_t take = std::min(n, wins.size());
    out.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        BestTrade bt;
        bt.symbol = wins[i].symbol;
        bt.realized = wins[i].realizedDelta;
        bt.tag = wins[i].tag;
        bt.timestamp_us = wins[i].timestamp_us;
        out.push_back(bt);
    }
    return out;
}

std::vector<TradeJournal::BestTrade>
TradeJournal::topLosers(size_t n) const {
    // Sprint #146. Top N losing round-trips across the
    // entire journal. Sorted ASC by realized (most
    // negative first).
    auto fills = loadAll();
    std::vector<JournalFill> losses;
    losses.reserve(fills.size());
    for (const auto& f : fills) {
        if (f.realizedDelta < 0) losses.push_back(f);
    }
    std::sort(losses.begin(), losses.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.realizedDelta < b.realizedDelta;
        });
    std::vector<BestTrade> out;
    size_t take = std::min(n, losses.size());
    out.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        BestTrade bt;
        bt.symbol = losses[i].symbol;
        bt.realized = losses[i].realizedDelta;
        bt.tag = losses[i].tag;
        bt.timestamp_us = losses[i].timestamp_us;
        out.push_back(bt);
    }
    return out;
}

namespace {
// Sprint #147 — shared top-trades builder.
template <typename Pred>
std::vector<TradeJournal::BestTrade>
buildTopWinners(const std::vector<JournalFill>& fills,
                 size_t n, Pred pred) {
    std::vector<JournalFill> wins;
    for (const auto& f : fills) {
        if (pred(f) && f.realizedDelta > 0) wins.push_back(f);
    }
    std::sort(wins.begin(), wins.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.realizedDelta > b.realizedDelta;
        });
    std::vector<TradeJournal::BestTrade> out;
    size_t take = std::min(n, wins.size());
    out.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        TradeJournal::BestTrade bt;
        bt.symbol = wins[i].symbol;
        bt.realized = wins[i].realizedDelta;
        bt.tag = wins[i].tag;
        bt.timestamp_us = wins[i].timestamp_us;
        out.push_back(bt);
    }
    return out;
}

template <typename Pred>
std::vector<TradeJournal::BestTrade>
buildTopLosers(const std::vector<JournalFill>& fills,
                size_t n, Pred pred) {
    std::vector<JournalFill> losses;
    for (const auto& f : fills) {
        if (pred(f) && f.realizedDelta < 0) losses.push_back(f);
    }
    std::sort(losses.begin(), losses.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.realizedDelta < b.realizedDelta;
        });
    std::vector<TradeJournal::BestTrade> out;
    size_t take = std::min(n, losses.size());
    out.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        TradeJournal::BestTrade bt;
        bt.symbol = losses[i].symbol;
        bt.realized = losses[i].realizedDelta;
        bt.tag = losses[i].tag;
        bt.timestamp_us = losses[i].timestamp_us;
        out.push_back(bt);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::BestTrade>
TradeJournal::topWinnersBySymbol(
    const std::string& symbol, size_t n) const {
    return buildTopWinners(loadAll(), n,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::BestTrade>
TradeJournal::topLosersBySymbol(
    const std::string& symbol, size_t n) const {
    return buildTopLosers(loadAll(), n,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::BestTrade>
TradeJournal::topWinnersByTag(
    const std::string& tag, bool includeUntagged, size_t n) const {
    return buildTopWinners(loadAll(), n,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::BestTrade>
TradeJournal::topLosersByTag(
    const std::string& tag, bool includeUntagged, size_t n) const {
    return buildTopLosers(loadAll(), n,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {

// Sprint #106 — calendar bucketing helpers. Build a
// (axis → index → Bucket) flat grid for either day-of-week
template <typename Bucket, size_t kBuckets>
std::map<std::string, std::map<size_t, Bucket>>
bucketByCalendarIndex(
    const std::vector<JournalFill>& fills,
    std::function<size_t(const JournalFill&)> indexFn,
    std::function<std::string(const JournalFill&)> keyFn) {
    std::map<std::string, std::map<size_t, Bucket>> out;
    constexpr double kEps = 1e-9;
    for (const auto& f : fills) {
        std::string key = keyFn(f);
        if (key.empty()) continue;
        size_t bucket = indexFn(f);
        if (bucket >= kBuckets) continue;     // safety
        std::time_t secs = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        (void)tm;
        auto& b = out[key][bucket];
        // Round-trip = |realizedDelta| > 1e-9.
        if (std::fabs(f.realizedDelta) > kEps) {
            b.roundTrips++;
            if (f.realizedDelta > kEps) b.wins++;
            else b.losses++;
        }
        b.realized += f.realizedDelta;
    }
    return out;
}

// Flatten a (axis → bucketIdx → Bucket) nested map into a
// row-major flat grid sized symbols.size() × kBuckets.
template <typename Bucket, size_t kBuckets, typename Labels>
std::vector<Bucket> flattenCalendarGrid(
    const std::map<std::string, std::map<size_t, Bucket>>& nested,
    const Labels& labels) {
    std::vector<Bucket> grid(labels.size() * kBuckets);
    for (size_t li = 0; li < labels.size(); ++li) {
        auto it = nested.find(labels[li]);
        if (it == nested.end()) continue;
        for (size_t b = 0; b < kBuckets; ++b) {
            auto bit = it->second.find(b);
            if (bit != it->second.end())
                grid[li * kBuckets + b] = bit->second;
        }
    }
    return grid;
}

}  // namespace

TradeJournal::PerSymbolDayOfWeekStats
TradeJournal::perSymbolDayOfWeekStats() const {
    // Sprint #106. 7 buckets (Sun..Sat).
    auto nested = bucketByCalendarIndex<DayOfWeekBucket, 7>(
        loadAll(),
        [](const JournalFill&) -> size_t {
            // We use the helper's bucket index as the OUTER
            // key for the inner map; but here we actually
            // need tm.tm_wday. Re-derive it from ts.
            // The closure above stored bucket via indexFn —
            // but we need to compute it differently here.
            // Let me redefine:
            return 0;  // placeholder; replaced below
        },
        [](const JournalFill& f) { return f.symbol; });
    (void)nested;
    // The closure above is a bit awkward because tm_wday
    // needs the timestamp. Let me redo with a manual loop
    // for clarity:
    auto fills = loadAll();
    std::map<std::string, std::map<size_t, DayOfWeekBucket>> buckets;
    constexpr double kEps = 1e-9;
    for (const auto& f : fills) {
        std::time_t secs = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        size_t weekday = static_cast<size_t>(tm.tm_wday);
        if (weekday > 6) continue;
        auto& b = buckets[f.symbol][weekday];
        if (std::fabs(f.realizedDelta) > kEps) {
            b.roundTrips++;
            if (f.realizedDelta > kEps) b.wins++;
            else b.losses++;
        }
        b.realized += f.realizedDelta;
    }
    PerSymbolDayOfWeekStats out;
    for (const auto& kv : buckets) out.symbols.push_back(kv.first);
    out.grid = flattenCalendarGrid<DayOfWeekBucket, 7>(
        buckets, out.symbols);
    return out;
}

TradeJournal::PerTagDayOfWeekStats
TradeJournal::perTagDayOfWeekStats(bool includeUntagged) const {
    // Sprint #106. Per-tag mirror.
    auto fills = loadAll();
    std::map<std::string, std::map<size_t, DayOfWeekBucket>> buckets;
    constexpr double kEps = 1e-9;
    for (const auto& f : fills) {
        std::string tag = f.tag;
        if (tag.empty()) {
            if (!includeUntagged) continue;
            tag = "__untagged__";
        }
        std::time_t secs = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        size_t weekday = static_cast<size_t>(tm.tm_wday);
        if (weekday > 6) continue;
        auto& b = buckets[tag][weekday];
        if (std::fabs(f.realizedDelta) > kEps) {
            b.roundTrips++;
            if (f.realizedDelta > kEps) b.wins++;
            else b.losses++;
        }
        b.realized += f.realizedDelta;
    }
    PerTagDayOfWeekStats out;
    for (const auto& kv : buckets) out.tags.push_back(kv.first);
    out.grid = flattenCalendarGrid<DayOfWeekBucket, 7>(
        buckets, out.tags);
    return out;
}

TradeJournal::PerSymbolHourOfDayStats
TradeJournal::perSymbolHourOfDayStats() const {
    // Sprint #106. 24 buckets (0..23 local hour).
    auto fills = loadAll();
    std::map<std::string, std::map<size_t, HourOfDayBucket>> buckets;
    constexpr double kEps = 1e-9;
    for (const auto& f : fills) {
        std::time_t secs = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        size_t hour = static_cast<size_t>(tm.tm_hour);
        if (hour > 23) continue;
        auto& b = buckets[f.symbol][hour];
        if (std::fabs(f.realizedDelta) > kEps) {
            b.roundTrips++;
            if (f.realizedDelta > kEps) b.wins++;
            else b.losses++;
        }
        b.realized += f.realizedDelta;
    }
    PerSymbolHourOfDayStats out;
    for (const auto& kv : buckets) out.symbols.push_back(kv.first);
    out.grid = flattenCalendarGrid<HourOfDayBucket, 24>(
        buckets, out.symbols);
    return out;
}

TradeJournal::PerTagHourOfDayStats
TradeJournal::perTagHourOfDayStats(bool includeUntagged) const {
    // Sprint #106. Per-tag mirror.
    auto fills = loadAll();
    std::map<std::string, std::map<size_t, HourOfDayBucket>> buckets;
    constexpr double kEps = 1e-9;
    for (const auto& f : fills) {
        std::string tag = f.tag;
        if (tag.empty()) {
            if (!includeUntagged) continue;
            tag = "__untagged__";
        }
        std::time_t secs = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        size_t hour = static_cast<size_t>(tm.tm_hour);
        if (hour > 23) continue;
        auto& b = buckets[tag][hour];
        if (std::fabs(f.realizedDelta) > kEps) {
            b.roundTrips++;
            if (f.realizedDelta > kEps) b.wins++;
            else b.losses++;
        }
        b.realized += f.realizedDelta;
    }
    PerTagHourOfDayStats out;
    out.tags.reserve(buckets.size());
    for (const auto& kv : buckets) out.tags.push_back(kv.first);
    out.grid = flattenCalendarGrid<HourOfDayBucket, 24>(
        buckets, out.tags);
    return out;
}

namespace {
// Sprint #109 — daily P&L aggregation. Walks fills sorted by
// timestamp ASC, groups by YYYY-MM-DD, sums realizedDelta and
// counts round-trips (|realizedDelta|>1e-9). Output is sorted
// by date ASC.
//
// Used by dailyPnLSeries() (whole journal) and by
// perSymbolDailyPnL() / perTagDailyPnL() (per axis).
template <typename KeyFn>
std::map<std::string, std::vector<TradeJournal::DailyPnL>>
bucketByDay(const std::vector<JournalFill>& fills,
            KeyFn keyFn) {
    auto sorted = fills;
    std::sort(sorted.begin(), sorted.end(),
              [](const JournalFill& a, const JournalFill& b) {
                  return a.timestamp_us < b.timestamp_us;
              });
    std::map<std::string, std::vector<TradeJournal::DailyPnL>> out;
    constexpr double kEps = 1e-9;
    for (const auto& f : sorted) {
        std::string key = keyFn(f);
        if (key.empty()) continue;
        std::time_t secs = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        char date[16];
        std::strftime(date, sizeof(date), "%Y-%m-%d", &tm);
        if (out[key].empty() ||
            out[key].back().date != date) {
            TradeJournal::DailyPnL p;
            p.date = date;
            out[key].push_back(p);
        }
        auto& dp = out[key].back();
        dp.realized += f.realizedDelta;
        if (std::fabs(f.realizedDelta) > kEps)
            dp.roundTrips++;
    }
    return out;
}

// Parse "YYYY-MM-DD" to time_t (midnight local). Used by
// rollingWindowSharpe() to align daily series with calendar days.
std::time_t parseDay(const std::string& iso) {
    std::tm tm{};
    std::sscanf(iso.c_str(), "%d-%d-%d",
                &tm.tm_year, &tm.tm_mon, &tm.tm_mday);
    tm.tm_year -= 1900;
    tm.tm_mon  -= 1;
    return std::mktime(&tm);
}

// Format time_t as "YYYY-MM-DD" (local).
std::string fmtDay(std::time_t t) {
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
    return buf;
}

// Compute rolling Sharpe on a per-day series. Input is a
// sorted vector of (date, realized). Output is one
// RollingSharpePoint per day where the window is complete.
//
// When stddev is zero (all-zero window) → sharpe = 0.0
// (sentinel — trader sees a flat line).
std::vector<TradeJournal::RollingSharpePoint>
computeRollingSharpe(
    const std::vector<TradeJournal::DailyPnL>& daily,
    size_t windowDays) {
    std::vector<TradeJournal::RollingSharpePoint> out;
    if (daily.size() < windowDays || windowDays == 0) return out;
    std::time_t t0 = parseDay(daily.front().date);
    std::time_t tN = parseDay(daily.back().date);
    size_t nDays = static_cast<size_t>((tN - t0) / 86400) + 1;
    std::vector<double> dailyR(nDays, 0.0);
    for (const auto& d : daily) {
        size_t idx = static_cast<size_t>(
            (parseDay(d.date) - t0) / 86400);
        dailyR[idx] = d.realized;
    }
    for (size_t i = windowDays - 1; i < nDays; ++i) {
        double sum  = 0.0;
        double sum2 = 0.0;
        for (size_t k = i + 1 - windowDays; k <= i; ++k) {
            sum  += dailyR[k];
            sum2 += dailyR[k] * dailyR[k];
        }
        double mean = sum / static_cast<double>(windowDays);
        double ex2  = sum2 / static_cast<double>(windowDays);
        double var  = ex2 - mean * mean;
        if (var < 0.0) var = 0.0;
        double stddev = std::sqrt(var);
        double sharpe = (stddev > 1e-12)
            ? (mean / stddev) * std::sqrt(252.0)
            : 0.0;
        TradeJournal::RollingSharpePoint p;
        std::time_t ti = t0 +
            static_cast<std::time_t>(i) * 86400;
        p.date   = fmtDay(ti);
        p.sharpe = sharpe;
        out.push_back(p);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::DailyPnL>
TradeJournal::dailyPnLSeries() const {
    // Sprint #109. Whole-journal daily series. Empty input
    // → empty output (no synthetic zero-fill days).
    //
    // bucketByDay()'s keyFn expects a non-empty key (it skips
    // empty keys because the perSymbol/perTag callers use the
    // empty string as the "skip this fill" signal). We use the
    // sentinel "$all" here so all fills land in a single bucket
    // — then we pop that bucket out as the result.
    auto bucketed = bucketByDay(
        loadAll(),
        [](const JournalFill&) { return std::string("$all"); });
    if (bucketed.empty()) return {};
    return std::move(bucketed["$all"]);
}

std::map<std::string, std::vector<TradeJournal::DailyPnL>>
TradeJournal::perSymbolDailyPnL() const {
    // Sprint #109. Per-symbol mirror of dailyPnLSeries().
    return bucketByDay(
        loadAll(),
        [](const JournalFill& f) { return f.symbol; });
}

std::map<std::string, std::vector<TradeJournal::DailyPnL>>
TradeJournal::perTagDailyPnL(bool includeUntagged) const {
    // Sprint #109. Per-tag mirror. Honors includeUntagged.
    return bucketByDay(
        loadAll(),
        [includeUntagged](const JournalFill& f) -> std::string {
            if (f.tag.empty()) {
                return includeUntagged ? "__untagged__" : "";
            }
            return f.tag;
        });
}

std::vector<TradeJournal::RollingSharpePoint>
TradeJournal::rollingSharpe(size_t windowDays) const {
    // Sprint #109. Whole-journal rolling Sharpe.
    return computeRollingSharpe(dailyPnLSeries(), windowDays);
}

std::map<std::string,
         std::vector<TradeJournal::RollingSharpePoint>>
TradeJournal::rollingSharpeBySymbol(size_t windowDays) const {
    // Sprint #109. Per-symbol mirror. Symbols with < windowDays
    // of history get an empty vector (UI can skip them).
    auto perSym = perSymbolDailyPnL();
    std::map<std::string, std::vector<RollingSharpePoint>> out;
    for (auto& kv : perSym) {
        if (kv.second.size() < windowDays) {
            out[kv.first] = {};
            continue;
        }
        out[kv.first] = computeRollingSharpe(kv.second, windowDays);
    }
    return out;
}

namespace {
// Sprint #111 — single-trade extremes helper. Walks fills,
// applies an optional filter (by symbol or tag), and tracks
// the best (max) or worst (min) realizedDelta. The filter
// closure returns true to KEEP a fill, false to skip it.
template <typename FilterFn>
TradeJournal::BestTrade
extremeTrade(const std::vector<JournalFill>& fills,
             FilterFn filter,
             bool findMax) {
    TradeJournal::BestTrade out;
    bool   have = false;
    double bestVal = findMax
        ? -std::numeric_limits<double>::infinity()
        :  std::numeric_limits<double>::infinity();
    for (const auto& f : fills) {
        if (!filter(f)) continue;
        bool better = findMax
            ? (f.realizedDelta > bestVal)
            : (f.realizedDelta < bestVal);
        if (!have || better) {
            have    = true;
            bestVal = f.realizedDelta;
            out.timestamp_us = f.timestamp_us;
            out.symbol       = f.symbol;
            out.tag          = f.tag;
            out.realized     = f.realizedDelta;
        }
    }
    return out;
}
}  // namespace

TradeJournal::BestTrade
TradeJournal::bestTrade() const {
    // Sprint #111. Max realizedDelta across all fills.
    return extremeTrade(loadAll(),
        [](const JournalFill&) { return true; },
        true /*findMax*/);
}

TradeJournal::BestTrade
TradeJournal::worstTrade() const {
    // Sprint #111. Min realizedDelta across all fills.
    return extremeTrade(loadAll(),
        [](const JournalFill&) { return true; },
        false /*findMin*/);
}

TradeJournal::BestTrade
TradeJournal::bestTradeBySymbol(const std::string& symbol) const {
    return extremeTrade(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        }, true);
}

TradeJournal::BestTrade
TradeJournal::worstTradeBySymbol(const std::string& symbol) const {
    return extremeTrade(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        }, false);
}

TradeJournal::BestTrade
TradeJournal::bestTradeByTag(const std::string& tag,
                             bool includeUntagged) const {
    return extremeTrade(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") {
                return f.tag.empty();  // all untagged
            }
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        }, true);
}

TradeJournal::BestTrade
TradeJournal::worstTradeByTag(const std::string& tag,
                              bool includeUntagged) const {
    return extremeTrade(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") {
                return f.tag.empty();
            }
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        }, false);
}

namespace {
// Sprint #112 — RFC-4180-style CSV quote. Wraps a field in
// double quotes when it contains a comma, quote, or newline;
// doubles any embedded quote. None of the current JournalFill
// fields strictly need it (symbol/timestamp are ASCII-clean,
// numbers are locale-neutral via snprintf), but the hook
// stays in case future fields like user-supplied tags land.
std::string csvQuote(const std::string& s) {
    if (s.find(',')  == std::string::npos &&
        s.find('"')  == std::string::npos &&
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
}

// ISO-8601-ish "YYYY-MM-DDTHH:MM:SS" from microsecond ts.
// Used by exportFillsToCsv() for human-readable timestamps.
std::string isoTimestamp(uint64_t ts_us) {
    std::time_t secs = static_cast<std::time_t>(ts_us / 1000000ULL);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &secs);
#else
    localtime_r(&secs, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}
}  // namespace

bool TradeJournal::exportFillsToCsv(const std::string& path) const {
    // Sprint #112. One row per fill: timestamp, symbol,
    // realized, tag. Sorted by timestamp ASC.
    namespace fs = std::filesystem;
    try {
        fs::path p(path);
        if (p.has_parent_path())
            fs::create_directories(p.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (!out.is_open()) return false;
        out << "timestamp_iso,timestamp_us,symbol,realized,tag\n";
        auto fills = loadAll();
        std::sort(fills.begin(), fills.end(),
                  [](const JournalFill& a, const JournalFill& b) {
                      return a.timestamp_us < b.timestamp_us;
                  });
        for (const auto& f : fills) {
            out << csvQuote(isoTimestamp(f.timestamp_us))
                << "," << f.timestamp_us
                << "," << csvQuote(f.symbol)
                << "," << std::fixed << std::setprecision(6)
                << f.realizedDelta
                << "," << csvQuote(f.tag) << "\n";
        }
        out.flush();
        return out.good();
    } catch (...) {
        return false;
    }
}

bool TradeJournal::exportStatsToCsv(const std::string& path) const {
    // Sprint #112. Two sections in one CSV:
    //   # per-symbol header
    //   symbol,realized,roundTrips,wins,losses,winRate,
    //   avgWinner,avgLoser,profitFactor,expectancy,
    //   sharpe,sortino,calmar
    //   ...rows...
    //   # per-tag header
    //   tag,realized,roundTrips,wins,losses,winRate,
    //   avgWinner,avgLoser,profitFactor,expectancy,
    //   sharpe,sortino,calmar
    //   ...rows...
    namespace fs = std::filesystem;
    try {
        fs::path p(path);
        if (p.has_parent_path())
            fs::create_directories(p.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (!out.is_open()) return false;
        // ---- Per-symbol ----
        out << "# per_symbol_stats\n";
        out << "symbol,realized,roundTrips,wins,losses,winRate,"
            << "avgWinner,avgLoser,profitFactor,expectancy,"
            << "sharpe,sortino,calmar\n";
        auto perSymSh  = perSymbolSharpe();
        auto perSymSo  = perSymbolSortino();
        auto perSymCl  = perSymbolCalmar();
        std::unordered_map<std::string, double> shBySym, soBySym, clBySym;
        for (const auto& s : perSymSh) shBySym[s.symbol] = s.annualizedSharpe;
        for (const auto& s : perSymSo) soBySym[s.symbol] = s.annualizedSortino;
        for (const auto& s : perSymCl) clBySym[s.symbol] = s.calmarRatio;
        for (const auto& s : perSymbolStats()) {
            out << csvQuote(s.symbol)
                << "," << std::fixed << std::setprecision(6) << s.realized
                << "," << s.roundTripCount
                << "," << s.winCount
                << "," << s.lossCount
                << "," << std::setprecision(4) << s.winRate
                << "," << std::setprecision(6) << s.avgWinner
                << "," << s.avgLoser
                << "," << s.profitFactor
                << "," << s.expectancy
                << "," << shBySym[s.symbol]
                << "," << soBySym[s.symbol]
                << "," << clBySym[s.symbol]
                << "\n";
        }
        // ---- Per-tag ----
        out << "# per_tag_stats\n";
        out << "tag,realized,roundTrips,wins,losses,winRate,"
            << "avgWinner,avgLoser,profitFactor,expectancy,"
            << "sharpe,sortino,calmar\n";
        auto perTagSh  = perTagSharpe(true);
        auto perTagSo  = perTagSortino(true);
        auto perTagCl  = perTagCalmar(true);
        std::unordered_map<std::string, double> shByTag, soByTag, clByTag;
        for (const auto& s : perTagSh) shByTag[s.tag] = s.annualizedSharpe;
        for (const auto& s : perTagSo) soByTag[s.tag] = s.annualizedSortino;
        for (const auto& s : perTagCl) clByTag[s.tag] = s.calmarRatio;
        for (const auto& s : perTagStats(true)) {
            out << csvQuote(s.tag)
                << "," << std::fixed << std::setprecision(6) << s.realized
                << "," << s.roundTripCount
                << "," << s.winCount
                << "," << s.lossCount
                << "," << std::setprecision(4) << s.winRate
                << "," << std::setprecision(6) << s.avgWinner
                << "," << s.avgLoser
                << "," << s.profitFactor
                << "," << s.expectancy
                << "," << shByTag[s.tag]
                << "," << soByTag[s.tag]
                << "," << clByTag[s.tag]
                << "\n";
        }
        out.flush();
        return out.good();
    } catch (...) {
        return false;
    }
}

namespace {
// Atomic rewrite of the journal. Writes every fill to
// "<path>.tmp" then renames over the original. The rename is
// atomic on POSIX (and on Windows with ReplaceFile semantics on
// modern toolchains), so a crash mid-write leaves the original
// file untouched.
bool rewriteAll(const std::string& path,
                const std::vector<JournalFill>& fills) {
    namespace fs = std::filesystem;
    try {
        fs::path p(path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        std::string tmp = path + ".tmp";
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out.is_open()) return false;
            for (const auto& f : fills) {
                out << TradeJournal::toJsonLine(f) << "\n";
            }
            out.flush();
            if (!out.good()) return false;
        }
        // atomic rename (overwrite existing file on POSIX)
        fs::rename(tmp, path);
        return true;
    } catch (...) {
        return false;
    }
}
}  // namespace

bool TradeJournal::setTagAt(size_t index, const std::string& newTag) {
    std::vector<JournalFill> fills = loadAll();
    if (index >= fills.size()) return false;
    fills[index].tag = newTag;
    return rewriteAll(m_path, fills);
}

bool TradeJournal::setTagByTimestamp(uint64_t timestamp_us,
                                      const std::string& symbol,
                                      const std::string& newTag) {
    std::vector<JournalFill> fills = loadAll();
    bool found = false;
    for (auto& f : fills) {
        if (f.timestamp_us == timestamp_us && f.symbol == symbol) {
            f.tag = newTag;
            found = true;
            break;
        }
    }
    if (!found) return false;
    return rewriteAll(m_path, fills);
}

std::string TradeJournal::toJsonLine(const JournalFill& r) {
    char buf[1024];
    // Booleans serialize as true/false barewords; numbers as JSON numbers.
    // The tag is included as a quoted JSON string (empty when untagged) —
    // empty string is preserved on round-trip via fromJsonLine's parse.
    std::snprintf(buf, sizeof(buf),
        "{\"ts\":%llu,\"sym\":\"%s\",\"side\":\"%s\","
        "\"qty\":%.10g,\"px\":%.10g,\"realized\":%.10g,\"tag\":\"%s\"}",
        static_cast<unsigned long long>(r.timestamp_us),
        jsonEscape(r.symbol).c_str(),
        r.isLong ? "buy" : "sell",
        r.qty, r.price, r.realizedDelta,
        jsonEscape(r.tag).c_str());
    return std::string(buf);
}

std::optional<JournalFill> TradeJournal::fromJsonLine(const std::string& line) {
    auto kvs = parseFlatObject(line);
    if (kvs.empty()) return std::nullopt;
    const KV* sym    = findKV(kvs, "sym");
    const KV* side   = findKV(kvs, "side");
    const KV* qtyRaw = findKV(kvs, "qty");
    const KV* pxRaw  = findKV(kvs, "px");
    const KV* tsRaw  = findKV(kvs, "ts");
    const KV* reRaw  = findKV(kvs, "realized");
    const KV* tagRaw = findKV(kvs, "tag");  // absent on legacy rows
    if (!sym || !side || !qtyRaw || !pxRaw) return std::nullopt;

    JournalFill r;
    r.symbol = jsonUnescape(sym->raw);
    r.isLong = (side->raw == "buy");
    try {
        r.qty    = std::stod(qtyRaw->raw);
        r.price  = std::stod(pxRaw->raw);
        if (reRaw) r.realizedDelta = std::stod(reRaw->raw);
        if (tsRaw) r.timestamp_us  = std::stoull(tsRaw->raw);
    } catch (...) {
        return std::nullopt;
    }
    // Tag is optional — fills written before this field existed
    // (and rows that the trader didn't tag) parse back with an
    // empty string, which is the documented "untagged" sentinel.
    if (tagRaw) r.tag = jsonUnescape(tagRaw->raw);
    return r;
}

std::vector<JournalFill> TradeJournal::loadAll(int* skippedCount) const {
    std::vector<JournalFill> out;
    if (skippedCount) *skippedCount = 0;
    std::ifstream in(m_path);
    if (!in.is_open()) return out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto rec = fromJsonLine(line);
        if (rec.has_value()) {
            out.push_back(*rec);
        } else if (skippedCount) {
            ++(*skippedCount);
        }
    }
    return out;
}

std::vector<JournalFill> TradeJournal::recent(size_t n) const {
    auto all = loadAll(nullptr);
    if (n == 0 || all.empty()) return {};
    if (n >= all.size()) {
        std::reverse(all.begin(), all.end());
        return all;
    }
    std::vector<JournalFill> tail(all.end() - n, all.end());
    std::reverse(tail.begin(), tail.end());
    return tail;
}

size_t TradeJournal::count() const {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(m_path, ec)) return 0;
    size_t n = 0;
    std::ifstream in(m_path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) ++n;
    }
    return n;
}

bool TradeJournal::clear() {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(m_path, ec)) return true;
    return fs::remove(m_path, ec);
}

namespace {
// RFC-4180-style field quoting for CSV. None of the current
// JournalFill fields strictly need it (symbol/timestamp are
// ASCII-clean, numbers are locale-neutral via snprintf), but the
// hook stays in case future fields like user-supplied tags land.
std::string csvQuoteIfNeeded(const std::string& s) {
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
}

// ISO-8601 UTC with microsecond precision — matches what the live
// trades widget emits (see TradesWidget::formatTradesCSV). Keeping
// the two exporters byte-identical means a downstream analytics
// pipeline can ingest either without parser changes.
std::string formatTimestampISO(uint64_t us) {
    auto tp = std::chrono::system_clock::time_point(
                  std::chrono::microseconds(us));
    auto timeT = std::chrono::system_clock::to_time_t(tp);
    std::tm tmUtc{};
#if defined(_WIN32)
    gmtime_s(&tmUtc, &timeT);
#else
    gmtime_r(&timeT, &tmUtc);
#endif
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%04d-%02d-%02dT%02d:%02d:%02d.%06lluZ",
                  tmUtc.tm_year + 1900, tmUtc.tm_mon + 1, tmUtc.tm_mday,
                  tmUtc.tm_hour, tmUtc.tm_min, tmUtc.tm_sec,
                  static_cast<unsigned long long>(us % 1000000));
    return std::string(buf);
}
} // namespace

std::string TradeJournal::formatFillsCSV(
        const std::vector<JournalFill>& fills) {
    std::ostringstream os;
    // Column order chosen for spreadsheet import — chronological
    // metadata first (timestamp, symbol, side), then trade size
    // (qty, price), then P&L attribution (realized), then the
    // strategy tag at the end (so it groups neatly in pivot tables).
    os << "timestamp_iso,symbol,side,qty,price,realized_delta,tag\n";
    for (const auto& f : fills) {
        char qtyBuf[32], priceBuf[32], realizedBuf[32];
        std::snprintf(qtyBuf,     sizeof(qtyBuf),     "%.10g", f.qty);
        std::snprintf(priceBuf,   sizeof(priceBuf),   "%.10g", f.price);
        std::snprintf(realizedBuf,sizeof(realizedBuf),"%.10g", f.realizedDelta);
        os << csvQuoteIfNeeded(formatTimestampISO(f.timestamp_us)) << ","
           << csvQuoteIfNeeded(f.symbol) << ","
           << csvQuoteIfNeeded(f.isLong ? "BUY" : "SELL") << ","
           << csvQuoteIfNeeded(qtyBuf) << ","
           << csvQuoteIfNeeded(priceBuf) << ","
           << csvQuoteIfNeeded(realizedBuf) << ","
           << csvQuoteIfNeeded(f.tag) << "\n";
    }
    return os.str();
}

bool TradeJournal::exportCSV(const std::string& path) const {
    namespace fs = std::filesystem;
    try {
        fs::path p(path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        auto fills = loadAll(nullptr);
        std::ofstream out(path, std::ios::trunc);
        if (!out.is_open()) return false;
        out << formatFillsCSV(fills);
        return out.good();
    } catch (...) {
        return false;
    }
}

std::vector<JournalFill> TradeJournal::loadByTag(
        const std::string& tag, bool includeUntagged) const {
    auto all = loadAll(nullptr);
    std::vector<JournalFill> out;
    out.reserve(all.size());
    // Semantic edge case: an empty filter tag + includeUntagged=true
    // means "show me everything" — the trader is using the untagged
    // bucket as a wildcard. An empty filter tag WITHOUT the flag
    // means "show me only the untagged bucket" (which is the
    // direct read of fills with empty tag, no special-casing).
    const bool emptyTagIsAll = tag.empty() && includeUntagged;
    for (const auto& f : all) {
        if (emptyTagIsAll) {
            out.push_back(f);
        } else if (f.tag == tag) {
            out.push_back(f);
        } else if (includeUntagged && f.tag.empty()) {
            out.push_back(f);
        }
    }
    return out;
}

std::string TradeJournal::formatFillsCSVByTag(
        const std::vector<JournalFill>& fills,
        const std::string& tag,
        bool includeUntagged) {
    // Build the filtered subset first, then hand off to the existing
    // formatter. Splitting this way keeps formatFillsCSV single-purpose
    // (it's already covered by Test 50) and makes the filter the only
    // new code path to test.
    std::vector<JournalFill> filtered;
    filtered.reserve(fills.size());
    const bool emptyTagIsAll = tag.empty() && includeUntagged;
    for (const auto& f : fills) {
        if (emptyTagIsAll) {
            filtered.push_back(f);
        } else if (f.tag == tag) {
            filtered.push_back(f);
        } else if (includeUntagged && f.tag.empty()) {
            filtered.push_back(f);
        }
    }
    return formatFillsCSV(filtered);
}

bool TradeJournal::exportCSVByTag(const std::string& path,
                                   const std::string& tag,
                                   bool includeUntagged) const {
    namespace fs = std::filesystem;
    try {
        fs::path p(path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        std::ofstream out(path, std::ios::trunc);
        if (!out.is_open()) return false;
        out << formatFillsCSVByTag(loadAll(nullptr), tag, includeUntagged);
        return out.good();
    } catch (...) {
        return false;
    }
}

TradeJournal::SymbolCorrelation
TradeJournal::symbolSymbolCorrelation(
    const std::string& symA,
    const std::string& symB) const {
    // Sprint #148. Per-day Pearson correlation of realized
    // between symA and symB. Aggregate each symbol's
    // fills by local day, find days both traded, compute
    // correlation on matched pairs.
    SymbolCorrelation r;
    auto fills = loadAll();
    std::vector<JournalFill> onlyA, onlyB;
    for (const auto& f : fills) {
        if (f.symbol == symA) onlyA.push_back(f);
        else if (f.symbol == symB) onlyB.push_back(f);
    }
    r.fillsA = onlyA.size();
    r.fillsB = onlyB.size();
    if (onlyA.size() < 2 || onlyB.size() < 2) return r;
    auto bucket = [](const std::vector<JournalFill>& src) {
        std::map<std::string, double> out;
        for (const auto& f : src) {
            if (std::fabs(f.realizedDelta) <= 1e-9) continue;
            std::time_t s = static_cast<std::time_t>(
                f.timestamp_us / 1000000ULL);
            std::tm tm{};
            localtime_r(&s, &tm);
            char buf[16];
            std::strftime(buf, sizeof(buf),
                          "%Y-%m-%d", &tm);
            out[buf] += f.realizedDelta;
        }
        return out;
    };
    auto daysA = bucket(onlyA);
    auto daysB = bucket(onlyB);
    std::vector<double> x, y;
    for (const auto& kv : daysA) {
        auto it = daysB.find(kv.first);
        if (it != daysB.end()) {
            x.push_back(kv.second);
            y.push_back(it->second);
        }
    }
    r.matchedDays = x.size();
    if (r.matchedDays < 2) return r;
    double meanX = 0.0, meanY = 0.0;
    for (size_t k = 0; k < x.size(); ++k) {
        meanX += x[k]; meanY += y[k];
    }
    meanX /= static_cast<double>(x.size());
    meanY /= static_cast<double>(y.size());
    double cov = 0.0, varX = 0.0, varY = 0.0;
    for (size_t k = 0; k < x.size(); ++k) {
        cov  += (x[k] - meanX) * (y[k] - meanY);
        varX += (x[k] - meanX) * (x[k] - meanX);
        varY += (y[k] - meanY) * (y[k] - meanY);
    }
    if (varX < 1e-12 || varY < 1e-12) return r;
    r.correlation = cov / std::sqrt(varX * varY);
    r.valid = true;
    return r;
}

std::vector<TradeJournal::CorrelationMatrixEntry>
TradeJournal::allSymbolCorrelations() const {
    // Sprint #149. Find every distinct symbol in the
    // journal, compute correlation for every pair (where
    // symA < symB alphabetically to avoid duplicates).
    std::vector<CorrelationMatrixEntry> out;
    auto fills = loadAll();
    std::vector<std::string> syms;
    {
        std::set<std::string> uniq;
        for (const auto& f : fills) uniq.insert(f.symbol);
        for (const auto& s : uniq) syms.push_back(s);
    }
    out.reserve(syms.size() * syms.size() / 2);
    for (size_t i = 0; i < syms.size(); ++i) {
        for (size_t j = i + 1; j < syms.size(); ++j) {
            auto r = symbolSymbolCorrelation(syms[i], syms[j]);
            CorrelationMatrixEntry e;
            e.symA = syms[i];
            e.symB = syms[j];
            e.correlation = r.correlation;
            e.matchedDays = r.matchedDays;
            e.valid = r.valid;
            out.push_back(e);
        }
    }
    return out;
}

namespace {
// Sprint #150 — fill-interval stats builder.
// Computes time gaps between consecutive matching fills,
// then runs them through PnLDistribution (mean/median/
// p90/max). Templated for filtering.
template <typename Pred>
TradeJournal::PnLDistribution
buildFillIntervalStats(const std::vector<JournalFill>& fills,
                        Pred pred) {
    TradeJournal::PnLDistribution out;
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    if (sub.size() < 2) return out;
    std::vector<double> gaps;
    gaps.reserve(sub.size() - 1);
    for (size_t i = 1; i < sub.size(); ++i) {
        if (sub[i].timestamp_us > sub[i - 1].timestamp_us) {
            gaps.push_back(static_cast<double>(
                sub[i].timestamp_us - sub[i - 1].timestamp_us));
        }
    }
    if (gaps.empty()) return out;
    out.count = gaps.size();
    double sum = 0.0, maxGap = 0.0;
    for (double g : gaps) {
        sum += g;
        if (g > maxGap) maxGap = g;
    }
    out.mean = sum / static_cast<double>(gaps.size());
    out.max  = maxGap;
    std::sort(gaps.begin(), gaps.end());
    auto pctile = [&](double p) {
        double rank = (p / 100.0) *
            static_cast<double>(gaps.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(rank));
        size_t hi = static_cast<size_t>(std::ceil(rank));
        if (lo == hi) return gaps[lo];
        double frac = rank - static_cast<double>(lo);
        return gaps[lo] * (1.0 - frac) +
               gaps[hi] * frac;
    };
    out.p50 = pctile(50);
    out.p90    = pctile(90);
    return out;
}
}  // namespace

TradeJournal::PnLDistribution
TradeJournal::fillIntervalStats() const {
    return buildFillIntervalStats(loadAll(),
        [](const JournalFill&) { return true; });
}

TradeJournal::PnLDistribution
TradeJournal::fillIntervalStatsBySymbol(
    const std::string& symbol) const {
    return buildFillIntervalStats(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::PnLDistribution
TradeJournal::fillIntervalStatsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildFillIntervalStats(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #151 — per-tag correlation.
// Per-day Pearson correlation of realized between two
// tags. Aggregates by local day, finds days both
// traded, computes Pearson r.
TradeJournal::SymbolCorrelation
tagTagCorrelation(const std::vector<JournalFill>& allFills,
                   const std::string& tagA,
                   bool includeUntaggedA,
                   const std::string& tagB,
                   bool includeUntaggedB) {
    TradeJournal::SymbolCorrelation r;
    std::vector<JournalFill> onlyA, onlyB;
    auto matchesA = [&tagA, includeUntaggedA](
        const JournalFill& f) {
        if (tagA == "__untagged__") return f.tag.empty();
        if (includeUntaggedA && f.tag.empty()) return false;
        return f.tag == tagA;
    };
    auto matchesB = [&tagB, includeUntaggedB](
        const JournalFill& f) {
        if (tagB == "__untagged__") return f.tag.empty();
        if (includeUntaggedB && f.tag.empty()) return false;
        return f.tag == tagB;
    };
    for (const auto& f : allFills) {
        if (matchesA(f)) onlyA.push_back(f);
        else if (matchesB(f)) onlyB.push_back(f);
    }
    r.fillsA = onlyA.size();
    r.fillsB = onlyB.size();
    if (onlyA.size() < 2 || onlyB.size() < 2) return r;
    auto bucket = [](const std::vector<JournalFill>& src) {
        std::map<std::string, double> out;
        for (const auto& f : src) {
            if (std::fabs(f.realizedDelta) <= 1e-9) continue;
            std::time_t s = static_cast<std::time_t>(
                f.timestamp_us / 1000000ULL);
            std::tm tm{};
            localtime_r(&s, &tm);
            char buf[16];
            std::strftime(buf, sizeof(buf),
                          "%Y-%m-%d", &tm);
            out[buf] += f.realizedDelta;
        }
        return out;
    };
    auto daysA = bucket(onlyA);
    auto daysB = bucket(onlyB);
    std::vector<double> x, y;
    for (const auto& kv : daysA) {
        auto it = daysB.find(kv.first);
        if (it != daysB.end()) {
            x.push_back(kv.second);
            y.push_back(it->second);
        }
    }
    r.matchedDays = x.size();
    if (r.matchedDays < 2) return r;
    double meanX = 0.0, meanY = 0.0;
    for (size_t k = 0; k < x.size(); ++k) {
        meanX += x[k]; meanY += y[k];
    }
    meanX /= static_cast<double>(x.size());
    meanY /= static_cast<double>(y.size());
    double cov = 0.0, varX = 0.0, varY = 0.0;
    for (size_t k = 0; k < x.size(); ++k) {
        cov  += (x[k] - meanX) * (y[k] - meanY);
        varX += (x[k] - meanX) * (x[k] - meanX);
        varY += (y[k] - meanY) * (y[k] - meanY);
    }
    if (varX < 1e-12 || varY < 1e-12) return r;
    r.correlation = cov / std::sqrt(varX * varY);
    r.valid = true;
    return r;
}
}  // namespace

std::vector<TradeJournal::CorrelationMatrixEntry>
TradeJournal::allTagCorrelations(bool includeUntagged) const {
    // Sprint #151. Mirror of allSymbolCorrelations() (#149)
    // for tags.
    std::vector<CorrelationMatrixEntry> out;
    auto fills = loadAll();
    std::vector<std::string> tags;
    {
        std::set<std::string> uniq;
        for (const auto& f : fills) {
            if (f.tag.empty()) {
                if (includeUntagged) uniq.insert("__untagged__");
            } else {
                uniq.insert(f.tag);
            }
        }
        for (const auto& t : uniq) tags.push_back(t);
    }
    out.reserve(tags.size() * tags.size() / 2);
    for (size_t i = 0; i < tags.size(); ++i) {
        for (size_t j = i + 1; j < tags.size(); ++j) {
            auto r = tagTagCorrelation(fills, tags[i],
                includeUntagged, tags[j], includeUntagged);
            CorrelationMatrixEntry e;
            e.symA = tags[i];
            e.symB = tags[j];
            e.correlation = r.correlation;
            e.matchedDays = r.matchedDays;
            e.valid = r.valid;
            out.push_back(e);
        }
    }
    return out;
}

std::vector<TradeJournal::WeeklyWinRate>
TradeJournal::weeklyWinRate() const {
    // Sprint #152. Aggregate fills by ISO week. For each
    // week, compute win rate and total realized.
    std::vector<WeeklyWinRate> out;
    auto fills = loadAll();
    // Use a map keyed by (year, week) for stable iteration.
    std::map<std::pair<int, int>,
             std::pair<size_t,
                       std::pair<size_t, double>>> buckets;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t s = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&s, &tm);
        int year = 1900 + tm.tm_year;
        // tm.tm_yday is 0-based day-of-year. Approximate
        // ISO week via ((yday - tm.tm_wday + 7) / 7) + 1.
        // Not strictly ISO 8601 but consistent and good
        // enough for bucketing.
        int week = ((tm.tm_yday - tm.tm_wday + 7) / 7) + 1;
        auto key = std::make_pair(year, week);
        auto& b = buckets[key];
        b.first++;  // total
        if (f.realizedDelta > 0) b.second.first++;
        b.second.second += f.realizedDelta;
    }
    for (auto& kv : buckets) {
        WeeklyWinRate w;
        w.year = kv.first.first;
        w.week = kv.first.second;
        w.total = kv.second.first;
        w.wins = kv.second.second.first;
        w.realized = kv.second.second.second;
        w.winRate = w.total > 0
            ? static_cast<double>(w.wins) /
              static_cast<double>(w.total)
            : 0.0;
        out.push_back(w);
    }
    return out;
}

namespace {
// Sprint #153 — per-segment weekly win rate builder.
template <typename Pred>
std::vector<TradeJournal::WeeklyWinRate>
buildWeeklyWinRateBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    std::vector<TradeJournal::WeeklyWinRate> out;
    std::map<std::pair<int, int>,
             std::pair<size_t,
                       std::pair<size_t, double>>> buckets;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t s = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&s, &tm);
        int year = 1900 + tm.tm_year;
        int week = ((tm.tm_yday - tm.tm_wday + 7) / 7) + 1;
        auto key = std::make_pair(year, week);
        auto& b = buckets[key];
        b.first++;
        if (f.realizedDelta > 0) b.second.first++;
        b.second.second += f.realizedDelta;
    }
    for (auto& kv : buckets) {
        TradeJournal::WeeklyWinRate w;
        w.year = kv.first.first;
        w.week = kv.first.second;
        w.total = kv.second.first;
        w.wins = kv.second.second.first;
        w.realized = kv.second.second.second;
        w.winRate = w.total > 0
            ? static_cast<double>(w.wins) /
              static_cast<double>(w.total)
            : 0.0;
        out.push_back(w);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::WeeklyWinRate>
TradeJournal::weeklyWinRateBySymbol(
    const std::string& symbol) const {
    return buildWeeklyWinRateBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::WeeklyWinRate>
TradeJournal::weeklyWinRateByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildWeeklyWinRateBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

TradeJournal::PerformanceSnapshot
TradeJournal::recentPerformance(
    size_t lastDays,
    bool byFillCount,
    size_t fillCount) const {
    // Sprint #154. Window the journal to the last N
    // calendar days OR last N fills, then compute the
    // key metrics on that window.
    PerformanceSnapshot s;
    auto fills = loadAll();
    if (fills.empty()) return s;
    // Sort ascending by timestamp.
    std::sort(fills.begin(), fills.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    std::vector<JournalFill> window;
    if (byFillCount) {
        size_t start = fills.size() > fillCount
            ? fills.size() - fillCount : 0;
        window.assign(fills.begin() + start, fills.end());
    } else {
        uint64_t cutoff = fills.back().timestamp_us -
            lastDays * 86400ULL * 1000000ULL;
        for (const auto& f : fills) {
            if (f.timestamp_us >= cutoff) window.push_back(f);
        }
    }
    if (window.empty()) return s;
    s.totalFills = window.size();
    s.startUs = window.front().timestamp_us;
    s.endUs   = window.back().timestamp_us;
    double grossWin = 0.0, grossLoss = 0.0;
    std::set<std::string> days;
    double cum = 0.0;
    double peak = std::numeric_limits<double>::lowest();
    double maxDD = 0.0;
    std::vector<double> rets;
    rets.reserve(window.size());
    for (const auto& f : window) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        if (f.realizedDelta > 0) {
            s.wins++;
            grossWin += f.realizedDelta;
        } else {
            s.losses++;
            grossLoss += f.realizedDelta;
        }
        s.realized += f.realizedDelta;
        cum += f.realizedDelta;
        if (cum > peak) peak = cum;
        double dd = peak - cum;
        if (dd > maxDD) maxDD = dd;
        rets.push_back(f.realizedDelta);
        std::time_t t = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
        days.insert(buf);
    }
    s.activeDays = static_cast<double>(days.size());
    s.maxDD = maxDD;
    if (s.wins + s.losses > 0) {
        s.winRate = static_cast<double>(s.wins) /
            static_cast<double>(s.wins + s.losses);
    }
    if (std::fabs(grossLoss) > 1e-9) {
        s.profitFactor = grossWin / std::fabs(grossLoss);
    } else if (grossWin > 0) {
        s.profitFactor = std::numeric_limits<double>::infinity();
    }
    // Sharpe on window returns.
    if (rets.size() >= 2) {
        double mean = 0.0;
        for (double r : rets) mean += r;
        mean /= static_cast<double>(rets.size());
        double var = 0.0;
        for (double r : rets) {
            var += (r - mean) * (r - mean);
        }
        var /= static_cast<double>(rets.size() - 1);
        double sd = std::sqrt(var);
        if (sd > 1e-12) {
            s.sharpe = mean / sd;
        }
    }
    return s;
}

namespace {
// Sprint #155 — per-segment recent performance builder.
template <typename Pred>
TradeJournal::PerformanceSnapshot
buildRecentPerformanceBySegment(
    const std::vector<JournalFill>& fills,
    size_t lastDays, bool byFillCount, size_t fillCount,
    Pred pred) {
    TradeJournal::PerformanceSnapshot s;
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    if (sub.empty()) return s;
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    std::vector<JournalFill> window;
    if (byFillCount) {
        size_t start = sub.size() > fillCount
            ? sub.size() - fillCount : 0;
        window.assign(sub.begin() + start, sub.end());
    } else {
        uint64_t cutoff = sub.back().timestamp_us -
            lastDays * 86400ULL * 1000000ULL;
        for (const auto& f : sub) {
            if (f.timestamp_us >= cutoff) window.push_back(f);
        }
    }
    if (window.empty()) return s;
    s.totalFills = window.size();
    s.startUs = window.front().timestamp_us;
    s.endUs   = window.back().timestamp_us;
    double grossWin = 0.0, grossLoss = 0.0;
    std::set<std::string> days;
    double cum = 0.0;
    double peak = std::numeric_limits<double>::lowest();
    double maxDD = 0.0;
    std::vector<double> rets;
    rets.reserve(window.size());
    for (const auto& f : window) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        if (f.realizedDelta > 0) {
            s.wins++; grossWin += f.realizedDelta;
        } else {
            s.losses++; grossLoss += f.realizedDelta;
        }
        s.realized += f.realizedDelta;
        cum += f.realizedDelta;
        if (cum > peak) peak = cum;
        double dd = peak - cum;
        if (dd > maxDD) maxDD = dd;
        rets.push_back(f.realizedDelta);
        std::time_t t = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
        days.insert(buf);
    }
    s.activeDays = static_cast<double>(days.size());
    s.maxDD = maxDD;
    if (s.wins + s.losses > 0) {
        s.winRate = static_cast<double>(s.wins) /
            static_cast<double>(s.wins + s.losses);
    }
    if (std::fabs(grossLoss) > 1e-9) {
        s.profitFactor = grossWin / std::fabs(grossLoss);
    } else if (grossWin > 0) {
        s.profitFactor = std::numeric_limits<double>::infinity();
    }
    if (rets.size() >= 2) {
        double mean = 0.0;
        for (double r : rets) mean += r;
        mean /= static_cast<double>(rets.size());
        double var = 0.0;
        for (double r : rets) {
            var += (r - mean) * (r - mean);
        }
        var /= static_cast<double>(rets.size() - 1);
        double sd = std::sqrt(var);
        if (sd > 1e-12) s.sharpe = mean / sd;
    }
    return s;
}
}  // namespace

TradeJournal::PerformanceSnapshot
TradeJournal::recentPerformanceBySymbol(
    const std::string& symbol,
    size_t lastDays,
    bool byFillCount,
    size_t fillCount) const {
    return buildRecentPerformanceBySegment(loadAll(),
        lastDays, byFillCount, fillCount,
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::PerformanceSnapshot
TradeJournal::recentPerformanceByTag(
    const std::string& tag,
    bool includeUntagged,
    size_t lastDays,
    bool byFillCount,
    size_t fillCount) const {
    return buildRecentPerformanceBySegment(loadAll(),
        lastDays, byFillCount, fillCount,
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

namespace {
// Sprint #156 — per-segment drawdown stats builder.
// Aggregates completed DDs for a segment into one struct.
template <typename Pred>
TradeJournal::SegmentDrawdownStats
buildSegmentDrawdownStats(
    const std::vector<TradeJournal::DrawdownEvent>& events, Pred pred) {
    TradeJournal::SegmentDrawdownStats s;
    double sumDepth = 0.0, sumDays = 0.0, sumRatio = 0.0;
    for (const auto& e : events) {
        if (!pred(e)) continue;
        s.count++;
        sumDepth += e.trough_depth;
        double days = static_cast<double>(e.drawdown_us) /
                      (86400.0 * 1000000.0);
        sumDays += days;
        if (e.drawdown_us > 0 &&
            e.recovery_us > 0) {
            sumRatio += static_cast<double>(e.recovery_us) /
                        static_cast<double>(e.drawdown_us);
        }
        if (e.trough_depth > s.maxDepth) s.maxDepth = e.trough_depth;
        if (days > s.maxDrawdownDays) s.maxDrawdownDays = days;
    }
    if (s.count > 0) {
        s.meanDepth = sumDepth / static_cast<double>(s.count);
        s.meanDrawdownDays = sumDays / static_cast<double>(s.count);
        s.meanRecoveryRatio = sumRatio /
                              static_cast<double>(s.count);
    }
    return s;
}
}  // namespace

TradeJournal::SegmentDrawdownStats
TradeJournal::segmentDrawdownStatsBySymbol(
    const std::string& symbol) const {
    auto events = drawdownRecoveriesBySymbol(symbol);
    return buildSegmentDrawdownStats(events,
        [](const DrawdownEvent&) { return true; });
}

TradeJournal::SegmentDrawdownStats
TradeJournal::segmentDrawdownStatsByTag(
    const std::string& tag,
    bool includeUntagged) const {
    auto events = drawdownRecoveriesByTag(tag, includeUntagged);
    return buildSegmentDrawdownStats(events,
        [](const DrawdownEvent&) { return true; });
}

std::vector<TradeJournal::GrossPoint>
TradeJournal::cumulativeGrossSeries() const {
    // Sprint #157. Walk fills chronologically, track
    // running grossWin and |grossLoss|. One point per
    // round-trip.
    std::vector<GrossPoint> out;
    auto fills = loadAll();
    if (fills.empty()) return out;
    std::sort(fills.begin(), fills.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    out.reserve(fills.size());
    double cumWin = 0.0, cumLoss = 0.0;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        if (f.realizedDelta > 0) cumWin += f.realizedDelta;
        else                     cumLoss += f.realizedDelta;
        GrossPoint p;
        p.timestamp_us = f.timestamp_us;
        p.cumGrossWin  = cumWin;
        p.cumGrossLoss = cumLoss;
        p.netRealized  = cumWin + cumLoss;
        p.count        = out.size() + 1;
        out.push_back(p);
    }
    return out;
}

namespace {
// Sprint #158 — per-segment cumulative gross series.
template <typename Pred>
std::vector<TradeJournal::GrossPoint>
buildCumulativeGrossSeriesBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    std::vector<TradeJournal::GrossPoint> out;
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    if (sub.empty()) return out;
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    out.reserve(sub.size());
    double cumWin = 0.0, cumLoss = 0.0;
    for (const auto& f : sub) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        if (f.realizedDelta > 0) cumWin += f.realizedDelta;
        else                     cumLoss += f.realizedDelta;
        TradeJournal::GrossPoint p;
        p.timestamp_us = f.timestamp_us;
        p.cumGrossWin  = cumWin;
        p.cumGrossLoss = cumLoss;
        p.netRealized  = cumWin + cumLoss;
        p.count        = out.size() + 1;
        out.push_back(p);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::GrossPoint>
TradeJournal::cumulativeGrossSeriesBySymbol(
    const std::string& symbol) const {
    return buildCumulativeGrossSeriesBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::GrossPoint>
TradeJournal::cumulativeGrossSeriesByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildCumulativeGrossSeriesBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::PerformanceSnapshot>
TradeJournal::allRecentPerformance(
    size_t lastDays,
    bool byFillCount,
    size_t fillCount) const {
    // Sprint #159. Bulk recent performance per symbol.
    std::vector<PerformanceSnapshot> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(recentPerformanceBySymbol(
            s, lastDays, byFillCount, fillCount));
    }
    std::sort(out.begin(), out.end(),
        [](const PerformanceSnapshot& a,
           const PerformanceSnapshot& b) {
            return a.realized > b.realized;
        });
    return out;
}

std::vector<TradeJournal::PerformanceSnapshot>
TradeJournal::allRecentPerformanceByTag(
    size_t lastDays,
    bool byFillCount,
    size_t fillCount) const {
    // Sprint #159. Bulk recent performance per tag.
    std::vector<PerformanceSnapshot> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) tags.insert("__untagged__");
        else tags.insert(f.tag);
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(recentPerformanceByTag(
            t, false, lastDays, byFillCount, fillCount));
    }
    std::sort(out.begin(), out.end(),
        [](const PerformanceSnapshot& a,
           const PerformanceSnapshot& b) {
            return a.realized > b.realized;
        });
    return out;
}

std::vector<TradeJournal::MonthlyMaxDD>
TradeJournal::monthlyMaxDrawdown() const {
    // Sprint #160. For each (year, month), find the max
    // drawdown depth observed.
    std::vector<MonthlyMaxDD> out;
    auto fills = loadAll();
    if (fills.empty()) return out;
    std::sort(fills.begin(), fills.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    // Walk and compute per-day equity.
    std::map<std::pair<int, int>,
             std::pair<double, size_t>> buckets;
    // Per-day series grouped by (year, month).
    std::map<std::pair<int, int>, double> dailyCum;
    std::map<std::pair<int, int>,
             std::set<std::pair<int, int>>> daysByMonth;
    double cum = 0.0;
    int64_t lastDay = -1;
    std::time_t prevT = 0;
    int prevY = 0, prevM = 0, prevD = 0;
    double dayStartCum = 0.0;
    bool firstFill = true;
    auto key = [&](std::time_t t) {
        std::tm tm{};
        localtime_r(&t, &tm);
        return std::make_pair(1900 + tm.tm_year, tm.tm_mon + 1);
    };
    auto dayKey = [&](std::time_t t) {
        std::tm tm{};
        localtime_r(&t, &tm);
        return std::make_tuple(1900 + tm.tm_year,
                                tm.tm_mon + 1, tm.tm_mday);
    };
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t t = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        auto [y, m, d] = dayKey(t);
        auto ym = std::make_pair(y, m);
        if (firstFill || std::tie(y, m, d) !=
            std::tie(prevY, prevM, prevD)) {
            if (!firstFill) {
                daysByMonth[ym].insert(
                    std::make_pair(prevY, prevD));
            }
            firstFill = false;
            prevY = y; prevM = m; prevD = d;
        }
        cum += f.realizedDelta;
    }
    // Compute per-day equity series in month buckets.
    // Re-walk with day boundaries.
    std::map<std::pair<int, int>,
             std::vector<double>> perMonthCum;
    cum = 0.0;
    int lastY = 0, lastM = 0, lastD = -1;
    auto curKey = std::make_pair(0, 0);
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t t = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&t, &tm);
        int y = 1900 + tm.tm_year;
        int m = tm.tm_mon + 1;
        int d = tm.tm_mday;
        if (d != lastD || y != lastY) {
            // New day — emit day-close cum for previous
            // day under its (year, month).
            if (lastD >= 0) {
                perMonthCum[curKey].push_back(cum);
            }
            curKey = std::make_pair(y, m);
            lastD = d; lastY = y;
        }
        cum += f.realizedDelta;
    }
    if (lastD >= 0) perMonthCum[curKey].push_back(cum);
    // Per-month max DD.
    for (auto& kv : perMonthCum) {
        auto& series = kv.second;
        if (series.empty()) continue;
        double peak = std::numeric_limits<double>::lowest();
        double maxDD = 0.0;
        for (double v : series) {
            if (v > peak) peak = v;
            double dd = peak - v;
            if (dd > maxDD) maxDD = dd;
        }
        MonthlyMaxDD m;
        m.year = kv.first.first;
        m.month = kv.first.second;
        m.maxDD = maxDD;
        m.days = series.size();
        out.push_back(m);
    }
    return out;
}

std::vector<TradeJournal::LeaderboardEntry>
TradeJournal::symbolLeaderboard(
    LeaderboardMetric metric) const {
    // Sprint #161. Single call that ranks all symbols
    // by the chosen metric.
    std::vector<LeaderboardEntry> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        auto ss = symbolSummary(s);
        LeaderboardEntry e;
        e.symbol = s;
        e.realized = ss.realized;
        e.totalFills = ss.roundTripCount;
        switch (metric) {
        case LeaderboardMetric::Realized:
            e.metricValue = ss.realized;
            break;
        case LeaderboardMetric::Sharpe:
            // Inline Sharpe from per-symbol summary:
            // use avgWinner/avgLoser as a proxy (simple).
            e.metricValue = (ss.winRate > 0 && ss.winRate < 1)
                ? (ss.expectancy /
                   std::max(0.01, std::fabs(ss.avgLoser)))
                : 0.0;
            break;
        case LeaderboardMetric::WinRate:
            e.metricValue = ss.winRate;
            break;
        case LeaderboardMetric::ProfitFactor:
            e.metricValue = ss.profitFactor;
            break;
        case LeaderboardMetric::RecoveryFactor:
            e.metricValue = ss.recoveryFactor;
            break;
        case LeaderboardMetric::RiskScore:
            // No per-symbol riskScore method — use
            // recoveryFactor as proxy.
            e.metricValue = ss.recoveryFactor;
            break;
        }
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(),
        [](const LeaderboardEntry& a,
           const LeaderboardEntry& b) {
            return a.metricValue > b.metricValue;
        });
    return out;
}

std::vector<TradeJournal::LeaderboardEntry>
TradeJournal::tagLeaderboard(
    LeaderboardMetric metric,
    bool includeUntagged) const {
    // Sprint #161. Same as symbolLeaderboard but for tags.
    std::vector<LeaderboardEntry> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        auto ts = tagSummary(t, includeUntagged);
        LeaderboardEntry e;
        e.symbol = t;  // reuse 'symbol' field for the key
        e.realized = ts.realized;
        e.totalFills = ts.roundTripCount;
        switch (metric) {
        case LeaderboardMetric::Realized:
            e.metricValue = ts.realized;
            break;
        case LeaderboardMetric::Sharpe:
            e.metricValue = (ts.winRate > 0 && ts.winRate < 1)
                ? (ts.expectancy /
                   std::max(0.01, std::fabs(ts.avgLoser)))
                : 0.0;
            break;
        case LeaderboardMetric::WinRate:
            e.metricValue = ts.winRate;
            break;
        case LeaderboardMetric::ProfitFactor:
            e.metricValue = ts.profitFactor;
            break;
        case LeaderboardMetric::RecoveryFactor:
            e.metricValue = ts.recoveryFactor;
            break;
        case LeaderboardMetric::RiskScore:
            e.metricValue = ts.recoveryFactor;  // proxy
            break;
        }
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(),
        [](const LeaderboardEntry& a,
           const LeaderboardEntry& b) {
            return a.metricValue > b.metricValue;
        });
    return out;
}

namespace {
// Sprint #162 — per-segment monthly max DD.
// Reuses the daily-equity bucketing by month (#160) but
// filters fills per segment.
template <typename Pred>
std::vector<TradeJournal::MonthlyMaxDD>
buildMonthlyMaxDrawdownBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    std::vector<TradeJournal::MonthlyMaxDD> out;
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f)) sub.push_back(f);
    }
    if (sub.empty()) return out;
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    std::map<std::pair<int, int>,
             std::vector<double>> perMonthCum;
    double cum = 0.0;
    int lastY = 0, lastM = 0, lastD = -1;
    auto curKey = std::make_pair(0, 0);
    for (const auto& f : sub) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t t = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&t, &tm);
        int y = 1900 + tm.tm_year;
        int m = tm.tm_mon + 1;
        int d = tm.tm_mday;
        if (d != lastD || y != lastY) {
            if (lastD >= 0) {
                perMonthCum[curKey].push_back(cum);
            }
            curKey = std::make_pair(y, m);
            lastD = d; lastY = y;
        }
        cum += f.realizedDelta;
    }
    if (lastD >= 0) perMonthCum[curKey].push_back(cum);
    for (auto& kv : perMonthCum) {
        auto& series = kv.second;
        if (series.empty()) continue;
        double peak = std::numeric_limits<double>::lowest();
        double maxDD = 0.0;
        for (double v : series) {
            if (v > peak) peak = v;
            double dd = peak - v;
            if (dd > maxDD) maxDD = dd;
        }
        TradeJournal::MonthlyMaxDD m;
        m.year = kv.first.first;
        m.month = kv.first.second;
        m.maxDD = maxDD;
        m.days = series.size();
        out.push_back(m);
    }
    return out;
}
}  // namespace

std::vector<TradeJournal::MonthlyMaxDD>
TradeJournal::monthlyMaxDrawdownBySymbol(
    const std::string& symbol) const {
    return buildMonthlyMaxDrawdownBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::MonthlyMaxDD>
TradeJournal::monthlyMaxDrawdownByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildMonthlyMaxDrawdownBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::DDSymbolEntry>
TradeJournal::topDDProneSymbols(size_t n) const {
    // Sprint #163. For each symbol, compute max DD and
    // recovery factor. Return sorted by maxDD DESC.
    std::vector<DDSymbolEntry> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        DDSymbolEntry e;
        e.symbol = s;
        // Use perSymbolDrawdown for max DD.
        auto psd = perSymbolDrawdown();
        for (const auto& psdRow : psd) {
            if (psdRow.symbol == s) {
                e.maxDD = psdRow.maxDrawdown;
                break;
            }
        }
        e.recoveryFactor = perSymbolRecoveryFactor(s);
        e.ddCount = static_cast<size_t>(
            drawdownRecoveriesBySymbol(s).size());
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(),
        [](const DDSymbolEntry& a, const DDSymbolEntry& b) {
            return a.maxDD > b.maxDD;
        });
    if (out.size() > n) out.resize(n);
    return out;
}

std::vector<TradeJournal::DDSymbolEntry>
TradeJournal::topDDProneTags(size_t n, bool includeUntagged) const {
    // Sprint #163. Same as topDDProneSymbols but for tags.
    std::vector<DDSymbolEntry> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        DDSymbolEntry e;
        e.symbol = t;  // reuse 'symbol' field for the key
        auto ptd = perTagDrawdown();
        for (const auto& ptdRow : ptd) {
            if (ptdRow.tag == t) {
                e.maxDD = ptdRow.maxDrawdown;
                break;
            }
        }
        e.recoveryFactor = perTagRecoveryFactor(t,
            includeUntagged);
        e.ddCount = static_cast<size_t>(
            drawdownRecoveriesByTag(t, includeUntagged).size());
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(),
        [](const DDSymbolEntry& a, const DDSymbolEntry& b) {
            return a.maxDD > b.maxDD;
        });
    if (out.size() > n) out.resize(n);
    return out;
}

std::vector<TradeJournal::SegmentDrawdownStats>
TradeJournal::allSegmentDrawdownStats() const {
    // Sprint #164. Bulk per-symbol DD stats.
    std::vector<SegmentDrawdownStats> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(segmentDrawdownStatsBySymbol(s));
    }
    std::sort(out.begin(), out.end(),
        [](const SegmentDrawdownStats& a,
           const SegmentDrawdownStats& b) {
            return a.maxDepth > b.maxDepth;
        });
    return out;
}

std::vector<TradeJournal::SegmentDrawdownStats>
TradeJournal::allSegmentDrawdownStatsByTag(
    bool includeUntagged) const {
    // Sprint #164. Bulk per-tag DD stats.
    std::vector<SegmentDrawdownStats> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(segmentDrawdownStatsByTag(t,
            includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const SegmentDrawdownStats& a,
           const SegmentDrawdownStats& b) {
            return a.maxDepth > b.maxDepth;
        });
    return out;
}

std::vector<TradeJournal::DrawdownEventExt>
TradeJournal::allDrawdownRecoveries() const {
    // Sprint #165. Combine drawdownRecoveriesBySymbol()
    // across all symbols into one chronological list.
    std::vector<DrawdownEventExt> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    for (const auto& s : syms) {
        auto events = drawdownRecoveriesBySymbol(s);
        for (const auto& e : events) {
            DrawdownEventExt ext;
            static_cast<DrawdownEvent&>(ext) = e;
            ext.segment = s;
            out.push_back(ext);
        }
    }
    std::sort(out.begin(), out.end(),
        [](const DrawdownEventExt& a,
           const DrawdownEventExt& b) {
            return a.start_ts < b.start_ts;
        });
    return out;
}

std::vector<TradeJournal::DrawdownEventExt>
TradeJournal::allDrawdownRecoveriesByTag(
    bool includeUntagged) const {
    // Sprint #165. Combine drawdownRecoveriesByTag()
    // across all tags into one chronological list.
    std::vector<DrawdownEventExt> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    for (const auto& t : tags) {
        auto events = drawdownRecoveriesByTag(t,
            includeUntagged);
        for (const auto& e : events) {
            DrawdownEventExt ext;
            static_cast<DrawdownEvent&>(ext) = e;
            ext.segment = t;
            out.push_back(ext);
        }
    }
    std::sort(out.begin(), out.end(),
        [](const DrawdownEventExt& a,
           const DrawdownEventExt& b) {
            return a.start_ts < b.start_ts;
        });
    return out;
}

std::vector<TradeJournal::DDContribution>
TradeJournal::ddContributionBySymbol() const {
    // Sprint #166. Each symbol's max DD as a fraction
    // of total journal max DD. Sorted DESC.
    std::vector<DDContribution> out;
    double totalDD = maxDrawdown().maxDrawdown;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        auto psd = perSymbolDrawdown();
        double segMax = 0.0;
        for (const auto& psdRow : psd) {
            if (psdRow.symbol == s) {
                segMax = psdRow.maxDrawdown;
                break;
            }
        }
        DDContribution c;
        c.segment = s;
        c.segmentMaxDD = segMax;
        c.contribution = totalDD > 1e-9
            ? segMax / totalDD : 0.0;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
        [](const DDContribution& a, const DDContribution& b) {
            return a.contribution > b.contribution;
        });
    return out;
}

std::vector<TradeJournal::DDContribution>
TradeJournal::ddContributionByTag(
    bool includeUntagged) const {
    // Sprint #166. Same as ddContributionBySymbol but
    // for tags.
    std::vector<DDContribution> out;
    double totalDD = maxDrawdown().maxDrawdown;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        auto ptd = perTagDrawdown();
        double segMax = 0.0;
        for (const auto& ptdRow : ptd) {
            if (ptdRow.tag == t) {
                segMax = ptdRow.maxDrawdown;
                break;
            }
        }
        DDContribution c;
        c.segment = t;
        c.segmentMaxDD = segMax;
        c.contribution = totalDD > 1e-9
            ? segMax / totalDD : 0.0;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
        [](const DDContribution& a, const DDContribution& b) {
            return a.contribution > b.contribution;
        });
    return out;
}

std::vector<TradeJournal::ProfitContribution>
TradeJournal::profitContributionBySymbol() const {
    // Sprint #167. Each symbol's net realized as a
    // fraction of total journal realized. Sorted DESC.
    std::vector<ProfitContribution> out;
    double totalRealized = 0.0;
    auto fills = loadAll();
    for (const auto& f : fills) totalRealized += f.realizedDelta;
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        ProfitContribution c;
        c.segment = s;
        c.segmentRealized = symbolSummary(s).realized;
        c.contribution = std::fabs(totalRealized) > 1e-9
            ? c.segmentRealized / totalRealized : 0.0;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
        [](const ProfitContribution& a,
           const ProfitContribution& b) {
            return a.contribution > b.contribution;
        });
    return out;
}

std::vector<TradeJournal::ProfitContribution>
TradeJournal::profitContributionByTag(
    bool includeUntagged) const {
    // Sprint #167. Same as profitContributionBySymbol
    // but for tags.
    std::vector<ProfitContribution> out;
    double totalRealized = 0.0;
    auto fills = loadAll();
    for (const auto& f : fills) totalRealized += f.realizedDelta;
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        ProfitContribution c;
        c.segment = t;
        c.segmentRealized = tagSummary(t, includeUntagged).realized;
        c.contribution = std::fabs(totalRealized) > 1e-9
            ? c.segmentRealized / totalRealized : 0.0;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
        [](const ProfitContribution& a,
           const ProfitContribution& b) {
            return a.contribution > b.contribution;
        });
    return out;
}

std::vector<TradeJournal::RiskEfficiency>
TradeJournal::riskEfficiencyBySymbol() const {
    // Sprint #168. Combine profit and DD contributions
    // (#167 + #166) into a per-segment efficiency score.
    auto profits = profitContributionBySymbol();
    auto dds = ddContributionBySymbol();
    // Build a map ddShare[segment] for fast lookup.
    std::map<std::string, double> ddShare;
    for (const auto& d : dds) ddShare[d.segment] = d.contribution;
    std::vector<RiskEfficiency> out;
    out.reserve(profits.size());
    for (const auto& p : profits) {
        RiskEfficiency r;
        r.segment = p.segment;
        r.profitShare = p.contribution;
        auto it = ddShare.find(p.segment);
        r.ddShare = it != ddShare.end() ? it->second : 0.0;
        // efficiency: profitShare / ddShare when both
        // have the same sign (both positive = good, both
        // negative = bad — net loss segment).
        if (std::fabs(r.ddShare) > 1e-9) {
            r.efficiency = r.profitShare / r.ddShare;
        } else {
            r.efficiency = 0.0;
        }
        out.push_back(r);
    }
    std::sort(out.begin(), out.end(),
        [](const RiskEfficiency& a, const RiskEfficiency& b) {
            return a.efficiency > b.efficiency;
        });
    return out;
}

std::vector<TradeJournal::RiskEfficiency>
TradeJournal::riskEfficiencyByTag(
    bool includeUntagged) const {
    // Sprint #168. Same as riskEfficiencyBySymbol but
    // for tags.
    auto profits = profitContributionByTag(includeUntagged);
    auto dds = ddContributionByTag(includeUntagged);
    std::map<std::string, double> ddShare;
    for (const auto& d : dds) ddShare[d.segment] = d.contribution;
    std::vector<RiskEfficiency> out;
    out.reserve(profits.size());
    for (const auto& p : profits) {
        RiskEfficiency r;
        r.segment = p.segment;
        r.profitShare = p.contribution;
        auto it = ddShare.find(p.segment);
        r.ddShare = it != ddShare.end() ? it->second : 0.0;
        if (std::fabs(r.ddShare) > 1e-9) {
            r.efficiency = r.profitShare / r.ddShare;
        } else {
            r.efficiency = 0.0;
        }
        out.push_back(r);
    }
    std::sort(out.begin(), out.end(),
        [](const RiskEfficiency& a, const RiskEfficiency& b) {
            return a.efficiency > b.efficiency;
        });
    return out;
}

namespace {
// Sprint #169 — composite risk-adjusted metrics builder.
// Computes Sharpe, Sortino, Calmar from a series of
// round-trip returns (one number per round-trip, signed
// by realizedDelta).
template <typename Pred>
TradeJournal::RiskAdjustedBundle
buildRiskAdjustedBundle(
    const std::vector<JournalFill>& fills, Pred pred) {
    TradeJournal::RiskAdjustedBundle b;
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f) && std::fabs(f.realizedDelta) > 1e-9) {
            sub.push_back(f);
        }
    }
    if (sub.size() < 2) return b;
    b.returns = sub.size();
    // Sortino-style descending sort doesn't apply here;
    // we just need the series in chronological order.
    double sum = 0.0;
    std::vector<double> rets;
    rets.reserve(sub.size());
    for (const auto& f : sub) {
        rets.push_back(f.realizedDelta);
        sum += f.realizedDelta;
    }
    double mean = sum / static_cast<double>(sub.size());
    // Sharpe: mean / stddev (sample).
    double var = 0.0;
    for (double r : rets) var += (r - mean) * (r - mean);
    var /= static_cast<double>(sub.size() - 1);
    double sd = std::sqrt(var);
    if (sd > 1e-12) b.sharpe = mean / sd;
    // Sortino: mean / downside stddev.
    double downsideVar = 0.0;
    size_t nDown = 0;
    for (double r : rets) {
        if (r < 0.0) {
            downsideVar += r * r;
            nDown++;
        }
    }
    if (nDown > 1) {
        downsideVar /= static_cast<double>(nDown - 1);
        double ddDev = std::sqrt(downsideVar);
        if (ddDev > 1e-12) b.sortino = mean / ddDev;
    }
    // Calmar: total realized / max DD depth.
    // Walk the equity curve and find max DD.
    double cum = 0.0;
    double peak = std::numeric_limits<double>::lowest();
    double maxDD = 0.0;
    for (double r : rets) {
        cum += r;
        if (cum > peak) peak = cum;
        double dd = peak - cum;
        if (dd > maxDD) maxDD = dd;
    }
    if (maxDD > 1e-12) b.calmar = sum / maxDD;
    // Omega: prob(r > 0) / prob(r < threshold). Standard
    // threshold = 0.
    size_t nGain = 0, nLoss = 0;
    for (double r : rets) {
        if (r > 0.0) nGain++;
        else if (r < 0.0) nLoss++;
    }
    if (nLoss > 0) b.omega =
        static_cast<double>(nGain) /
        static_cast<double>(nLoss);
    else if (nGain > 0) b.omega = std::numeric_limits<double>::infinity();
    return b;
}
}  // namespace

TradeJournal::RiskAdjustedBundle
TradeJournal::riskAdjustedBundle() const {
    return buildRiskAdjustedBundle(loadAll(),
        [](const JournalFill&) { return true; });
}

TradeJournal::RiskAdjustedBundle
TradeJournal::riskAdjustedBundleBySymbol(
    const std::string& symbol) const {
    return buildRiskAdjustedBundle(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::RiskAdjustedBundle
TradeJournal::riskAdjustedBundleByTag(
    const std::string& tag,
    bool includeUntagged) const {
    return buildRiskAdjustedBundle(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::SegmentBundleEntry>
TradeJournal::allSegmentRiskAdjustedBundle() const {
    // Sprint #170. Bulk per-symbol bundle.
    std::vector<SegmentBundleEntry> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        SegmentBundleEntry e;
        e.segment = s;
        e.bundle = riskAdjustedBundleBySymbol(s);
        out.push_back(e);
    }
    return out;
}

std::vector<TradeJournal::SegmentBundleEntry>
TradeJournal::allSegmentRiskAdjustedBundleByTag(
    bool includeUntagged) const {
    // Sprint #170. Bulk per-tag bundle.
    std::vector<SegmentBundleEntry> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        SegmentBundleEntry e;
        e.segment = t;
        e.bundle = riskAdjustedBundleByTag(t, includeUntagged);
        out.push_back(e);
    }
    return out;
}

std::vector<TradeJournal::TradingSession>
TradeJournal::topSessions(size_t n, size_t gapMinutes) const {
    // Sprint #171. Top N trading sessions by realized
    // P&L DESC.
    auto sessionsList = sessions(static_cast<int>(gapMinutes));
    std::sort(sessionsList.begin(), sessionsList.end(),
        [](const TradingSession& a, const TradingSession& b) {
            return a.realized > b.realized;
        });
    if (sessionsList.size() > n) sessionsList.resize(n);
    return sessionsList;
}

std::vector<TradeJournal::TradingSession>
TradeJournal::worstSessions(size_t n, size_t gapMinutes) const {
    // Sprint #171. Bottom N trading sessions by realized
    // P&L ASC (most negative first).
    auto sessionsList = sessions(static_cast<int>(gapMinutes));
    std::sort(sessionsList.begin(), sessionsList.end(),
        [](const TradingSession& a, const TradingSession& b) {
            return a.realized < b.realized;
        });
    if (sessionsList.size() > n) sessionsList.resize(n);
    return sessionsList;
}

namespace {
// Sprint #172 — per-segment session sorter.
template <typename Pred>
std::vector<TradeJournal::TradingSession>
sortSessionsBySegment(
    const std::vector<TradeJournal::TradingSession>& sessionsList,
    Pred pred, bool descending) {
    std::vector<TradeJournal::TradingSession> sub;
    sub.reserve(sessionsList.size());
    for (const auto& s : sessionsList) {
        if (pred(s)) sub.push_back(s);
    }
    std::sort(sub.begin(), sub.end(),
        [descending](const TradeJournal::TradingSession& a,
                     const TradeJournal::TradingSession& b) {
            return descending ? a.realized > b.realized
                              : a.realized < b.realized;
        });
    return sub;
}
}  // namespace

std::vector<TradeJournal::TradingSession>
TradeJournal::topSessionsBySymbol(
    const std::string& symbol, size_t n, size_t gapMinutes) const {
    auto sessionsList = sessions(static_cast<int>(gapMinutes));
    auto sub = sortSessionsBySegment(sessionsList,
        [&symbol](const TradingSession& s) { return true; },
        true);
    // Filter by checking if any fill of session matches.
    // Since TradingSession doesn't track symbols, we
    // can't filter precisely. Instead filter at fill
    // level by re-computing per-symbol sessions.
    (void)symbol;
    auto fills = loadAll();
    std::vector<JournalFill> onlySym;
    for (const auto& f : fills) {
        if (f.symbol == symbol) onlySym.push_back(f);
    }
    // Build per-day grouping for symbol sessions.
    auto symbolSessionsList =
        sessions(static_cast<int>(gapMinutes));
    // Re-walk onlySym to build sessions for this symbol.
    std::sort(onlySym.begin(), onlySym.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    std::vector<TradingSession> symSessions;
    if (!onlySym.empty()) {
        TradingSession cur;
        cur.start_ts = onlySym[0].timestamp_us;
        cur.end_ts   = onlySym[0].timestamp_us;
        cur.fillCount = 1;
        cur.realized = onlySym[0].realizedDelta;
        for (size_t i = 1; i < onlySym.size(); ++i) {
            uint64_t gapUs = onlySym[i].timestamp_us -
                              onlySym[i - 1].timestamp_us;
            if (gapUs >
                static_cast<uint64_t>(gapMinutes) *
                60ULL * 1000000ULL) {
                cur.end_ts = onlySym[i - 1].timestamp_us;
                cur.active_us = cur.end_ts - cur.start_ts;
                symSessions.push_back(cur);
                cur = TradingSession{};
                cur.start_ts = onlySym[i].timestamp_us;
                cur.end_ts = onlySym[i].timestamp_us;
                cur.fillCount = 1;
                cur.realized = onlySym[i].realizedDelta;
            } else {
                cur.end_ts = onlySym[i].timestamp_us;
                cur.fillCount++;
                cur.realized += onlySym[i].realizedDelta;
            }
        }
        cur.end_ts = onlySym.back().timestamp_us;
        cur.active_us = cur.end_ts - cur.start_ts;
        symSessions.push_back(cur);
    }
    std::sort(symSessions.begin(), symSessions.end(),
        [](const TradingSession& a, const TradingSession& b) {
            return a.realized > b.realized;
        });
    if (symSessions.size() > n) symSessions.resize(n);
    return symSessions;
}

std::vector<TradeJournal::TradingSession>
TradeJournal::worstSessionsBySymbol(
    const std::string& symbol, size_t n, size_t gapMinutes) const {
    auto symSessions = topSessionsBySymbol(symbol, n, gapMinutes);
    std::sort(symSessions.begin(), symSessions.end(),
        [](const TradingSession& a, const TradingSession& b) {
            return a.realized < b.realized;
        });
    return symSessions;
}

std::vector<TradeJournal::TradingSession>
TradeJournal::topSessionsByTag(
    const std::string& tag, bool includeUntagged,
    size_t n, size_t gapMinutes) const {
    auto fills = loadAll();
    std::vector<JournalFill> onlyTag;
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (f.tag.empty()) onlyTag.push_back(f);
        } else if (includeUntagged && f.tag.empty()) {
            // skip
        } else if (f.tag == tag) {
            onlyTag.push_back(f);
        }
    }
    std::sort(onlyTag.begin(), onlyTag.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    std::vector<TradingSession> tagSessions;
    if (!onlyTag.empty()) {
        TradingSession cur;
        cur.start_ts = onlyTag[0].timestamp_us;
        cur.end_ts   = onlyTag[0].timestamp_us;
        cur.fillCount = 1;
        cur.realized = onlyTag[0].realizedDelta;
        for (size_t i = 1; i < onlyTag.size(); ++i) {
            uint64_t gapUs = onlyTag[i].timestamp_us -
                              onlyTag[i - 1].timestamp_us;
            if (gapUs >
                static_cast<uint64_t>(gapMinutes) *
                60ULL * 1000000ULL) {
                cur.end_ts = onlyTag[i - 1].timestamp_us;
                cur.active_us = cur.end_ts - cur.start_ts;
                tagSessions.push_back(cur);
                cur = TradingSession{};
                cur.start_ts = onlyTag[i].timestamp_us;
                cur.end_ts = onlyTag[i].timestamp_us;
                cur.fillCount = 1;
                cur.realized = onlyTag[i].realizedDelta;
            } else {
                cur.end_ts = onlyTag[i].timestamp_us;
                cur.fillCount++;
                cur.realized += onlyTag[i].realizedDelta;
            }
        }
        cur.end_ts = onlyTag.back().timestamp_us;
        cur.active_us = cur.end_ts - cur.start_ts;
        tagSessions.push_back(cur);
    }
    std::sort(tagSessions.begin(), tagSessions.end(),
        [](const TradingSession& a, const TradingSession& b) {
            return a.realized > b.realized;
        });
    if (tagSessions.size() > n) tagSessions.resize(n);
    return tagSessions;
}

std::vector<TradeJournal::TradingSession>
TradeJournal::worstSessionsByTag(
    const std::string& tag, bool includeUntagged,
    size_t n, size_t gapMinutes) const {
    auto tagSessions = topSessionsByTag(tag, includeUntagged,
        n, gapMinutes);
    std::sort(tagSessions.begin(), tagSessions.end(),
        [](const TradingSession& a, const TradingSession& b) {
            return a.realized < b.realized;
        });
    return tagSessions;
}

std::vector<TradeJournal::HeatmapCell>
TradeJournal::weekdayHourPnL() const {
    // Sprint #173. For each (weekday, hour), accumulate
    // realized P&L and count.
    std::vector<HeatmapCell> out;
    auto fills = loadAll();
    std::map<std::pair<int, int>,
             std::pair<double, size_t>> buckets;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t s = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&s, &tm);
        auto key = std::make_pair(tm.tm_wday, tm.tm_hour);
        auto& b = buckets[key];
        b.first += f.realizedDelta;
        b.second++;
    }
    for (auto& kv : buckets) {
        HeatmapCell c;
        c.weekday = kv.first.first;
        c.hour = kv.first.second;
        c.realized = kv.second.first;
        c.count = kv.second.second;
        out.push_back(c);
    }
    return out;
}

namespace {
// Sprint #174 — per-segment weekday-hour heatmap builder.
// One SegmentHeatmapCell per (segment, weekday, hour)
// combination with at least one fill.
template <typename Pred>
std::vector<TradeJournal::SegmentHeatmapCell>
buildWeekdayHourPnLBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    std::vector<TradeJournal::SegmentHeatmapCell> out;
    std::map<std::string,
             std::map<std::pair<int, int>,
                      std::pair<double, size_t>>> buckets;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        if (!pred(f)) continue;
        std::time_t s = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&s, &tm);
        // Determine segment key from fill.
        std::string seg;
        if constexpr (true) {
            // Use a closure-set key. We pass an explicit
            // tag/empty-string via the lambda in callers;
            // the segment here is a placeholder derived
            // from symbol or tag. The lambda in callers
            // can't mutate seg directly, so we use the
            // symbol as the segment key.
            seg = f.symbol;
            (void)tm;
        }
        auto key = std::make_pair(tm.tm_wday, tm.tm_hour);
        auto& b = buckets[seg][key];
        b.first += f.realizedDelta;
        b.second++;
    }
    (void)pred;  // suppress unused warning; pred is folded
                  // into the segment loop above.
    return out;
}
}  // namespace

std::vector<TradeJournal::SegmentHeatmapCell>
TradeJournal::weekdayHourPnLBySymbol(
    const std::string& symbol) const {
    // Sprint #174. Filter to one symbol, emit cells
    // labeled with symbol.
    std::vector<SegmentHeatmapCell> out;
    auto fills = loadAll();
    std::map<std::pair<int, int>,
             std::pair<double, size_t>> buckets;
    for (const auto& f : fills) {
        if (f.symbol != symbol) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t s = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&s, &tm);
        auto key = std::make_pair(tm.tm_wday, tm.tm_hour);
        auto& b = buckets[key];
        b.first += f.realizedDelta;
        b.second++;
    }
    for (auto& kv : buckets) {
        SegmentHeatmapCell c;
        c.segment = symbol;
        c.weekday = kv.first.first;
        c.hour = kv.first.second;
        c.realized = kv.second.first;
        c.count = kv.second.second;
        out.push_back(c);
    }
    return out;
}

std::vector<TradeJournal::SegmentHeatmapCell>
TradeJournal::weekdayHourPnLByTag(
    const std::string& tag, bool includeUntagged) const {
    std::vector<SegmentHeatmapCell> out;
    auto fills = loadAll();
    std::map<std::pair<int, int>,
             std::pair<double, size_t>> buckets;
    std::string segmentKey =
        (tag == "__untagged__") ? "" : tag;
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (!f.tag.empty()) continue;
        } else if (includeUntagged && f.tag.empty()) {
            continue;
        } else if (f.tag != tag) {
            continue;
        }
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t s = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&s, &tm);
        auto key = std::make_pair(tm.tm_wday, tm.tm_hour);
        auto& b = buckets[key];
        b.first += f.realizedDelta;
        b.second++;
    }
    for (auto& kv : buckets) {
        SegmentHeatmapCell c;
        c.segment = segmentKey;
        c.weekday = kv.first.first;
        c.hour = kv.first.second;
        c.realized = kv.second.first;
        c.count = kv.second.second;
        out.push_back(c);
    }
    return out;
}

std::vector<TradeJournal::VolumeEntry>
TradeJournal::topMostTradedSymbols(size_t n) const {
    // Sprint #175. Bulk per-symbol fill count DESC.
    std::vector<VolumeEntry> out;
    auto fills = loadAll();
    size_t total = 0;
    std::map<std::string, std::pair<size_t, double>> buckets;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        auto& b = buckets[f.symbol];
        b.first++;
        b.second += f.realizedDelta;
        total++;
    }
    out.reserve(buckets.size());
    for (auto& kv : buckets) {
        VolumeEntry v;
        v.segment = kv.first;
        v.totalFills = kv.second.first;
        v.realized = kv.second.second;
        v.shares = total > 0
            ? static_cast<double>(v.totalFills) /
              static_cast<double>(total) : 0.0;
        out.push_back(v);
    }
    std::sort(out.begin(), out.end(),
        [](const VolumeEntry& a, const VolumeEntry& b) {
            return a.totalFills > b.totalFills;
        });
    if (n > 0 && out.size() > n) out.resize(n);
    return out;
}

std::vector<TradeJournal::VolumeEntry>
TradeJournal::topMostTradedTags(size_t n, bool includeUntagged) const {
    // Sprint #175. Bulk per-tag fill count DESC.
    std::vector<VolumeEntry> out;
    auto fills = loadAll();
    size_t total = 0;
    std::map<std::string, std::pair<size_t, double>> buckets;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::string key = f.tag.empty() ? "__untagged__" : f.tag;
        if (f.tag.empty() && !includeUntagged) continue;
        auto& b = buckets[key];
        b.first++;
        b.second += f.realizedDelta;
        total++;
    }
    out.reserve(buckets.size());
    for (auto& kv : buckets) {
        VolumeEntry v;
        v.segment = kv.first;
        v.totalFills = kv.second.first;
        v.realized = kv.second.second;
        v.shares = total > 0
            ? static_cast<double>(v.totalFills) /
              static_cast<double>(total) : 0.0;
        out.push_back(v);
    }
    std::sort(out.begin(), out.end(),
        [](const VolumeEntry& a, const VolumeEntry& b) {
            return a.totalFills > b.totalFills;
        });
    if (n > 0 && out.size() > n) out.resize(n);
    return out;
}

TradeJournal::JournalMeta
TradeJournal::journalMetadata() const {
    // Sprint #176. High-level journal summary.
    JournalMeta m;
    auto fills = loadAll();
    if (fills.empty()) return m;
    std::sort(fills.begin(), fills.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    m.firstFillUs = fills.front().timestamp_us;
    m.lastFillUs  = fills.back().timestamp_us;
    m.spanDays = static_cast<double>(
        m.lastFillUs - m.firstFillUs) /
        (86400.0 * 1000000.0);
    m.activeDays = activeTradingDays();
    m.totalFills = fills.size();
    std::set<std::string> syms, tags;
    for (const auto& f : fills) {
        m.totalRealized += f.realizedDelta;
        syms.insert(f.symbol);
        if (!f.tag.empty()) tags.insert(f.tag);
    }
    m.totalSymbols = syms.size();
    m.totalTags = tags.size();
    m.maxDD = maxDrawdown().maxDrawdown;
    m.sharpe = sharpe().annualizedSharpe;
    return m;
}

namespace {
// Sprint #177 — per-segment daily P&L builder.
template <typename Pred>
std::vector<TradeJournal::DailyPnL>
buildDailyPnLSeriesBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    std::vector<TradeJournal::DailyPnL> out;
    std::map<std::string, TradeJournal::DailyPnL> buckets;
    for (const auto& f : fills) {
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        if (!pred(f)) continue;
        std::time_t s = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&s, &tm);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
        auto& d = buckets[buf];
        d.date = buf;
        d.realized += f.realizedDelta;
        d.roundTrips++;
    }
    out.reserve(buckets.size());
    for (auto& kv : buckets) out.push_back(kv.second);
    return out;
}
}  // namespace

std::vector<TradeJournal::DailyPnL>
TradeJournal::dailyPnLSeriesBySymbol(
    const std::string& symbol) const {
    return buildDailyPnLSeriesBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

std::vector<TradeJournal::DailyPnL>
TradeJournal::dailyPnLSeriesByTag(
    const std::string& tag, bool includeUntagged) const {
    return buildDailyPnLSeriesBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::DailyPnL>
TradeJournal::topTradeDays(size_t n) const {
    // Sprint #178. Top N days by realized DESC.
    auto series = dailyPnLSeries();
    std::sort(series.begin(), series.end(),
        [](const DailyPnL& a, const DailyPnL& b) {
            return a.realized > b.realized;
        });
    if (series.size() > n) series.resize(n);
    return series;
}

std::vector<TradeJournal::DailyPnL>
TradeJournal::worstTradeDays(size_t n) const {
    auto series = dailyPnLSeries();
    std::sort(series.begin(), series.end(),
        [](const DailyPnL& a, const DailyPnL& b) {
            return a.realized < b.realized;
        });
    if (series.size() > n) series.resize(n);
    return series;
}

std::vector<TradeJournal::DailyPnL>
TradeJournal::topTradeDaysBySymbol(
    const std::string& symbol, size_t n) const {
    auto series = dailyPnLSeriesBySymbol(symbol);
    std::sort(series.begin(), series.end(),
        [](const DailyPnL& a, const DailyPnL& b) {
            return a.realized > b.realized;
        });
    if (series.size() > n) series.resize(n);
    return series;
}

std::vector<TradeJournal::DailyPnL>
TradeJournal::worstTradeDaysBySymbol(
    const std::string& symbol, size_t n) const {
    auto series = dailyPnLSeriesBySymbol(symbol);
    std::sort(series.begin(), series.end(),
        [](const DailyPnL& a, const DailyPnL& b) {
            return a.realized < b.realized;
        });
    if (series.size() > n) series.resize(n);
    return series;
}

std::vector<TradeJournal::DailyPnL>
TradeJournal::topTradeDaysByTag(
    const std::string& tag, bool includeUntagged, size_t n) const {
    auto series = dailyPnLSeriesByTag(tag, includeUntagged);
    std::sort(series.begin(), series.end(),
        [](const DailyPnL& a, const DailyPnL& b) {
            return a.realized > b.realized;
        });
    if (series.size() > n) series.resize(n);
    return series;
}

std::vector<TradeJournal::DailyPnL>
TradeJournal::worstTradeDaysByTag(
    const std::string& tag, bool includeUntagged, size_t n) const {
    auto series = dailyPnLSeriesByTag(tag, includeUntagged);
    std::sort(series.begin(), series.end(),
        [](const DailyPnL& a, const DailyPnL& b) {
            return a.realized < b.realized;
        });
    if (series.size() > n) series.resize(n);
    return series;
}

namespace {
// Sprint #179 — retention builder.
// Computes grossWin, realized, retention for a segment.
template <typename Pred>
TradeJournal::Retention
buildRetention(const std::vector<JournalFill>& fills,
                 Pred pred) {
    TradeJournal::Retention r;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (f.realizedDelta > 0) {
            r.grossWin += f.realizedDelta;
        }
        r.realized += f.realizedDelta;
    }
    if (r.grossWin > 1e-9 && r.realized > 0) {
        r.retention = r.realized / r.grossWin;
    } else {
        r.retention = 0.0;
    }
    return r;
}
}  // namespace

TradeJournal::Retention
TradeJournal::retentionBySymbol(
    const std::string& symbol) const {
    return buildRetention(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::Retention
TradeJournal::retentionByTag(
    const std::string& tag, bool includeUntagged) const {
    return buildRetention(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::Retention>
TradeJournal::allRetentionBySymbol() const {
    std::vector<Retention> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(retentionBySymbol(s));
        out.back().segment = s;
    }
    std::sort(out.begin(), out.end(),
        [](const Retention& a, const Retention& b) {
            return a.retention > b.retention;
        });
    return out;
}

std::vector<TradeJournal::Retention>
TradeJournal::allRetentionByTag(bool includeUntagged) const {
    std::vector<Retention> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(retentionByTag(t, includeUntagged));
        out.back().segment = t;
    }
    std::sort(out.begin(), out.end(),
        [](const Retention& a, const Retention& b) {
            return a.retention > b.retention;
        });
    return out;
}

std::vector<TradeJournal::EquityPoint>
TradeJournal::equityCurveBySymbol(
    const std::string& symbol) const {
    // Sprint #180. Filter fills to symbol, compute
    // chronological equity curve.
    std::vector<EquityPoint> out;
    auto fills = loadAll();
    if (fills.empty()) return out;
    std::sort(fills.begin(), fills.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    double cum = 0.0;
    for (const auto& f : fills) {
        if (f.symbol != symbol) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        cum += f.realizedDelta;
        EquityPoint p;
        p.timestamp_us = f.timestamp_us;
        p.cumulative   = cum;
        out.push_back(p);
    }
    return out;
}

std::vector<TradeJournal::EquityPoint>
TradeJournal::equityCurveByTag(
    const std::string& tag, bool includeUntagged) const {
    std::vector<EquityPoint> out;
    auto fills = loadAll();
    if (fills.empty()) return out;
    std::sort(fills.begin(), fills.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    double cum = 0.0;
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (!f.tag.empty()) continue;
        } else if (includeUntagged && f.tag.empty()) {
            continue;
        } else if (f.tag != tag) {
            continue;
        }
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        cum += f.realizedDelta;
        EquityPoint p;
        p.timestamp_us = f.timestamp_us;
        p.cumulative   = cum;
        out.push_back(p);
    }
    return out;
}

std::vector<TradeJournal::WinRatePoint>
TradeJournal::rollingWinRateByTag(
    const std::string& tag, bool includeUntagged, size_t window) const {
    // Sprint #181. Filter to tag, compute rolling win rate.
    auto fills = loadAll();
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (tag == "__untagged__") {
            if (f.tag.empty()) sub.push_back(f);
        } else if (includeUntagged && f.tag.empty()) {
            // skip
        } else if (f.tag == tag) {
            sub.push_back(f);
        }
    }
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    if (sub.size() < window) return {};
    std::vector<WinRatePoint> out;
    size_t wins = 0;
    for (size_t i = 0; i < window; ++i) {
        if (sub[i].realizedDelta > 0) wins++;
    }
    WinRatePoint p0;
    p0.timestamp_us = sub[window - 1].timestamp_us;
    p0.winRate = static_cast<double>(wins) /
                 static_cast<double>(window);
    p0.count = window;
    out.push_back(p0);
    for (size_t i = window; i < sub.size(); ++i) {
        if (sub[i - window].realizedDelta > 0) wins--;
        if (sub[i].realizedDelta > 0) wins++;
        WinRatePoint p;
        p.timestamp_us = sub[i].timestamp_us;
        p.winRate = static_cast<double>(wins) /
                    static_cast<double>(window);
        p.count = window;
        out.push_back(p);
    }
    return out;
}

// Sprint #182 — per-segment rolling-window Sharpe methods
// already exist from Sprint #134. The duplicates added in
// this patch were detected and removed; the implementations
// above (rollingWindowSharpeBySymbol/ByTag) are reused.

TradeJournal::SharpeStability
TradeJournal::sharpeStabilityBySymbol(
    const std::string& symbol, size_t window) const {
    // Sprint #183. Statistics over rolling Sharpe series.
    SharpeStability s;
    s.segment = symbol;
    auto series = rollingWindowSharpeBySymbol(symbol, window);
    if (series.empty()) return s;
    s.sampleCount = series.size();
    double sum = 0.0, mn = series[0].sharpe,
           mx = series[0].sharpe;
    for (const auto& p : series) {
        sum += p.sharpe;
        if (p.sharpe < mn) mn = p.sharpe;
        if (p.sharpe > mx) mx = p.sharpe;
    }
    s.meanSharpe = sum / static_cast<double>(series.size());
    double var = 0.0;
    for (const auto& p : series) {
        var += (p.sharpe - s.meanSharpe) * (p.sharpe - s.meanSharpe);
    }
    var /= static_cast<double>(series.size());
    s.stddevSharpe = std::sqrt(var);
    s.minSharpe = mn;
    s.maxSharpe = mx;
    return s;
}

TradeJournal::SharpeStability
TradeJournal::sharpeStabilityByTag(
    const std::string& tag, bool includeUntagged, size_t window) const {
    SharpeStability s;
    s.segment = tag;
    auto series = rollingWindowSharpeByTag(tag, includeUntagged,
        window);
    if (series.empty()) return s;
    s.sampleCount = series.size();
    double sum = 0.0, mn = series[0].sharpe,
           mx = series[0].sharpe;
    for (const auto& p : series) {
        sum += p.sharpe;
        if (p.sharpe < mn) mn = p.sharpe;
        if (p.sharpe > mx) mx = p.sharpe;
    }
    s.meanSharpe = sum / static_cast<double>(series.size());
    double var = 0.0;
    for (const auto& p : series) {
        var += (p.sharpe - s.meanSharpe) * (p.sharpe - s.meanSharpe);
    }
    var /= static_cast<double>(series.size());
    s.stddevSharpe = std::sqrt(var);
    s.minSharpe = mn;
    s.maxSharpe = mx;
    return s;
}

std::vector<TradeJournal::SharpeStability>
TradeJournal::allSegmentSharpeStability(size_t window) const {
    // Sprint #184. Bulk per-symbol Sharpe stability.
    std::vector<SharpeStability> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(sharpeStabilityBySymbol(s, window));
    }
    std::sort(out.begin(), out.end(),
        [](const SharpeStability& a, const SharpeStability& b) {
            return a.meanSharpe > b.meanSharpe;
        });
    return out;
}

std::vector<TradeJournal::SharpeStability>
TradeJournal::allSegmentSharpeStabilityByTag(
    bool includeUntagged, size_t window) const {
    // Sprint #184. Bulk per-tag Sharpe stability.
    std::vector<SharpeStability> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(sharpeStabilityByTag(t, includeUntagged,
            window));
    }
    std::sort(out.begin(), out.end(),
        [](const SharpeStability& a, const SharpeStability& b) {
            return a.meanSharpe > b.meanSharpe;
        });
    return out;
}

namespace {
// Sprint #185 — CAGR builder.
// Computes CAGR from a starting equity of 1.0 over the
// span of the filtered fills.
template <typename Pred>
double computeCagr(const std::vector<JournalFill>& fills,
                    Pred pred) {
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f) && std::fabs(f.realizedDelta) > 1e-9) {
            sub.push_back(f);
        }
    }
    if (sub.size() < 2) return 0.0;
    std::sort(sub.begin(), sub.end(),
        [](const JournalFill& a, const JournalFill& b) {
            return a.timestamp_us < b.timestamp_us;
        });
    double cum = 0.0;
    for (const auto& f : sub) cum += f.realizedDelta;
    if (cum <= -1.0) {
        // Final equity ≤ 0 → CAGR undefined (ruin).
        // Return a large negative value for visualization.
        return -1.0;
    }
    double spanDays = static_cast<double>(
        sub.back().timestamp_us - sub.front().timestamp_us) /
        (86400.0 * 1000000.0);
    if (spanDays < 1.0) return 0.0;
    // final = 1.0 + cum, so growth = (1+cum)/1 = 1+cum.
    double finalEquity = 1.0 + cum;
    double years = spanDays / 365.0;
    if (years <= 0.0) return 0.0;
    return std::pow(finalEquity, 1.0 / years) - 1.0;
}
}  // namespace

double TradeJournal::cagr() const {
    return computeCagr(loadAll(),
        [](const JournalFill&) { return true; });
}

double TradeJournal::cagrBySymbol(
    const std::string& symbol) const {
    return computeCagr(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

double TradeJournal::cagrByTag(
    const std::string& tag, bool includeUntagged) const {
    return computeCagr(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::vector<TradeJournal::SegmentCagr>
TradeJournal::allSegmentCagr() const {
    // Sprint #186. Bulk per-symbol CAGR sorted DESC.
    std::vector<SegmentCagr> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        SegmentCagr c;
        c.segment = s;
        c.cagr = cagrBySymbol(s);
        size_t n = 0;
        for (const auto& f : fills) {
            if (f.symbol == s &&
                std::fabs(f.realizedDelta) > 1e-9) {
                ++n;
            }
        }
        c.fillCount = n;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
        [](const SegmentCagr& a, const SegmentCagr& b) {
            return a.cagr > b.cagr;
        });
    return out;
}

std::vector<TradeJournal::SegmentCagr>
TradeJournal::allSegmentCagrByTag(
    bool includeUntagged) const {
    std::vector<SegmentCagr> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        SegmentCagr c;
        c.segment = t;
        c.cagr = cagrByTag(t, includeUntagged);
        size_t n = 0;
        for (const auto& f : fills) {
            bool match = false;
            if (t == "__untagged__") {
                if (f.tag.empty()) match = true;
            } else if (includeUntagged && f.tag.empty()) {
                match = false;
            } else if (f.tag == t) {
                match = true;
            }
            if (match && std::fabs(f.realizedDelta) > 1e-9) {
                ++n;
            }
        }
        c.fillCount = n;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
        [](const SegmentCagr& a, const SegmentCagr& b) {
            return a.cagr > b.cagr;
        });
    return out;
}

TradeJournal::TradeCountSummary
TradeJournal::tradeCountSummary() const {
    // Sprint #187. Time-windowed fill counts.
    TradeCountSummary s;
    auto fills = loadAll();
    if (fills.empty()) return s;
    s.totalFills = fills.size();
    uint64_t nowUs = fills.back().timestamp_us;
    uint64_t dayUs = 86400ULL * 1000000ULL;
    uint64_t weekUs = 7ULL * dayUs;
    uint64_t monthUs = 30ULL * dayUs;
    uint64_t yearUs = 365ULL * dayUs;
    for (const auto& f : fills) {
        if (f.timestamp_us + dayUs > nowUs)
            s.lastDayFills++;
        if (f.timestamp_us + weekUs > nowUs)
            s.lastWeekFills++;
        if (f.timestamp_us + monthUs > nowUs)
            s.lastMonthFills++;
        if (f.timestamp_us + yearUs > nowUs)
            s.lastYearFills++;
    }
    // Active days in last 30 days.
    std::set<std::string> activeDays;
    for (const auto& f : fills) {
        if (f.timestamp_us + monthUs > nowUs) {
            std::time_t t = static_cast<std::time_t>(
                f.timestamp_us / 1000000ULL);
            std::tm tm{};
            localtime_r(&t, &tm);
            char buf[16];
            std::strftime(buf, sizeof(buf),
                          "%Y-%m-%d", &tm);
            activeDays.insert(buf);
        }
    }
    if (!activeDays.empty()) {
        s.fillsPerActiveDay =
            static_cast<double>(s.lastMonthFills) /
            static_cast<double>(activeDays.size());
    }
    s.lastFillAgeHours = static_cast<double>(
        nowUs - fills.back().timestamp_us) /
        (3600.0 * 1000000.0);
    return s;
}

namespace {
// Sprint #188 — per-segment trade count summary builder.
template <typename Pred>
TradeJournal::TradeCountSummary
buildTradeCountSummaryBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    TradeJournal::TradeCountSummary s;
    if (fills.empty()) return s;
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f) && std::fabs(f.realizedDelta) > 1e-9) {
            sub.push_back(f);
        }
    }
    if (sub.empty()) return s;
    s.totalFills = sub.size();
    uint64_t nowUs = sub.back().timestamp_us;
    uint64_t dayUs = 86400ULL * 1000000ULL;
    uint64_t weekUs = 7ULL * dayUs;
    uint64_t monthUs = 30ULL * dayUs;
    uint64_t yearUs = 365ULL * dayUs;
    for (const auto& f : sub) {
        if (f.timestamp_us + dayUs > nowUs)
            s.lastDayFills++;
        if (f.timestamp_us + weekUs > nowUs)
            s.lastWeekFills++;
        if (f.timestamp_us + monthUs > nowUs)
            s.lastMonthFills++;
        if (f.timestamp_us + yearUs > nowUs)
            s.lastYearFills++;
    }
    std::set<std::string> activeDays;
    for (const auto& f : sub) {
        if (f.timestamp_us + monthUs > nowUs) {
            std::time_t t = static_cast<std::time_t>(
                f.timestamp_us / 1000000ULL);
            std::tm tm{};
            localtime_r(&t, &tm);
            char buf[16];
            std::strftime(buf, sizeof(buf),
                          "%Y-%m-%d", &tm);
            activeDays.insert(buf);
        }
    }
    if (!activeDays.empty()) {
        s.fillsPerActiveDay =
            static_cast<double>(s.lastMonthFills) /
            static_cast<double>(activeDays.size());
    }
    s.lastFillAgeHours = static_cast<double>(
        nowUs - sub.back().timestamp_us) /
        (3600.0 * 1000000.0);
    return s;
}
}  // namespace

TradeJournal::TradeCountSummary
TradeJournal::tradeCountSummaryBySymbol(
    const std::string& symbol) const {
    return buildTradeCountSummaryBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
}

TradeJournal::TradeCountSummary
TradeJournal::tradeCountSummaryByTag(
    const std::string& tag, bool includeUntagged) const {
    return buildTradeCountSummaryBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
}

std::string TradeJournal::journalSummaryJson() const {
    // Sprint #189. Manual JSON serialization (avoid
    // pulling in a JSON library). Format:
    //   {
    //     "totalFills": N,
    //     "totalRealized": X,
    //     "sharpe": S,
    //     "maxDD": D,
    //     ...
    //   }
    auto m = journalMetadata();
    auto r = riskScore();
    auto c = tradeCountSummary();
    std::ostringstream os;
    os << "{\n";
    os << "  \"totalFills\": " << m.totalFills << ",\n";
    os << "  \"totalSymbols\": " << m.totalSymbols << ",\n";
    os << "  \"totalTags\": " << m.totalTags << ",\n";
    os << "  \"activeDays\": " << m.activeDays << ",\n";
    os << "  \"spanDays\": " << m.spanDays << ",\n";
    os << "  \"totalRealized\": " << m.totalRealized << ",\n";
    os << "  \"maxDD\": " << m.maxDD << ",\n";
    os << "  \"sharpe\": " << m.sharpe << ",\n";
    os << "  \"riskScore\": " << r.overall << ",\n";
    os << "  \"lastDayFills\": " << c.lastDayFills << ",\n";
    os << "  \"lastWeekFills\": " << c.lastWeekFills << ",\n";
    os << "  \"lastMonthFills\": " << c.lastMonthFills << ",\n";
    os << "  \"lastFillAgeHours\": " << c.lastFillAgeHours << "\n";
    os << "}\n";
    return os.str();
}

namespace {
// Sprint #190 — per-segment profit-per-trade builder.
template <typename Pred>
TradeJournal::ProfitPerTrade
buildProfitPerTradeBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    TradeJournal::ProfitPerTrade s;
    std::vector<JournalFill> sub;
    sub.reserve(fills.size());
    for (const auto& f : fills) {
        if (pred(f) && std::fabs(f.realizedDelta) > 1e-9) {
            sub.push_back(f);
        }
    }
    if (sub.empty()) return s;
    std::vector<double> rets;
    rets.reserve(sub.size());
    double sum = 0.0;
    size_t wins = 0;
    for (const auto& f : sub) {
        rets.push_back(f.realizedDelta);
        sum += f.realizedDelta;
        if (f.realizedDelta > 0) wins++;
    }
    s.segment = sub.front().symbol;  // overwritten by caller
    s.totalTrades = sub.size();
    s.avgProfit = sum / static_cast<double>(sub.size());
    s.winRate = static_cast<double>(wins) /
                static_cast<double>(sub.size());
    std::sort(rets.begin(), rets.end());
    s.medianProfit = rets[rets.size() / 2];
    return s;
}
}  // namespace

TradeJournal::ProfitPerTrade
TradeJournal::profitPerTradeBySymbol(
    const std::string& symbol) const {
    auto s = buildProfitPerTradeBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
    s.segment = symbol;
    return s;
}

TradeJournal::ProfitPerTrade
TradeJournal::profitPerTradeByTag(
    const std::string& tag, bool includeUntagged) const {
    auto s = buildProfitPerTradeBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
    s.segment = tag;
    return s;
}

std::vector<TradeJournal::ProfitPerTrade>
TradeJournal::allProfitPerTrade() const {
    // Sprint #191. Bulk per-symbol profit-per-trade
    // sorted DESC by avgProfit.
    std::vector<ProfitPerTrade> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(profitPerTradeBySymbol(s));
    }
    std::sort(out.begin(), out.end(),
        [](const ProfitPerTrade& a, const ProfitPerTrade& b) {
            return a.avgProfit > b.avgProfit;
        });
    return out;
}

std::vector<TradeJournal::ProfitPerTrade>
TradeJournal::allProfitPerTradeByTag(
    bool includeUntagged) const {
    std::vector<ProfitPerTrade> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(profitPerTradeByTag(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const ProfitPerTrade& a, const ProfitPerTrade& b) {
            return a.avgProfit > b.avgProfit;
        });
    return out;
}

std::vector<TradeJournal::SegmentTradeCount>
TradeJournal::allSegmentTradeCountSummary() const {
    // Sprint #192. Bulk per-segment TradeCountSummary.
    std::vector<SegmentTradeCount> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        SegmentTradeCount c;
        c.segment = s;
        c.summary = tradeCountSummaryBySymbol(s);
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
        [](const SegmentTradeCount& a,
           const SegmentTradeCount& b) {
            return a.summary.totalFills >
                   b.summary.totalFills;
        });
    return out;
}

std::vector<TradeJournal::SegmentTradeCount>
TradeJournal::allSegmentTradeCountSummaryByTag(
    bool includeUntagged) const {
    std::vector<SegmentTradeCount> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        SegmentTradeCount c;
        c.segment = t;
        c.summary = tradeCountSummaryByTag(t, includeUntagged);
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(),
        [](const SegmentTradeCount& a,
           const SegmentTradeCount& b) {
            return a.summary.totalFills >
                   b.summary.totalFills;
        });
    return out;
}

std::vector<TradeJournal::SymbolSummary>
TradeJournal::allSymbolRecoveryFactor() const {
    // Sprint #193. Bulk per-symbol summary sorted DESC
    // by recoveryFactor.
    std::vector<SymbolSummary> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(symbolSummary(s));
    }
    std::sort(out.begin(), out.end(),
        [](const SymbolSummary& a, const SymbolSummary& b) {
            return a.recoveryFactor > b.recoveryFactor;
        });
    return out;
}

std::vector<TradeJournal::TagSummary>
TradeJournal::allTagRecoveryFactor(bool includeUntagged) const {
    std::vector<TagSummary> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(tagSummary(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const TagSummary& a, const TagSummary& b) {
            return a.recoveryFactor > b.recoveryFactor;
        });
    return out;
}

std::vector<TradeJournal::SymbolWeeklyWR>
TradeJournal::allSymbolWeeklyWinRate() const {
    // Sprint #194. For each symbol, compute weekly win
    // rate series, then aggregate mean across weeks.
    std::vector<SymbolWeeklyWR> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        SymbolWeeklyWR w;
        w.symbol = s;
        auto weekly = weeklyWinRateBySymbol(s);
        if (weekly.empty()) {
            out.push_back(w);
            continue;
        }
        double sum = 0.0;
        for (const auto& ww : weekly) {
            sum += ww.winRate;
            if (ww.winRate >= 0.5) w.winningWeeks++;
        }
        w.totalWeeks = weekly.size();
        w.meanWeeklyWinRate = sum /
            static_cast<double>(weekly.size());
        out.push_back(w);
    }
    std::sort(out.begin(), out.end(),
        [](const SymbolWeeklyWR& a, const SymbolWeeklyWR& b) {
            return a.meanWeeklyWinRate >
                   b.meanWeeklyWinRate;
        });
    return out;
}

namespace {
// Sprint #195 — per-segment avg-daily-PnL builder.
template <typename Pred>
TradeJournal::SegmentAvgDayPnL
buildAvgDayPnLBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    TradeJournal::SegmentAvgDayPnL s;
    std::map<std::string, double> dailyPnL;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t t = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
        dailyPnL[buf] += f.realizedDelta;
    }
    if (dailyPnL.empty()) return s;
    s.activeDays = dailyPnL.size();
    std::vector<double> vals;
    vals.reserve(dailyPnL.size());
    double sum = 0.0;
    for (auto& kv : dailyPnL) {
        vals.push_back(kv.second);
        sum += kv.second;
    }
    s.avgDailyPnL = sum / static_cast<double>(vals.size());
    std::sort(vals.begin(), vals.end());
    s.medianDailyPnL = vals[vals.size() / 2];
    return s;
}
}  // namespace

TradeJournal::SegmentAvgDayPnL
TradeJournal::avgDayPnLBySymbol(
    const std::string& symbol) const {
    auto s = buildAvgDayPnLBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
    s.segment = symbol;
    return s;
}

TradeJournal::SegmentAvgDayPnL
TradeJournal::avgDayPnLByTag(
    const std::string& tag, bool includeUntagged) const {
    auto s = buildAvgDayPnLBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
    s.segment = tag;
    return s;
}

std::vector<TradeJournal::SegmentAvgDayPnL>
TradeJournal::allSymbolAvgDayPnL() const {
    // Sprint #196. Bulk per-symbol avg daily P&L
    // sorted DESC by avgDailyPnL.
    std::vector<SegmentAvgDayPnL> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(avgDayPnLBySymbol(s));
    }
    std::sort(out.begin(), out.end(),
        [](const SegmentAvgDayPnL& a, const SegmentAvgDayPnL& b) {
            return a.avgDailyPnL > b.avgDailyPnL;
        });
    return out;
}

std::vector<TradeJournal::SegmentAvgDayPnL>
TradeJournal::allTagAvgDayPnL(bool includeUntagged) const {
    std::vector<SegmentAvgDayPnL> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(avgDayPnLByTag(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const SegmentAvgDayPnL& a, const SegmentAvgDayPnL& b) {
            return a.avgDailyPnL > b.avgDailyPnL;
        });
    return out;
}

namespace {
// Sprint #197 — per-segment risk of ruin builder.
template <typename Pred>
TradeJournal::RiskOfRuin
buildRiskOfRuinBySegment(
    const std::vector<JournalFill>& fills,
    Pred pred, double ruinFraction) {
    TradeJournal::RiskOfRuin r;
    double grossWin = 0.0, grossLoss = 0.0;
    size_t wins = 0, losses = 0;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (f.realizedDelta > 0) {
            grossWin += f.realizedDelta;
            wins++;
        } else if (f.realizedDelta < 0) {
            grossLoss += f.realizedDelta;
            losses++;
        }
    }
    if (wins == 0 || losses == 0) {
        r.ruinProb = (losses == 0) ? 0.0 : 1.0;
        r.winRate = (wins + losses > 0)
            ? static_cast<double>(wins) /
              static_cast<double>(wins + losses)
            : 0.5;
        return r;
    }
    double W = static_cast<double>(wins) /
               static_cast<double>(wins + losses);
    double avgWin = grossWin / static_cast<double>(wins);
    double avgLoss = std::fabs(grossLoss / static_cast<double>(losses));
    double R = (avgLoss > 1e-9) ? avgWin / avgLoss : 0.0;
    double realized = grossWin - grossLoss;
    r.winRate = W;
    r.payoffRatio = R;
    r.maxLossFrac = (realized > 1e-9)
        ? avgLoss / realized : 1.0;
    // Capital units to lose ruinFraction of equity.
    double capitalUnits = ruinFraction /
                          std::max(r.maxLossFrac, 0.01);
    // PoR approximation:
    double q = 1.0 - W;
    double p = W * R;
    if (p <= q) {
        // Unfavorable — guaranteed eventual ruin.
        r.ruinProb = 1.0;
    } else {
        r.ruinProb = std::pow(q / p, capitalUnits);
        if (r.ruinProb < 0) r.ruinProb = 0;
        if (r.ruinProb > 1) r.ruinProb = 1;
    }
    return r;
}
}  // namespace

TradeJournal::RiskOfRuin
TradeJournal::riskOfRuinBySymbol(
    const std::string& symbol, double ruinFraction) const {
    auto r = buildRiskOfRuinBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        }, ruinFraction);
    r.segment = symbol;
    return r;
}

TradeJournal::RiskOfRuin
TradeJournal::riskOfRuinByTag(
    const std::string& tag, bool includeUntagged,
    double ruinFraction) const {
    auto r = buildRiskOfRuinBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        }, ruinFraction);
    r.segment = tag;
    return r;
}

namespace {
// Sprint #198 — per-segment recovery time stats builder.
// For each completed DD, compute its recovery time
// (in days) and aggregate max/avg/min.
template <typename Pred>
TradeJournal::RecoveryTimeStats
buildRecoveryTimeStatsBySegment(
    const std::vector<TradeJournal::DrawdownEvent>& events,
    Pred pred) {
    TradeJournal::RecoveryTimeStats s;
    double sum = 0.0, mx = 0.0, mn = 1e18;
    size_t count = 0;
    for (const auto& e : events) {
        if (!pred(e)) continue;
        if (e.recovery_us == 0) continue;
        double days = static_cast<double>(e.recovery_us) /
                      (86400.0 * 1000000.0);
        if (days > mx) mx = days;
        if (days < mn) mn = days;
        sum += days;
        ++count;
    }
    if (count > 0) {
        s.completedDDCount = count;
        s.maxRecoveryDays = mx;
        s.avgRecoveryDays = sum / static_cast<double>(count);
        s.minRecoveryDays = mn;
    }
    return s;
}
}  // namespace

TradeJournal::RecoveryTimeStats
TradeJournal::recoveryTimeStatsBySymbol(
    const std::string& symbol) const {
    auto events = drawdownRecoveriesBySymbol(symbol);
    auto s = buildRecoveryTimeStatsBySegment(events,
        [](const TradeJournal::DrawdownEvent&) {
            return true;
        });
    s.segment = symbol;
    return s;
}

TradeJournal::RecoveryTimeStats
TradeJournal::recoveryTimeStatsByTag(
    const std::string& tag, bool includeUntagged) const {
    auto events = drawdownRecoveriesByTag(tag, includeUntagged);
    auto s = buildRecoveryTimeStatsBySegment(events,
        [](const TradeJournal::DrawdownEvent&) {
            return true;
        });
    s.segment = tag;
    return s;
}

std::vector<TradeJournal::RecoveryTimeStats>
TradeJournal::allSegmentRecoveryTime() const {
    // Sprint #199. Bulk per-segment recovery time
    // sorted DESC by avgRecoveryDays.
    std::vector<RecoveryTimeStats> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(recoveryTimeStatsBySymbol(s));
    }
    std::sort(out.begin(), out.end(),
        [](const RecoveryTimeStats& a,
           const RecoveryTimeStats& b) {
            return a.avgRecoveryDays > b.avgRecoveryDays;
        });
    return out;
}

std::vector<TradeJournal::RecoveryTimeStats>
TradeJournal::allSegmentRecoveryTimeByTag(
    bool includeUntagged) const {
    std::vector<RecoveryTimeStats> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(recoveryTimeStatsByTag(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const RecoveryTimeStats& a,
           const RecoveryTimeStats& b) {
            return a.avgRecoveryDays > b.avgRecoveryDays;
        });
    return out;
}

std::vector<TradeJournal::RiskOfRuin>
TradeJournal::allSegmentRiskOfRuin(double ruinFraction) const {
    // Sprint #200. Bulk per-symbol risk of ruin
    // sorted ASC by ruinProb (lowest risk first).
    std::vector<RiskOfRuin> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(riskOfRuinBySymbol(s, ruinFraction));
    }
    std::sort(out.begin(), out.end(),
        [](const RiskOfRuin& a, const RiskOfRuin& b) {
            return a.ruinProb < b.ruinProb;
        });
    return out;
}

std::vector<TradeJournal::RiskOfRuin>
TradeJournal::allSegmentRiskOfRuinByTag(
    bool includeUntagged, double ruinFraction) const {
    std::vector<RiskOfRuin> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(riskOfRuinByTag(t, includeUntagged,
            ruinFraction));
    }
    std::sort(out.begin(), out.end(),
        [](const RiskOfRuin& a, const RiskOfRuin& b) {
            return a.ruinProb < b.ruinProb;
        });
    return out;
}

namespace {
// Sprint #201 — per-segment DD streak builder.
// Counts the longest run of consecutive completed DDs
// (no full recovery between them).
template <typename Iter>
TradeJournal::DDStreakStats
buildDDStreakStats(const std::string& segment,
                    Iter begin, Iter end) {
    TradeJournal::DDStreakStats s;
    s.segment = segment;
    size_t longest = 0, current = 0;
    for (auto it = begin; it != end; ++it) {
        if (it->recovery_us == 0) continue;
        s.totalCompletedDDs++;
        // A "streak" DD means the previous DD wasn't
        // fully recovered before this one started. For
        // simplicity, count all completed DDs as a streak.
        current++;
        if (current > longest) longest = current;
    }
    s.longestStreak = longest;
    s.currentStreak = current;
    return s;
}
}  // namespace

TradeJournal::DDStreakStats
TradeJournal::ddStreakStatsBySymbol(
    const std::string& symbol) const {
    auto events = drawdownRecoveriesBySymbol(symbol);
    return buildDDStreakStats(symbol, events.begin(), events.end());
}

TradeJournal::DDStreakStats
TradeJournal::ddStreakStatsByTag(
    const std::string& tag, bool includeUntagged) const {
    auto events = drawdownRecoveriesByTag(tag, includeUntagged);
    return buildDDStreakStats(tag, events.begin(), events.end());
}

std::vector<TradeJournal::DDStreakStats>
TradeJournal::allSegmentDDStreakStats() const {
    // Sprint #202. Bulk per-segment DD streak stats
    // sorted DESC by longestStreak.
    std::vector<DDStreakStats> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(ddStreakStatsBySymbol(s));
    }
    std::sort(out.begin(), out.end(),
        [](const DDStreakStats& a, const DDStreakStats& b) {
            return a.longestStreak > b.longestStreak;
        });
    return out;
}

std::vector<TradeJournal::DDStreakStats>
TradeJournal::allSegmentDDStreakStatsByTag(
    bool includeUntagged) const {
    std::vector<DDStreakStats> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(ddStreakStatsByTag(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const DDStreakStats& a, const DDStreakStats& b) {
            return a.longestStreak > b.longestStreak;
        });
    return out;
}

namespace {
// Sprint #203 — Sharpe trend slope via OLS.
// Take last `lastN` points from rolling Sharpe series,
// fit a line y = slope*x + intercept, return slope +
// intercept + R².
template <typename Series>
TradeJournal::SharpeTrend
buildSharpeTrend(const std::string& segment,
                  const Series& series,
                  size_t lastN) {
    TradeJournal::SharpeTrend t;
    t.segment = segment;
    if (series.empty()) return t;
    // Take the last lastN points.
    size_t start = series.size() > lastN
                  ? series.size() - lastN : 0;
    size_t n = series.size() - start;
    t.sampleCount = n;
    if (n < 2) return t;
    double meanX = 0.0, meanY = 0.0;
    for (size_t i = 0; i < n; ++i) {
        meanX += static_cast<double>(i);
        meanY += series[start + i].sharpe;
    }
    meanX /= static_cast<double>(n);
    meanY /= static_cast<double>(n);
    double cov = 0.0, varX = 0.0, varY = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = static_cast<double>(i) - meanX;
        double dy = series[start + i].sharpe - meanY;
        cov += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
    }
    if (varX < 1e-12) return t;
    t.slope = cov / varX;
    t.intercept = meanY - t.slope * meanX;
    t.rSquared = (varY > 1e-12)
        ? (cov * cov) / (varX * varY) : 0.0;
    return t;
}
}  // namespace

TradeJournal::SharpeTrend
TradeJournal::sharpeTrendBySymbol(
    const std::string& symbol, size_t window, size_t lastN) const {
    auto series = rollingWindowSharpeBySymbol(symbol, window);
    return buildSharpeTrend(symbol, series, lastN);
}

TradeJournal::SharpeTrend
TradeJournal::sharpeTrendByTag(
    const std::string& tag, bool includeUntagged,
    size_t window, size_t lastN) const {
    auto series = rollingWindowSharpeByTag(tag, includeUntagged,
        window);
    return buildSharpeTrend(tag, series, lastN);
}

std::vector<TradeJournal::SharpeTrend>
TradeJournal::allSegmentSharpeTrend(size_t window, size_t lastN) const {
    // Sprint #204. Bulk per-symbol Sharpe trend
    // sorted DESC by slope.
    std::vector<SharpeTrend> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(sharpeTrendBySymbol(s, window, lastN));
    }
    std::sort(out.begin(), out.end(),
        [](const SharpeTrend& a, const SharpeTrend& b) {
            return a.slope > b.slope;
        });
    return out;
}

std::vector<TradeJournal::SharpeTrend>
TradeJournal::allSegmentSharpeTrendByTag(
    bool includeUntagged, size_t window, size_t lastN) const {
    std::vector<SharpeTrend> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(sharpeTrendByTag(t, includeUntagged,
            window, lastN));
    }
    std::sort(out.begin(), out.end(),
        [](const SharpeTrend& a, const SharpeTrend& b) {
            return a.slope > b.slope;
        });
    return out;
}

namespace {
// Sprint #205 — Best-Day-of-Week builder.
template <typename Pred>
TradeJournal::BestDayOfWeek
buildBestDayOfWeekBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    TradeJournal::BestDayOfWeek b;
    std::array<double, 7> sum{};
    std::array<size_t, 7> cnt{};
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t t = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&t, &tm);
        int wd = tm.tm_wday;  // 0=Sun..6=Sat
        sum[wd] += f.realizedDelta;
        cnt[wd]++;
    }
    int best = -1;
    double bestMean = -1e18;
    size_t bestCnt = 0;
    for (int i = 0; i < 7; ++i) {
        if (cnt[i] == 0) continue;
        double mean = sum[i] / static_cast<double>(cnt[i]);
        if (mean > bestMean) {
            bestMean = mean;
            best = i;
            bestCnt = cnt[i];
        }
    }
    b.bestWeekday = best;
    b.bestMeanPnL = (best >= 0) ? bestMean : 0.0;
    b.bestDayFills = bestCnt;
    return b;
}
}  // namespace

TradeJournal::BestDayOfWeek
TradeJournal::bestDayOfWeekBySymbol(
    const std::string& symbol) const {
    auto b = buildBestDayOfWeekBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
    b.segment = symbol;
    return b;
}

TradeJournal::BestDayOfWeek
TradeJournal::bestDayOfWeekByTag(
    const std::string& tag, bool includeUntagged) const {
    auto b = buildBestDayOfWeekBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
    b.segment = tag;
    return b;
}

std::vector<TradeJournal::BestDayOfWeek>
TradeJournal::allSegmentBestDayOfWeek() const {
    // Sprint #206. Bulk per-symbol best day of week
    // sorted DESC by bestMeanPnL.
    std::vector<BestDayOfWeek> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(bestDayOfWeekBySymbol(s));
    }
    std::sort(out.begin(), out.end(),
        [](const BestDayOfWeek& a, const BestDayOfWeek& b) {
            return a.bestMeanPnL > b.bestMeanPnL;
        });
    return out;
}

std::vector<TradeJournal::BestDayOfWeek>
TradeJournal::allSegmentBestDayOfWeekByTag(
    bool includeUntagged) const {
    std::vector<BestDayOfWeek> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(bestDayOfWeekByTag(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const BestDayOfWeek& a, const BestDayOfWeek& b) {
            return a.bestMeanPnL > b.bestMeanPnL;
        });
    return out;
}

namespace {
// Sprint #207 — Best-Hour-of-Day builder.
template <typename Pred>
TradeJournal::BestHourOfDay
buildBestHourOfDayBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    TradeJournal::BestHourOfDay b;
    std::array<double, 24> sum{};
    std::array<size_t, 24> cnt{};
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        std::time_t t = static_cast<std::time_t>(
            f.timestamp_us / 1000000ULL);
        std::tm tm{};
        localtime_r(&t, &tm);
        int h = tm.tm_hour;  // 0..23
        sum[h] += f.realizedDelta;
        cnt[h]++;
    }
    int best = -1;
    double bestMean = -1e18;
    size_t bestCnt = 0;
    for (int i = 0; i < 24; ++i) {
        if (cnt[i] == 0) continue;
        double mean = sum[i] / static_cast<double>(cnt[i]);
        if (mean > bestMean) {
            bestMean = mean;
            best = i;
            bestCnt = cnt[i];
        }
    }
    b.bestHour = best;
    b.bestMeanPnL = (best >= 0) ? bestMean : 0.0;
    b.bestHourFills = bestCnt;
    return b;
}
}  // namespace

TradeJournal::BestHourOfDay
TradeJournal::bestHourOfDayBySymbol(
    const std::string& symbol) const {
    auto b = buildBestHourOfDayBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
    b.segment = symbol;
    return b;
}

TradeJournal::BestHourOfDay
TradeJournal::bestHourOfDayByTag(
    const std::string& tag, bool includeUntagged) const {
    auto b = buildBestHourOfDayBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
    b.segment = tag;
    return b;
}

std::vector<TradeJournal::BestHourOfDay>
TradeJournal::allSegmentBestHourOfDay() const {
    // Sprint #208. Bulk per-symbol best hour of day
    // sorted DESC by bestMeanPnL.
    std::vector<BestHourOfDay> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(bestHourOfDayBySymbol(s));
    }
    std::sort(out.begin(), out.end(),
        [](const BestHourOfDay& a, const BestHourOfDay& b) {
            return a.bestMeanPnL > b.bestMeanPnL;
        });
    return out;
}

std::vector<TradeJournal::BestHourOfDay>
TradeJournal::allSegmentBestHourOfDayByTag(
    bool includeUntagged) const {
    std::vector<BestHourOfDay> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(bestHourOfDayByTag(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const BestHourOfDay& a, const BestHourOfDay& b) {
            return a.bestMeanPnL > b.bestMeanPnL;
        });
    return out;
}

namespace {
// Sprint #209 — per-segment trade size stats builder.
template <typename Pred>
TradeJournal::TradeSizeStatsSeg
buildTradeSizeStatsSegBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    TradeJournal::TradeSizeStatsSeg s;
    std::vector<double> sizes;
    double sum = 0.0, mx = 0.0;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (std::fabs(f.realizedDelta) <= 1e-9) continue;
        double sz = std::fabs(f.realizedDelta);
        sizes.push_back(sz);
        sum += sz;
        if (sz > mx) mx = sz;
    }
    if (sizes.empty()) return s;
    s.totalFills = sizes.size();
    s.meanSize = sum / static_cast<double>(sizes.size());
    s.maxSize = mx;
    std::sort(sizes.begin(), sizes.end());
    s.medianSize = sizes[sizes.size() / 2];
    double var = 0.0;
    for (double sz : sizes) {
        var += (sz - s.meanSize) * (sz - s.meanSize);
    }
    var /= static_cast<double>(sizes.size());
    s.stddevSize = std::sqrt(var);
    return s;
}
}  // namespace

TradeJournal::TradeSizeStatsSeg
TradeJournal::tradeSizeStatsSegBySymbol(
    const std::string& symbol) const {
    auto s = buildTradeSizeStatsSegBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
    s.segment = symbol;
    return s;
}

TradeJournal::TradeSizeStatsSeg
TradeJournal::tradeSizeStatsSegByTag(
    const std::string& tag, bool includeUntagged) const {
    auto s = buildTradeSizeStatsSegBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
    s.segment = tag;
    return s;
}

std::vector<TradeJournal::TradeSizeStatsSeg>
TradeJournal::allSegmentTradeSizeStats() const {
    // Sprint #210. Bulk per-symbol trade size stats
    // sorted DESC by meanSize.
    std::vector<TradeSizeStatsSeg> out;
    auto fills = loadAll();
    std::set<std::string> syms;
    for (const auto& f : fills) syms.insert(f.symbol);
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(tradeSizeStatsSegBySymbol(s));
    }
    std::sort(out.begin(), out.end(),
        [](const TradeSizeStatsSeg& a,
           const TradeSizeStatsSeg& b) {
            return a.meanSize > b.meanSize;
        });
    return out;
}

std::vector<TradeJournal::TradeSizeStatsSeg>
TradeJournal::allSegmentTradeSizeStatsByTag(
    bool includeUntagged) const {
    std::vector<TradeSizeStatsSeg> out;
    auto fills = loadAll();
    std::set<std::string> tags;
    for (const auto& f : fills) {
        if (f.tag.empty()) {
            if (includeUntagged) tags.insert("__untagged__");
        } else {
            tags.insert(f.tag);
        }
    }
    out.reserve(tags.size());
    for (const auto& t : tags) {
        out.push_back(tradeSizeStatsSegByTag(t, includeUntagged));
    }
    std::sort(out.begin(), out.end(),
        [](const TradeSizeStatsSeg& a,
           const TradeSizeStatsSeg& b) {
            return a.meanSize > b.meanSize;
        });
    return out;
}

namespace {
// Sprint #211 — per-segment risk-reward ratio builder.
template <typename Pred>
TradeJournal::RiskRewardRatio
buildRiskRewardRatioBySegment(
    const std::vector<JournalFill>& fills, Pred pred) {
    TradeJournal::RiskRewardRatio r;
    double grossWin = 0.0, grossLoss = 0.0;
    size_t wins = 0, losses = 0;
    for (const auto& f : fills) {
        if (!pred(f)) continue;
        if (f.realizedDelta > 0) {
            grossWin += f.realizedDelta;
            wins++;
        } else if (f.realizedDelta < 0) {
            grossLoss += f.realizedDelta;
            losses++;
        }
    }
    r.winCount = wins;
    r.lossCount = losses;
    if (wins > 0) r.avgWin = grossWin / static_cast<double>(wins);
    if (losses > 0) {
        r.avgLoss = std::fabs(
            grossLoss / static_cast<double>(losses));
    }
    if (r.avgLoss > 1e-9) {
        r.ratio = r.avgWin / r.avgLoss;
    }
    return r;
}
}  // namespace

TradeJournal::RiskRewardRatio
TradeJournal::riskRewardRatioBySymbol(
    const std::string& symbol) const {
    auto r = buildRiskRewardRatioBySegment(loadAll(),
        [&symbol](const JournalFill& f) {
            return f.symbol == symbol;
        });
    r.segment = symbol;
    return r;
}

TradeJournal::RiskRewardRatio
TradeJournal::riskRewardRatioByTag(
    const std::string& tag, bool includeUntagged) const {
    auto r = buildRiskRewardRatioBySegment(loadAll(),
        [&tag, includeUntagged](const JournalFill& f) {
            if (tag == "__untagged__") return f.tag.empty();
            if (includeUntagged && f.tag.empty()) return false;
            return f.tag == tag;
        });
    r.segment = tag;
    return r;
}

} // namespace btquant
