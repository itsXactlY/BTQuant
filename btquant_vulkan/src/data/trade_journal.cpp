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
    // Sprint #102. Per-tag mirror. Tag mode honors
    // includeUntagged.
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

} // namespace btquant
