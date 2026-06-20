#include "trade_journal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
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

TradeJournal::Drawdown TradeJournal::maxDrawdown() const {
    Drawdown dd;
    auto daily = realizedByDay();
    if (daily.empty()) return dd;

    // Walk the equity curve. Track:
    //   running peak (the highest equity seen so far)
    //   current drawdown (peak - current equity)
    //   worst drawdown seen (and the dates that bracket it)
    double equity  = 0.0;
    double peak    = 0.0;
    double worstDD = 0.0;
    std::string peakDateAtWorst;   // date of the high that preceded worstDD
    std::string troughDateAtWorst; // date of the low that ended worstDD

    for (const auto& kv : daily) {
        equity += kv.second;
        if (equity > peak) {
            peak = equity;
            // A new high water mark resets the peakDateAtWorst to
            // the date the peak was reached — but only if we
            // haven't yet seen any drawdown. Once we've recorded a
            // worstDD, the peakDate for the *current* drawdown is
            // whatever the peak was when this drawdown started,
            // not necessarily today.
        }
        double curDD = peak - equity;  // >= 0
        if (curDD > worstDD + 1e-9) {
            worstDD = curDD;
            troughDateAtWorst = kv.first;
            // peakDateAtWorst: we need the date of the high that
            // preceded this drawdown. Walk backwards from today
            // until we find the last peak. Simpler: track it
            // forward — when equity first exceeded the previous
            // peak, record that date as the new "peak anchor".
        }
    }

    // Recompute peakDateAtWorst properly: walk forward, tracking
    // the date of the most recent equity-high (running peak).
    // The peak anchor for the worst drawdown is the last date on
    // which equity reached the peak that the drawdown started
    // from. Re-walking costs O(N) which matches the loop above —
    // could fuse but clarity wins.
    equity = 0.0;
    double anchorPeak = 0.0;
    std::string anchorDate;
    std::string troughAnchor;   // troughDate → anchorDate mapping
    double runningWorstDD = 0.0;
    for (const auto& kv : daily) {
        equity += kv.second;
        if (equity >= anchorPeak) {
            anchorPeak = equity;
            anchorDate = kv.first;
        }
        double curDD = anchorPeak - equity;
        if (curDD > runningWorstDD + 1e-9) {
            runningWorstDD = curDD;
            troughAnchor = kv.first;
            // The peak that started this drawdown is anchorDate.
        }
    }

    dd.maxDrawdown = runningWorstDD;
    if (runningWorstDD > 1e-9) {
        // Final peakDateAtWorst: walk forward once more, this time
        // stopping when we hit the troughDate and recording the
        // peak that was current at that moment.
        equity = 0.0;
        double p = 0.0;
        std::string lastPeakDate;
        for (const auto& kv : daily) {
            equity += kv.second;
            if (equity >= p) {
                p = equity;
                lastPeakDate = kv.first;
            }
            if (kv.first == troughAnchor) {
                dd.peakDate   = lastPeakDate;
                dd.troughDate = troughAnchor;
                break;
            }
        }
    }

    // currentDD: peak - last equity.
    equity = 0.0;
    double lastPeak = 0.0;
    for (const auto& kv : daily) {
        equity += kv.second;
        if (equity > lastPeak) lastPeak = equity;
    }
    dd.currentDD = lastPeak - equity;

    return dd;
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
    // Annualization factor for trading days. 252 is the
    // industry-standard convention (US equity markets). For
    // crypto, 365 might be more accurate — but the trader is
    // asking for the classic Sharpe so we match expectations.
    constexpr double kTradingDays = 252.0;

    Sharpe out;
    auto daily = realizedByDay();
    out.sampleSize = daily.size();
    if (daily.empty()) return out;

    // 1) Mean.
    double sum = 0.0;
    for (const auto& kv : daily) sum += kv.second;
    out.meanDailyReturn = sum / static_cast<double>(daily.size());

    // 2) Sample stddev (Bessel-corrected, n-1). Single day → 0.
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

    // 3) Sharpe = mean / stddev. Annualized by sqrt(252).
    if (out.stddevDailyReturn > 1e-9) {
        out.dailySharpe = out.meanDailyReturn / out.stddevDailyReturn;
        out.annualizedSharpe = out.dailySharpe * std::sqrt(kTradingDays);
    } else {
        // All days have the same return → stddev 0, ratio
        // undefined. Sentinel: 0 (not inf, not NaN) — caller can
        // format as "—" without special-casing.
        out.dailySharpe      = 0.0;
        out.annualizedSharpe = 0.0;
    }
    return out;
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
