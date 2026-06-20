#include "trade_journal.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

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

std::string TradeJournal::toJsonLine(const JournalFill& r) {
    char buf[512];
    // Booleans serialize as true/false barewords; numbers as JSON numbers.
    std::snprintf(buf, sizeof(buf),
        "{\"ts\":%llu,\"sym\":\"%s\",\"side\":\"%s\","
        "\"qty\":%.10g,\"px\":%.10g,\"realized\":%.10g}",
        static_cast<unsigned long long>(r.timestamp_us),
        jsonEscape(r.symbol).c_str(),
        r.isLong ? "buy" : "sell",
        r.qty, r.price, r.realizedDelta);
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
    // (qty, price), then P&L attribution (realized). Header uses
    // snake_case to match BTQuant's other CSV exports.
    os << "timestamp_iso,symbol,side,qty,price,realized_delta\n";
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
           << csvQuoteIfNeeded(realizedBuf) << "\n";
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

} // namespace btquant
