#include "layout_io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>

namespace btquant::util {

namespace {

// ---- Tiny JSON helpers (object: { "k": v, "k2": v2 }) ----
// Recognizes string / number / bool / null / nested object. Arrays
// not used by the layout format (yet). Strict enough to reject typos;
// permissive on whitespace.

struct Parser {
    const std::string& s;
    size_t pos = 0;
    explicit Parser(const std::string& src) : s(src) {}

    void skipWs() {
        while (pos < s.size() &&
               std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }
    char peek() {
        skipWs();
        return pos < s.size() ? s[pos] : '\0';
    }
    bool consume(char c) {
        skipWs();
        if (pos < s.size() && s[pos] == c) { ++pos; return true; }
        return false;
    }
    bool consumeStr(const std::string& lit) {
        skipWs();
        if (pos + lit.size() > s.size()) return false;
        if (s.compare(pos, lit.size(), lit) != 0) return false;
        pos += lit.size();
        return true;
    }

    // Parses a JSON string token (without surrounding quotes); the
    // caller has already consumed the opening quote.
    bool parseStringRaw(std::string& out) {
        out.clear();
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                char esc = s[pos + 1];
                switch (esc) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'n':  out.push_back('\n'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    default:   out.push_back(esc);  break;  // permissive
                }
                pos += 2;
            } else {
                out.push_back(s[pos++]);
            }
        }
        if (pos >= s.size()) return false;
        ++pos;  // skip closing quote
        return true;
    }

    bool parseString(std::string& out) {
        skipWs();
        if (pos >= s.size() || s[pos] != '"') return false;
        ++pos;
        return parseStringRaw(out);
    }

    bool parseNumber(double& out) {
        skipWs();
        size_t start = pos;
        if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
        while (pos < s.size() &&
               (std::isdigit(static_cast<unsigned char>(s[pos])) ||
                s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' ||
                s[pos] == '+' || s[pos] == '-')) ++pos;
        if (pos == start) return false;
        try { out = std::stod(s.substr(start, pos - start)); }
        catch (...) { return false; }
        return true;
    }

    bool parseBool(bool& out) {
        if (consumeStr("true"))  { out = true;  return true; }
        if (consumeStr("false")) { out = false; return true; }
        return false;
    }

    bool parseValue(std::string& out) { return parseString(out); }
    bool parseValue(double& out)     { return parseNumber(out); }
    bool parseValue(bool& out)       { return parseBool(out); }

    // Parses { "k1": v1, "k2": v2 } and invokes a callback for each
    // key/value pair. Stops at matching close brace. Returns true on
    // well-formed object (possibly empty).
    template <typename Fn>
    bool parseObject(Fn onKey) {
        if (!consume('{')) return false;
        if (consume('}')) return true;
        while (true) {
            std::string key;
            if (!parseString(key)) return false;
            if (!consume(':'))    return false;
            onKey(key);
            if (!consume(',')) break;
        }
        return consume('}');
    }
};

// JSON-escape a string value.
std::string jsonEscape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 2);
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
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

// Write a flat key=value pair (one per line) — readability + diff-
// friendliness over a single-line blob. Caller is responsible for
// emitting the surrounding braces + commas (see writeKvList).
void writeKv(std::ostream& out, const std::string& key,
             const std::string& val, int indent) {
    out << std::string(indent, ' ') << "\"" << jsonEscape(key) << "\": \""
        << jsonEscape(val) << "\"";
}
void writeKv(std::ostream& out, const std::string& key,
             double val, int indent) {
    out << std::string(indent, ' ') << "\"" << jsonEscape(key) << "\": "
        << val;
}
void writeKv(std::ostream& out, const std::string& key,
             bool val, int indent) {
    out << std::string(indent, ' ') << "\"" << jsonEscape(key) << "\": "
        << (val ? "true" : "false");
}
void writeKv(std::ostream& out, const std::string& key,
             long val, int indent) {
    out << std::string(indent, ' ') << "\"" << jsonEscape(key) << "\": "
        << val;
}

// Collect key/value pairs then emit with commas BETWEEN but not AFTER
// each entry — produces strict (no trailing comma) JSON which any
// conforming parser accepts.
struct KvPair {
    std::string key;
    std::string body;       // pre-formatted "key": value (no comma)
};
void writeKvList(std::ostream& out, int indent,
                 const std::vector<KvPair>& pairs) {
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (i > 0) out << ",\n";
        out << std::string(indent, ' ') << pairs[i].body;
    }
    if (!pairs.empty()) out << "\n";
}

} // namespace

std::filesystem::path LayoutIO::layoutDir() {
    // Reuse Settings::defaultPath().parent_path() — that already points
    // at ~/.config/btquant_vulkan/, profiles/ subdir goes under it.
    namespace fs = std::filesystem;
    fs::path base = Settings::defaultPath().parent_path();
    return base / "profiles";
}

std::filesystem::path LayoutIO::layoutPath(const std::string& name) {
    // Sanitize like Settings::profilePath — keep a-zA-Z0-9_-. only,
    // truncate to 64. Pure copy of the logic so a layout called
    // "../../etc/passwd" still lands safely under profiles/.
    std::string clean;
    clean.reserve(name.size());
    for (char c : name) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-') {
            clean.push_back(c);
        }
    }
    if (clean.empty()) clean = "unnamed";
    if (clean.size() > 64) clean.resize(64);
    return layoutDir() / (clean + ".btqlayout");
}

LayoutSnapshot LayoutIO::fromSettings(const Settings& s,
                                      const std::string& dockLayout,
                                      const std::string& name) {
    LayoutSnapshot snap;
    snap.settings = s;
    snap.dockLayout = dockLayout;
    snap.name = name;
    snap.version = LayoutSnapshot::kLayoutVersion;
    return snap;
}

bool LayoutIO::save(const std::filesystem::path& path,
                    const LayoutSnapshot& snap) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path);
    if (!out.is_open()) return false;

    // Build per-block vectors so writeKvList can emit strict JSON
    // (commas BETWEEN entries, no trailing comma).
    std::vector<KvPair> top;
    {
        std::ostringstream s;
        s << "\"version\": " << static_cast<long>(snap.version);
        top.push_back({"version", s.str()});
    }
    if (!snap.name.empty()) {
        std::ostringstream s;
        s << "\"name\": \"" << jsonEscape(snap.name) << "\"";
        top.push_back({"name", s.str()});
    }

    const auto& s = snap.settings;

    std::vector<KvPair> widgets;
    {
        std::ostringstream ss;
        ss << "\"showOrderBook\": " << (s.showOrderBook ? "true" : "false");
        widgets.push_back({"showOrderBook", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showOrderBookDepth\": " << (s.showOrderBookDepth ? "true" : "false");
        widgets.push_back({"showOrderBookDepth", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showFootprint\": " << (s.showFootprint ? "true" : "false");
        widgets.push_back({"showFootprint", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showVPVR\": " << (s.showVPVR ? "true" : "false");
        widgets.push_back({"showVPVR", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showMultiVWAP\": " << (s.showMultiVWAP ? "true" : "false");
        widgets.push_back({"showMultiVWAP", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showRiskPanel\": " << (s.showRiskPanel ? "true" : "false");
        widgets.push_back({"showRiskPanel", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showDOM\": " << (s.showDOM ? "true" : "false");
        widgets.push_back({"showDOM", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showTrades\": " << (s.showTrades ? "true" : "false");
        widgets.push_back({"showTrades", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showTPO\": " << (s.showTPO ? "true" : "false");
        widgets.push_back({"showTPO", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showSettings\": " << (s.showSettings ? "true" : "false");
        widgets.push_back({"showSettings", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"showStatsOverlay\": " << (s.showStatsOverlay ? "true" : "false");
        widgets.push_back({"showStatsOverlay", ss.str()});
    }

    std::vector<KvPair> general;
    {
        std::ostringstream ss;
        ss << "\"fpsLimit\": " << s.fpsLimit;
        general.push_back({"fpsLimit", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"heatmapDensity\": " << s.heatmapDensity;
        general.push_back({"heatmapDensity", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"tradeWindowSeconds\": " << s.tradeWindowSeconds;
        general.push_back({"tradeWindowSeconds", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"theme\": " << s.theme;
        general.push_back({"theme", ss.str()});
    }

    std::vector<KvPair> risk;
    {
        std::ostringstream ss;
        ss << "\"maxPositionSizeUSD\": " << s.risk_maxPositionSizeUSD;
        risk.push_back({"maxPositionSizeUSD", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"maxLeverage\": " << s.risk_maxLeverage;
        risk.push_back({"maxLeverage", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"killOnDailyLossUSD\": " << s.risk_killOnDailyLossUSD;
        risk.push_back({"killOnDailyLossUSD", ss.str()});
    }
    {
        std::ostringstream ss;
        ss << "\"equityUSD\": " << s.risk_equityUSD;
        risk.push_back({"equityUSD", ss.str()});
    }

    {
        std::ostringstream ss;
        ss << "\"dockLayout\": \"" << jsonEscape(snap.dockLayout) << "\"";
        top.push_back({"dockLayout", ss.str()});
    }

    out << "{\n";
    writeKvList(out, 1, top);
    out << ",\n  \"widgets\": {\n";
    writeKvList(out, 4, widgets);
    out << "  },\n  \"general\": {\n";
    writeKvList(out, 4, general);
    out << "  },\n  \"risk\": {\n";
    writeKvList(out, 4, risk);
    out << "  }\n}\n";
    out.close();
    return out.good();
}

std::optional<LayoutSnapshot> LayoutIO::load(
    const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) return std::nullopt;
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    if (text.empty()) return std::nullopt;

    Parser p(text);
    LayoutSnapshot snap;
    bool haveVersion = false;

    auto onKey = [&](const std::string& key) {
        // Each value branch reads one JSON value and assigns to the
        // matching member. Unknown keys still need to consume the
        // value (string / number / bool / nested object) — otherwise
        // forward-compat files break the parser on the next key.
        char c = p.peek();
        if (key == "version" && (c == '-' || std::isdigit(static_cast<unsigned char>(c)))) {
            double v; if (p.parseNumber(v)) { snap.version = static_cast<int>(v); haveVersion = true; }
        } else if (key == "name" && c == '"') {
            p.parseString(snap.name);
        } else if (key == "widgets" && c == '{') {
            p.parseObject([&](const std::string& k) {
                bool b;
                if (p.parseBool(b)) {
                    if      (k == "showOrderBook")      snap.settings.showOrderBook = b;
                    else if (k == "showOrderBookDepth") snap.settings.showOrderBookDepth = b;
                    else if (k == "showFootprint")      snap.settings.showFootprint = b;
                    else if (k == "showVPVR")           snap.settings.showVPVR = b;
                    else if (k == "showMultiVWAP")      snap.settings.showMultiVWAP = b;
                    else if (k == "showRiskPanel")      snap.settings.showRiskPanel = b;
                    else if (k == "showDOM")            snap.settings.showDOM = b;
                    else if (k == "showTrades")         snap.settings.showTrades = b;
                    else if (k == "showTPO")            snap.settings.showTPO = b;
                    else if (k == "showSettings")       snap.settings.showSettings = b;
                    else if (k == "showStatsOverlay")   snap.settings.showStatsOverlay = b;
                }
            });
        } else if (key == "general" && c == '{') {
            p.parseObject([&](const std::string& k) {
                if (k == "fpsLimit") {
                    double v; if (p.parseNumber(v))
                        snap.settings.fpsLimit = static_cast<long>(v);
                } else if (k == "heatmapDensity") {
                    double v; if (p.parseNumber(v))
                        snap.settings.heatmapDensity = static_cast<long>(v);
                } else if (k == "tradeWindowSeconds") {
                    double v; if (p.parseNumber(v))
                        snap.settings.tradeWindowSeconds = v;
                } else if (k == "theme") {
                    double v; if (p.parseNumber(v))
                        snap.settings.theme = static_cast<long>(v);
                }
            });
        } else if (key == "risk" && c == '{') {
            p.parseObject([&](const std::string& k) {
                double v;
                if (!p.parseNumber(v)) return;
                if      (k == "maxPositionSizeUSD") snap.settings.risk_maxPositionSizeUSD = v;
                else if (k == "maxLeverage")        snap.settings.risk_maxLeverage        = v;
                else if (k == "killOnDailyLossUSD") snap.settings.risk_killOnDailyLossUSD = v;
                else if (k == "equityUSD")          snap.settings.risk_equityUSD          = v;
            });
        } else if (key == "dockLayout" && c == '"') {
            p.parseString(snap.dockLayout);
        } else {
            // Unknown key — consume its value so we can keep parsing.
            if (c == '"') {
                std::string tmp; p.parseString(tmp);
            } else if (c == '{' || c == '[') {
                p.parseObject([&](const std::string&) {});
            } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
                double tmp; p.parseNumber(tmp);
            } else {
                bool tmp; p.parseBool(tmp);
            }
        }
    };

    if (!p.parseObject(onKey))   return std::nullopt;
    if (!haveVersion)            return std::nullopt;
    if (snap.version != LayoutSnapshot::kLayoutVersion) {
        // Future version — refuse rather than mis-apply.
        return std::nullopt;
    }
    return snap;
}

std::vector<std::filesystem::path> LayoutIO::list(
    const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> out;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return out;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".btqlayout") continue;
        out.push_back(entry.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool LayoutIO::exportTo(const std::filesystem::path& destPath,
                        const std::string& name) {
    auto srcPath = layoutPath(name);
    auto snap = load(srcPath);
    if (!snap.has_value()) return false;
    return save(destPath, *snap);
}

std::optional<LayoutSnapshot> LayoutIO::importFrom(
        const std::filesystem::path& srcPath,
        const std::string& destName) {
    auto snap = load(srcPath);
    if (!snap.has_value()) return std::nullopt;
    // Derive the destination name from the file's stem if the caller
    // didn't pass one explicitly. Strips any path components so the
    // profile name matches what list() would display.
    std::string finalName = destName;
    if (finalName.empty()) {
        finalName = srcPath.stem().string();
    }
    // Strip any directory components from a user-supplied name (defence
    // in depth — the caller might pass "/foo/bar" instead of "bar").
    auto slash = finalName.find_last_of("/\\");
    if (slash != std::string::npos) {
        finalName = finalName.substr(slash + 1);
    }
    if (finalName.empty()) return std::nullopt;
    snap->name = finalName;
    auto destPath = layoutPath(finalName);
    if (!save(destPath, *snap)) return std::nullopt;
    return snap;
}

} // namespace btquant::util
