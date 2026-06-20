#include "settings.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

namespace btquant::util {

namespace {

bool parseBool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
}

// True iff every char is a digit / '.' / '-' / '+' / 'e' / 'E' AND the
// string has at least one digit — used to guard std::stod against
// throwing on clearly non-numeric values like "not_a_number".
bool looksLikeDouble(const std::string& v) {
    if (v.empty()) return false;
    bool hasDigit = false;
    for (char c : v) {
        if (std::isdigit((unsigned char)c)) hasDigit = true;
        else if (c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E')
            return false;
    }
    return hasDigit;
}

} // namespace

std::filesystem::path Settings::defaultPath() {
    if (const char* env = std::getenv("BTQUANT_CONFIG")) {
        return std::filesystem::path(env);
    }
    const char* home = std::getenv("HOME");
    std::filesystem::path base = home ? std::filesystem::path(home)
                                     : std::filesystem::current_path();
    return base / ".config" / "btquant_vulkan" / "state.ini";
}

Settings Settings::load(const std::filesystem::path& path) {
    Settings s;
    std::ifstream in(path);
    if (!in) return s;  // missing file is fine — defaults.

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        // Trim whitespace.
        auto trim = [](std::string& t) {
            while (!t.empty() && std::isspace((unsigned char)t.front())) t.erase(t.begin());
            while (!t.empty() && std::isspace((unsigned char)t.back())) t.pop_back();
        };
        trim(key);
        trim(val);

        // Type inference.
        bool isBool = (val == "0" || val == "1" || val == "true" || val == "false");
        bool isInt = !isBool;
        if (isInt) {
            for (char c : val) if (!std::isdigit((unsigned char)c) && c != '-') { isInt = false; break; }
        }
        bool isDouble = !isBool && !isInt;

        if (key == "showOrderBook")        s.showOrderBook = parseBool(val);
        else if (key == "showOrderBookDepth") s.showOrderBookDepth = parseBool(val);
        else if (key == "showFootprint")    s.showFootprint = parseBool(val);
        else if (key == "showVPVR")         s.showVPVR = parseBool(val);
        else if (key == "showMultiVWAP")    s.showMultiVWAP = parseBool(val);
        else if (key == "showRiskPanel")    s.showRiskPanel = parseBool(val);
        else if (key == "showDOM")          s.showDOM = parseBool(val);
        else if (key == "showTrades")       s.showTrades = parseBool(val);
        else if (key == "showTPO")          s.showTPO = parseBool(val);
        else if (key == "showSettings")     s.showSettings = parseBool(val);
        else if (key == "showStatsOverlay") s.showStatsOverlay = parseBool(val);
        else if (key == "fpsLimit" && isInt)          s.fpsLimit = std::stol(val);
        else if (key == "heatmapDensity" && isInt)    s.heatmapDensity = std::stol(val);
        else if (key == "tradeWindowSeconds" && looksLikeDouble(val))
            s.tradeWindowSeconds = std::stod(val);
        else if (key == "theme") {
            if (val == "dark" || val == "0") s.theme = 0;
            else if (val == "light" || val == "1") s.theme = 1;
            else if (isInt) s.theme = std::stol(val);
        }
        else if (key == "risk_maxPositionSizeUSD" && looksLikeDouble(val))
            s.risk_maxPositionSizeUSD = std::stod(val);
        else if (key == "risk_maxLeverage" && looksLikeDouble(val))
            s.risk_maxLeverage = std::stod(val);
        else if (key == "risk_killOnDailyLossUSD" && looksLikeDouble(val))
            s.risk_killOnDailyLossUSD = std::stod(val);
        else if (key == "risk_equityUSD" && looksLikeDouble(val))
            s.risk_equityUSD = std::stod(val);
    }
    return s;
}

void Settings::save(const std::filesystem::path& path) const {
    namespace fs = std::filesystem;
    fs::create_directories(path.parent_path());

    // Atomic write: tmp + rename.
    fs::path tmp = path;
    tmp += ".tmp";

    std::ofstream out(tmp, std::ios::trunc);
    out << "# btquant_vulkan state — auto-generated, do not edit by hand\n";
    out << "showOrderBook=" << (showOrderBook ? 1 : 0) << "\n";
    out << "showOrderBookDepth=" << (showOrderBookDepth ? 1 : 0) << "\n";
    out << "showFootprint=" << (showFootprint ? 1 : 0) << "\n";
    out << "showVPVR=" << (showVPVR ? 1 : 0) << "\n";
    out << "showMultiVWAP=" << (showMultiVWAP ? 1 : 0) << "\n";
    out << "showRiskPanel=" << (showRiskPanel ? 1 : 0) << "\n";
    out << "showDOM=" << (showDOM ? 1 : 0) << "\n";
    out << "showTrades=" << (showTrades ? 1 : 0) << "\n";
    out << "showTPO=" << (showTPO ? 1 : 0) << "\n";
    out << "showSettings=" << (showSettings ? 1 : 0) << "\n";
    out << "showStatsOverlay=" << (showStatsOverlay ? 1 : 0) << "\n";
    out << "fpsLimit=" << fpsLimit << "\n";
    out << "heatmapDensity=" << heatmapDensity << "\n";
    // Fixed notation guarantees the dot is always present, so the parser's
    // isDouble heuristic (looks for '.') matches on roundtrip.
    out << "tradeWindowSeconds=" << std::fixed << std::setprecision(6)
        << tradeWindowSeconds << "\n";
    out << "theme=" << (theme == 0 ? "dark" : "light") << "\n";
    out << "# Risk limits (editable via Risk Dashboard panel)\n";
    out << "risk_maxPositionSizeUSD=" << std::fixed << std::setprecision(2)
        << risk_maxPositionSizeUSD << "\n";
    out << "risk_maxLeverage=" << std::fixed << std::setprecision(4)
        << risk_maxLeverage << "\n";
    out << "risk_killOnDailyLossUSD=" << std::fixed << std::setprecision(2)
        << risk_killOnDailyLossUSD << "\n";
    out << "risk_equityUSD=" << std::fixed << std::setprecision(2)
        << risk_equityUSD << "\n";
    out.close();

    std::error_code ec;
    fs::rename(tmp, path, ec);  // best-effort; if rename fails we still have the tmp
}

std::filesystem::path Settings::profilePath(const std::string& name) {
    // Sanitize: only [a-zA-Z0-9_-] allowed, max 64 chars. Stops path traversal.
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
    return defaultPath().parent_path() / "profiles" / (clean + ".ini");
}

void Settings::applyTo(bool& showOB, bool& showOBD, bool& showFootprint,
                       bool& showVPVR, bool& showMVWAP, bool& showRisk,
                       bool& showDOM, bool& showTrades, bool& showTPO,
                       long& themeOut, long& densityOut) const {
    // Use this-> to disambiguate from the parameter names (otherwise
    // `showFootprint = showFootprint` would be a self-assign).
    showOB       = this->showOrderBook;
    showOBD      = this->showOrderBookDepth;
    showFootprint = this->showFootprint;
    showVPVR     = this->showVPVR;
    showMVWAP    = this->showMultiVWAP;
    showRisk     = this->showRiskPanel;
    showDOM      = this->showDOM;
    showTrades   = this->showTrades;
    showTPO      = this->showTPO;
    themeOut     = this->theme;
    densityOut   = this->heatmapDensity;
}

Settings Settings::presetScalper() {
    Settings s;
    s.showOrderBook = false;
    s.showOrderBookDepth = false;
    s.showFootprint = true;
    s.showVPVR = false;
    s.showMultiVWAP = true;
    s.showRiskPanel = true;
    s.showDOM = true;       // order-flow scalper — DOM is king
    s.showTrades = true;
    s.showTPO = false;
    s.theme = 0;
    s.heatmapDensity = 256;
    return s;
}

Settings Settings::presetMarketMaker() {
    Settings s;
    s.showOrderBook = true;
    s.showOrderBookDepth = true;
    s.showFootprint = false;
    s.showVPVR = false;
    s.showMultiVWAP = false;
    s.showRiskPanel = true;
    s.showDOM = true;
    s.showTrades = false;
    s.showTPO = false;
    s.theme = 0;
    s.heatmapDensity = 128;
    return s;
}

Settings Settings::presetVolatility() {
    Settings s;
    s.showOrderBook = false;
    s.showOrderBookDepth = false;
    s.showFootprint = true;
    s.showVPVR = true;
    s.showMultiVWAP = true;
    s.showRiskPanel = true;
    s.showDOM = false;
    s.showTrades = true;
    s.showTPO = true;
    s.theme = 0;
    s.heatmapDensity = 384;
    return s;
}

Settings Settings::presetFullscreen() {
    Settings s;
    s.showOrderBook = true;
    s.showOrderBookDepth = true;
    s.showFootprint = true;
    s.showVPVR = true;
    s.showMultiVWAP = true;
    s.showRiskPanel = true;
    s.showDOM = true;
    s.showTrades = true;
    s.showTPO = true;
    s.theme = 0;
    s.heatmapDensity = 512;
    return s;
}

} // namespace btquant::util
