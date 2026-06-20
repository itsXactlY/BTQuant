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
        else if (key == "tradeWindowSeconds" && isDouble)
            s.tradeWindowSeconds = std::stod(val);
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
    out.close();

    std::error_code ec;
    fs::rename(tmp, path, ec);  // best-effort; if rename fails we still have the tmp
}

} // namespace btquant::util
