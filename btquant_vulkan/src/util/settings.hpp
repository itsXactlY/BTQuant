#ifndef BTQUANT_SETTINGS_HPP
#define BTQUANT_SETTINGS_HPP

#include <filesystem>
#include <string>

namespace btquant::util {

// Hand-rolled key=value settings file. Avoids pulling in nlohmann/json for what
// is essentially a flat map of bools/ints/doubles.
//
// Format (lines starting with '#' are comments, blank lines are ignored):
//
//   # btquant_vulkan state
//   showOrderBook=1
//   showOrderBookDepth=1
//   fpsLimit=60
//   heatmapDensity=128
//
// Type is inferred from the literal: "0"/"1" → bool, anything parseable as
// double → double, anything parseable as long → long, otherwise → string.
struct Settings {
    // Widget visibility (one per show* window in WindowManager).
    bool showOrderBook = true;
    bool showOrderBookDepth = true;
    bool showFootprint = true;
    bool showVPVR = true;
    bool showMultiVWAP = true;
    bool showRiskPanel = true;
    bool showDOM = true;
    bool showTrades = true;
    bool showTPO = true;

    // General.
    long fpsLimit = 60;          // 0 = uncapped (glfwSwapInterval 0)
    long heatmapDensity = 128;   // resolution of GPU heatmap texture
    double tradeWindowSeconds = 60.0;

    // Resolve the canonical config path (~/.config/btquant_vulkan/state.ini by
    // default, overridable via BTQUANT_CONFIG env var).
    static std::filesystem::path defaultPath();

    // Load from file. Missing file → defaults. Malformed lines → silently
    // skipped (best-effort forward compatibility).
    static Settings load(const std::filesystem::path& path);

    // Save current state. Creates parent directories as needed. Always
    // overwrites the file — atomic rename to avoid corruption on crash.
    void save(const std::filesystem::path& path) const;
};

} // namespace btquant::util

#endif
