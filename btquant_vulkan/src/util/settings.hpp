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

    // Settings dialog visibility.
    bool showSettings = false;

    // Stats overlay visibility (FPS / frame-time in top-right corner).
    bool showStatsOverlay = true;

    // General.
    long fpsLimit = 60;          // 0 = uncapped (glfwSwapInterval 0)
    long heatmapDensity = 128;   // resolution of GPU heatmap texture
    double tradeWindowSeconds = 60.0;

    // Theme: 0 = Dark (Kraken Purple), 1 = Light (off-white + blue).
    // String "dark"/"light" also accepted for human readability.
    long theme = 0;

    // Risk limits — persisted so RiskLimitsPanel edits survive restart.
    // Defaults mirror RiskConfig's built-in defaults so a fresh install
    // loads the same values either way.
    double risk_maxPositionSizeUSD = 100000.0;
    double risk_maxLeverage        = 10.0;
    double risk_killOnDailyLossUSD = 5000.0;
    double risk_equityUSD          = 10000.0;

    // Resolve the canonical config path (~/.config/btquant_vulkan/state.ini by
    // default, overridable via BTQUANT_CONFIG env var).
    static std::filesystem::path defaultPath();

    // Load from file. Missing file → defaults. Malformed lines → silently
    // skipped (best-effort forward compatibility).
    static Settings load(const std::filesystem::path& path);

    // Save current state. Creates parent directories as needed. Always
    // overwrites the file — atomic rename to avoid corruption on crash.
    void save(const std::filesystem::path& path) const;

    // ----- Named profiles (saved/loaded independently from the main settings)
    // A profile captures widget visibility + theme + heatmap density so the
    // user can switch between named layouts (e.g. "Scalper", "Market Maker")
    // without manually toggling each widget. Each profile lives in its own
    // file under ~/.config/btquant_vulkan/profiles/<name>.ini.
    static std::filesystem::path profilePath(const std::string& name);

    // Apply this profile to the supplied bool refs / longs. Used by the
    // View → Profiles menu to switch the workspace in one click.
    void applyTo(bool& showOB, bool& showOBD, bool& showFootprint, bool& showVPVR,
                 bool& showMVWAP, bool& showRisk, bool& showDOM, bool& showTrades,
                 bool& showTPO, long& themeOut, long& densityOut) const;

    // Static factories for the built-in presets (returned by value).
    static Settings presetScalper();
    static Settings presetMarketMaker();
    static Settings presetVolatility();
    static Settings presetFullscreen();
};

} // namespace btquant::util

#endif
