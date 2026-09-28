#include "theme_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace btquant::ui {

std::filesystem::path ThemeIO::defaultPath() {
    // Same env override as Settings::defaultPath().
    if (const char* env = std::getenv("BTQUANT_CONFIG")) {
        std::filesystem::path p(env);
        return p.parent_path() / "theme.ini";
    }
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME")) {
        return std::filesystem::path(xdg) / "btquant_vulkan" / "theme.ini";
    }
    return std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME") : ".")
           / ".config" / "btquant_vulkan" / "theme.ini";
}

std::optional<ThemeEditor::Snapshot> ThemeIO::load(
    const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;

    ThemeEditor::Snapshot s;
    // Default-init all floats to 0 — if file has them, they get overwritten.
    // For colors, we use a "seen" flag per slot so missing colors stay at 0.
    std::vector<std::array<bool, 4>> seen(ThemeEditor::kColorCount,
                                          {false, false, false, false});

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        try {
            // Format: c<i>.<k>=<float>. Split on '.', 3 parts.
            if (key.size() >= 3 && key[0] == 'c' && key.find('.') != std::string::npos) {
                auto dot = key.find('.');
                int idx, ch;
                try {
                    idx = std::stoi(key.substr(1, dot - 1));
                    ch  = std::stoi(key.substr(dot + 1));
                } catch (...) { continue; }
                float v = std::stof(val);
                if (idx >= 0 && idx < ThemeEditor::kColorCount &&
                    ch  >= 0 && ch  < 4) {
                    s.colors[idx][ch] = v;
                    seen[idx][ch] = true;
                }
            } else if (key == "windowPadding") {
                s.windowPadding = std::stof(val);
            } else if (key == "framePadding") {
                s.framePadding = std::stof(val);
            } else if (key == "rounding") {
                s.rounding = std::stof(val);
            } else if (key == "alpha") {
                s.alpha = std::stof(val);
            } else if (key == "dark") {
                s.dark = (std::stoi(val) != 0);
            }
        } catch (...) {
            // Malformed line — skip.
            continue;
        }
    }

    return s;
}

bool ThemeIO::save(const std::filesystem::path& path,
                   const ThemeEditor::Snapshot& s) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) return false;

    fs::path tmp = path;
    tmp += ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) return false;

    out << "# btquant_vulkan theme snapshot — auto-generated\n";
    out << std::fixed << std::setprecision(6);
    // Format: c<i>.<k>=<float> — unambiguous: single-letter prefix, period
    // separator (can't be confused with digits in the index itself).
    for (int i = 0; i < ThemeEditor::kColorCount; ++i) {
        for (int k = 0; k < 4; ++k) {
            out << "c" << i << "." << k << "=" << s.colors[i][k] << "\n";
        }
    }
    out << "windowPadding=" << s.windowPadding << "\n";
    out << "framePadding="  << s.framePadding  << "\n";
    out << "rounding="      << s.rounding      << "\n";
    out << "alpha="         << s.alpha         << "\n";
    out << "dark="          << (s.dark ? 1 : 0) << "\n";
    out.close();
    if (!out.good()) return false;

    ec.clear();
    fs::rename(tmp, path, ec);
    return !ec;
}

} // namespace btquant::ui
