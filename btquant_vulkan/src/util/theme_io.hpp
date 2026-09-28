#ifndef BTQUANT_THEME_IO_HPP
#define BTQUANT_THEME_IO_HPP

#include <filesystem>
#include <optional>

#include "../widgets/theme_editor.hpp"

namespace btquant::ui {

// Theme persistence — reads/writes ThemeEditor::Snapshot to a small INI file
// sitting next to the main settings (~/.config/btquant_vulkan/theme.ini).
// Format is `themeColor_<idx>_<channel>=<float>` plus 4 floats for the
// style knobs, so it's diff-friendly and human-editable.
class ThemeIO {
public:
    // Canonical path: same dir as settings.ini, file name "theme.ini".
    static std::filesystem::path defaultPath();

    // Read snapshot from disk. Missing file → nullopt. Malformed lines
    // silently skipped (forward compat).
    static std::optional<ThemeEditor::Snapshot> load(
        const std::filesystem::path& path);

    // Write snapshot atomically (tmp + rename). Creates parent dirs.
    static bool save(const std::filesystem::path& path,
                     const ThemeEditor::Snapshot& s);
};

} // namespace btquant::ui

#endif
