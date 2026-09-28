#ifndef BTQUANT_LAYOUT_IO_HPP
#define BTQUANT_LAYOUT_IO_HPP

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "settings.hpp"

namespace btquant::util {

// A versioned snapshot of the entire layout-relevant state: widget
// visibility, general settings, risk config, and the ImGui dock-layout
// text (from ImGui::SaveDockBuilderToText). Saved to .btqlayout files
// under ~/.config/btquant_vulkan/profiles/.
//
// Version field lets future changes bump the format without breaking
// old files — load() rejects unknown versions explicitly.
struct LayoutSnapshot {
    Settings settings;        // covers widget visibility + risk_* + general
    std::string dockLayout;   // empty → caller leaves dock alone
    int version = kLayoutVersion;
    std::string name;         // human-readable, set by save() callers

    static constexpr int kLayoutVersion = 1;
};

// Layout IO — minimal hand-rolled JSON (no external deps). Writes are
// pretty-printed; loads accept whitespace-tolerant input.
class LayoutIO {
public:
    // Resolve ~/.config/btquant_vulkan/profiles/<name>.btqlayout with
    // the same sanitization as profilePath(). Creates the parent dir
    // if missing on save.
    static std::filesystem::path layoutPath(const std::string& name);
    static std::filesystem::path layoutDir();

    // Write / read round-trip. Returns true on successful write,
    // std::nullopt on missing / malformed / unsupported-version files.
    // load() does NOT mutate dockLayout on parse failure — caller
    // decides what to do with a half-loaded snapshot.
    static bool                        save(const std::filesystem::path& path,
                                            const LayoutSnapshot& snap);
    static std::optional<LayoutSnapshot> load(const std::filesystem::path& path);

    // Export a named profile to an arbitrary path (e.g. for sharing
    // between machines via scp/USB). Reads the named profile and writes
    // it verbatim — same on-disk format, same version, same metadata.
    // Returns true on success; false if the source profile doesn't
    // exist or the destination can't be opened for writing.
    static bool exportTo(const std::filesystem::path& destPath,
                         const std::string& name);

    // Import a .btqlayout file from an arbitrary path. Loads it,
    // rewrites the name field to match the destination filename (so the
    // file lists cleanly under profiles/), and saves it under the
    // default layout dir. Returns true on success; nullopt on
    // load failure. `destName` is derived from destPath's stem if empty.
    static std::optional<LayoutSnapshot> importFrom(
        const std::filesystem::path& srcPath,
        const std::string& destName = "");

    // Build a snapshot from live state. Pure helper — caller supplies
    // the dockLayout string (typically from
    // ImGui::SaveDockBuilderToText(window->DockNode)).
    static LayoutSnapshot fromSettings(const Settings& s,
                                       const std::string& dockLayout,
                                       const std::string& name = "");

    // List .btqlayout files in the default dir. Returned as paths
    // sorted by name; caller filters / displays.
    static std::vector<std::filesystem::path>
        list(const std::filesystem::path& dir = layoutDir());
};

} // namespace btquant::util

#endif
