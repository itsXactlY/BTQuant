#ifndef BTQUANT_PROFILE_MANAGER_HPP
#define BTQUANT_PROFILE_MANAGER_HPP

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace btquant::util { class Settings; }

namespace btquant::ui {

// Profile Manager window — list / save / delete / reload named layout
// profiles from ~/.config/btquant_vulkan/profiles/. Each profile is an INI
// file containing a snapshot of the WindowManager's settings (widget
// visibility, theme, heatmap density).
class ProfileManager {
public:
    using CaptureFn = std::function<util::Settings()>;
    using ApplyFn   = std::function<void(const std::string& profileName)>;

    void render();

    // Trigger from menu / hotkey.
    void setOpen(bool v) { m_open = v; }
    bool isOpen() const   { return m_open; }

    // External hooks.
    void setCaptureFn(CaptureFn fn) { m_capture = std::move(fn); }
    void setApplyFn  (ApplyFn   fn) { m_apply   = std::move(fn); }

    // Test accessors.
    const std::vector<std::filesystem::path>& profiles() const { return m_profiles; }
    void setProfilesDir(const std::filesystem::path& p) { m_profilesDir = p; refresh(); }

    bool saveCurrentAs(const std::string& name);
    bool deleteProfile(const std::filesystem::path& path);
    void refresh();

private:
    bool m_open = false;
    std::filesystem::path m_profilesDir;
    std::vector<std::filesystem::path> m_profiles;
    char m_newName[128] = "";
    std::string m_status;

    CaptureFn m_capture;
    ApplyFn   m_apply;
};

} // namespace btquant::ui

#endif
