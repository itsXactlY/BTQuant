#include "profile_manager.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <imgui.h>

#include "../util/settings.hpp"

namespace btquant::ui {

namespace {
std::string pathBasename(const std::filesystem::path& p) {
    auto s = p.stem().string();
    return s;
}
std::string formatMtime(const std::filesystem::path& p) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(p, ec);
    if (ec) return "?";
    // Convert fs time to system_clock for ctime.
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        t - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
    auto tt = std::chrono::system_clock::to_time_t(sctp);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", std::localtime(&tt));
    return buf;
}
std::string formatSize(uintmax_t sz) {
    char buf[32];
    if (sz < 1024) std::snprintf(buf, sizeof(buf), "%lu B", (unsigned long)sz);
    else            std::snprintf(buf, sizeof(buf), "%.1f KB", sz / 1024.0);
    return buf;
}
} // namespace

void ProfileManager::render() {
    if (!m_open) return;
    if (!ImGui::Begin("Profile Manager", &m_open)) {
        ImGui::End();
        return;
    }

    if (m_profilesDir.empty()) {
        m_profilesDir = util::Settings::profilePath("").parent_path();
        refresh();
    }

    ImGui::Text("Directory: %s", m_profilesDir.string().c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Reload")) refresh();

    ImGui::Separator();

    // Save current as new profile.
    ImGui::Text("Save current layout as:");
    ImGui::PushItemWidth(220);
    ImGui::InputText("##newname", m_newName, sizeof(m_newName));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
        std::string name(m_newName);
        if (name.empty()) {
            m_status = "name is empty";
        } else if (saveCurrentAs(name)) {
            m_status = "saved \"" + name + "\"";
            std::memset(m_newName, 0, sizeof(m_newName));
            refresh();
        } else {
            m_status = "save failed (bad name?)";
        }
    }
    if (!m_status.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.30f, 1.0f), "%s", m_status.c_str());
    }

    ImGui::Separator();

    // Existing profiles table.
    if (ImGui::BeginTable("profiles", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Name",       ImGuiTableColumnFlags_WidthFixed, 160.0f);
        ImGui::TableSetupColumn("Modified",   ImGuiTableColumnFlags_WidthFixed, 140.0f);
        ImGui::TableSetupColumn("Size",       ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Actions",    ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (const auto& p : m_profiles) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(pathBasename(p).c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(formatMtime(p).c_str());

            ImGui::TableNextColumn();
            std::error_code ec;
            auto sz = std::filesystem::file_size(p, ec);
            ImGui::TextUnformatted(ec ? "?" : formatSize(sz).c_str());

            ImGui::TableNextColumn();
            std::string nm = pathBasename(p);
            ImGui::PushID(nm.c_str());
            if (ImGui::SmallButton("Load")) {
                if (m_apply) m_apply(nm);
                m_status = "loaded \"" + nm + "\"";
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete")) {
                if (deleteProfile(p)) {
                    m_status = "deleted \"" + nm + "\"";
                    refresh();
                } else {
                    m_status = "delete failed";
                }
            }
            ImGui::PopID();
        }

        ImGui::EndTable();
    }

    ImGui::End();
}

void ProfileManager::refresh() {
    m_profiles.clear();
    if (m_profilesDir.empty() || !std::filesystem::is_directory(m_profilesDir)) return;
    std::error_code ec;
    for (auto& entry : std::filesystem::directory_iterator(m_profilesDir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".ini") continue;
        m_profiles.push_back(entry.path());
    }
    std::sort(m_profiles.begin(), m_profiles.end(),
              [](const auto& a, const auto& b) { return a < b; });
}

bool ProfileManager::saveCurrentAs(const std::string& name) {
    // Path-traversal protection — reject anything that would resolve outside
    // the profiles dir (matches Settings::profilePath's policy).
    if (name.empty()) return false;
    if (name.find('/') != std::string::npos)  return false;
    if (name.find('\\') != std::string::npos) return false;
    if (name.find("..") != std::string::npos) return false;
    if (name.find('\0') != std::string::npos) return false;

    auto path = m_profilesDir / (name + ".ini");
    if (!m_capture) return false;
    auto snap = m_capture();
    snap.save(path);
    return true;
}

bool ProfileManager::deleteProfile(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::remove(path, ec);
}

} // namespace btquant::ui
