#pragma once

#include <imgui.h>

namespace UI {

/**
 * @brief Layout Manager for MMT (Market Monkey Terminal) docking configuration
 * 
 * Overrides user .ini files and programmatically creates a 5-region dock layout:
 * - Tools Left (3%)
 * - Charts Center
 * - DOM/OB Right-Top
 * - Tape Right-Bottom
 */
class LayoutManager {
public:
    // Singleton access
    static LayoutManager& getInstance() {
        static LayoutManager instance;
        return instance;
    }

    // Delete copy and move constructors
    LayoutManager(const LayoutManager&) = delete;
    LayoutManager& operator=(const LayoutManager&) = delete;
    LayoutManager(LayoutManager&&) = delete;
    LayoutManager& operator=(LayoutManager&&) = delete;

    /**
     * @brief Initialize the MMT docking layout
     * Must be called after ImGui::NewFrame() and before any dockable windows
     */
    void initializeDockingLayout();

    /**
     * @brief Build the 5-region MMT layout programmatically
     * Overrides any saved .ini configuration
     */
    void buildMMTLayout();

    /**
     * @brief Check if docking layout has been initialized
     */
    bool isLayoutInitialized() const { return layout_initialized_; }

    /**
     * @brief Reset the docking layout to MMT defaults
     */
    void resetToMMTDefaults();

private:
    LayoutManager() = default;
    ~LayoutManager() = default;

    bool layout_initialized_ = false;

    /**
     * @brief Disable ImGui .ini file persistence for dock nodes
     */
    void disableIniPersistence();
};

} // namespace UI
