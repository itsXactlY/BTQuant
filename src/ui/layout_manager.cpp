#include "layout_manager.hpp"

#include <imgui.h>
#include <imgui_internal.h>

namespace UI {

void LayoutManager::disableIniPersistence() {
    // Disable ImGui's automatic .ini file saving/loading for dock configuration
    // This ensures our programmatic layout always takes precedence
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // Disable .ini file output
    
    // Clear any existing docking configuration
    ImGuiDockNode* root_node = ImGui::DockSpaceOverViewport(
        ImGui::GetMainViewport(),
        ImGuiDockNodeFlags_PassthruCentralNode
    );
    
    if (root_node) {
        ImGui::DockBuilderRemoveNode(root_node->ID);
    }
}

void LayoutManager::initializeDockingLayout() {
    if (layout_initialized_) {
        return;
    }

    // Disable .ini persistence to override user configuration
    disableIniPersistence();

    // Build the MMT layout
    buildMMTLayout();

    layout_initialized_ = true;
}

void LayoutManager::buildMMTLayout() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGuiID root_id = ImGui::GetID("MMT_Root_DockSpace");

    // Remove existing dock node configuration
    ImGuiDockNode* existing_node = ImGui::DockBuilderGetNode(root_id);
    if (existing_node) {
        ImGui::DockBuilderRemoveNode(root_id);
    }

    // Create root dock node
    ImGui::DockBuilderAddNode(root_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(root_id, viewport->Size);

    // Define dock node IDs for the 5-region MMT layout
    ImGuiID dock_tools_left   = ImGui::DockBuilderSplitNode(root_id, ImGuiDir_Left,  0.03f, nullptr, &root_id);
    ImGuiID dock_charts       = ImGui::DockBuilderSplitNode(root_id, ImGuiDir_Up,    0.70f, &root_id, &root_id);
    ImGuiID dock_right_top    = ImGui::DockBuilderSplitNode(root_id, ImGuiDir_Up,    0.50f, &root_id, &root_id);
    ImGuiID dock_right_bottom = root_id;

    // Assign windows to dock nodes
    ImGui::DockBuilderDockWindow("Tools", dock_tools_left);
    ImGui::DockBuilderDockWindow("Charts", dock_charts);
    ImGui::DockBuilderDockWindow("DOM", dock_right_top);
    ImGui::DockBuilderDockWindow("OrderBook", dock_right_top);
    ImGui::DockBuilderDockWindow("Tape", dock_right_bottom);

    // Finalize the docking configuration
    ImGui::DockBuilderFinish(root_id);
}

void LayoutManager::resetToMMTDefaults() {
    layout_initialized_ = false;
    
    // Clear all existing dock nodes
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGuiID root_id = ImGui::GetID("MMT_Root_DockSpace");
    
    ImGuiDockNode* root_node = ImGui::DockBuilderGetNode(root_id);
    if (root_node) {
        ImGui::DockBuilderRemoveNode(root_id);
    }
    
    // Reinitialize the layout
    buildMMTLayout();
    layout_initialized_ = true;
}

} // namespace UI
