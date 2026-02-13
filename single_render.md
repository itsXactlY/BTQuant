# TASK_SINGLE_RENDER.md: The "One Brain" Protocol

**Objective:** Eliminate duplicate render passes and duplicate panel instantiation. Establish a strict, single-path rendering hierarchy: `VulkanDashboard` -> `QuantWorkspaceComponent` -> `PanelManager` -> `Panels`.

---

## Phase 1: The Render Loop Correction
**Goal:** Stop the main frame loop from calling the Panel Manager twice.

- [x] **1.1: Cleanse `VulkanDashboard::render_frame()`**
    - **File:** `src/vulkan_dashboard_advanced.cpp`
    - **Locate:** The `render_frame()` function, specifically the section where ImGui rendering is dispatched.
    - **Action:** Find the line calling `panel_manager_->render_panels()` (or `panel_manager_->render()`). **DELETE IT.**
    - **Action:** Ensure that the ONLY top-level UI call in this loop is `workspace_->render_gui()`.
    - *Why:* `QuantWorkspaceComponent` owns the Main DockSpace. It is the Workspace's job to iterate through the `PanelManager` and render the panels *inside* the dockspace. Calling the PanelManager directly from the Dashboard draws a second, un-docked copy of every panel.

- [x] **1.2: Verify Workspace Render Logic**
    - **File:** `src/components/quant_workspace_component.cpp`
    - **Locate:** `render_gui()`
    - **Action:** Ensure that this function sets up the DockSpace (`ImGui::DockSpaceOverViewport`), renders the Top Menu Bar, and then calls `panel_manager_->render()`. This must be the *only* place `panel_manager_->render()` is invoked.

---

## Phase 2: The Instantiation Correction
**Goal:** Stop the application from creating the default panels three times on startup.

- [x] **2.1: Purge Hardcoded Startup Panels**
    - **File:** `src/main_trading_terminal.cpp`
    - **Locate:** Inside `main()`, look for any leftover `std::make_shared<ChartPanel>(...)` or `panel_manager->add_panel(...)` logic.
    - **Action:** **DELETE** all manual panel creation.
    - *Why:* The layout system should completely dictate what gets created.

- [x] **2.2: Ensure Clean Preset Loading**
    - **File:** `src/components/panel_manager.cpp`
    - **Locate:** `apply_layout_preset(LayoutPreset preset)`
    - **Action:** The very first line of this function MUST be `remove_all_panels();` or `panels_.clear();`.
    - *Why:* If you apply a layout on top of an existing layout without clearing the vectors/maps first, you end up with 3 Chart Panels, 3 Orderbooks, etc., in memory.

- [x] **2.3: Prevent Constructor Duplication**
    - **File:** `src/components/panel_manager.cpp`
    - **Locate:** The constructor `PanelManager::PanelManager(...)`.
    - **Action:** Do NOT call `apply_layout_preset` or add default panels in the constructor. Leave the list empty. Let `main_trading_terminal.cpp` trigger the initial state explicitly by calling `workspace->set_layout(...)` right before the main loop starts.

---

## Phase 3: The Microstructure Relic
**Goal:** Ensure the legacy `MarketMicrostructureRenderer` is truly dead in the Vulkan pipeline.

- [x] **3.1: Final Vulkan Cleanup**
    - **File:** `src/vulkan_dashboard_advanced.cpp`
    - **Locate:** `render_frame()` and Vulkan command buffer recording.
    - **Action:** Verify there are no lingering calls to `micro_renderer_->render(...)` outside of ImGui. If the old monolithic renderer is still drawing to the framebuffer beneath ImGui, it will create a visual echo of the market data.
    - *Why:* The new architecture uses ImGui-native rendering (e.g., `ImPlot`, custom `ImDrawList` calls inside the panel classes). The legacy Vulkan-native renderer must not run concurrently.