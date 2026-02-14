# TASK_ULTIMA_MMT_CLONE_MAPPING.md

**Objective:** Transform BTQ Terminal into a 144Hz institutional Order Flow platform by mapping MMT features to the existing C++26 decoupled architecture.
**Core Principle:** Data is pulled from `MarketDataProcessor` atomics; UI is composed via `LayoutManager` DockBuilder; Heavy visuals use `VulkanCore` textures.

---

## Phase 1: The Docking Matrix (Workspace Composition)
**Target:** `src/ui/layout_manager.cpp` & `src/components/quant_workspace_component.cpp`

- [x] **1.1: MMT programmatic Grid Construction**
    - Inside `LayoutManager::apply_layout_preset(MODERN_TRADING)`:
    - [ ] Create central `ChartSuperNode`.
    - [ ] Split Left (3%) -> `drawing_tools_panel`.
    - [ ] Split Right (25%) -> `dom_surface_panel` + `orderbook_panel`.
    - [ ] Split Right-Bottom (40%) -> `time_and_sales` (Trades).
    - [ ] Split Center-Bottom (15%) -> `time_histogram_panel`.
- [ ] **1.2: Seamless Visual Blending**
    - [ ] Globally set `ImGuiStyleVar_WindowPadding` to `(0,0)` in `QuantWorkspaceComponent::render_gui`.
    - [ ] Set `ImGuiCol_Separator` to `rgba(94, 82, 64, 0.2)` in `unified_theme_system.cpp`.

---

## Phase 2: The GPU Heatmap Spine (LOB Engine)
**Target:** `src/components/dom_surface_panel.cpp` & `shaders/lob_heatmap.comp`

- [ ] **2.1: Lock-Free Snapshot Pipeline**
    - [ ] In `MarketDataProcessor`, implement a ring buffer of `OrderBookSnapshot` structs (Standard Layout POD).
    - [ ] Use `std::atomic<uint64_t> snapshot_head` to signal `VulkanCore` that new data is ready for the Compute Shader.
- [ ] **2.2: Compute-to-ImGui Bind**
    - [ ] Update `DomSurfacePanel` to retrieve the `VkImage` descriptor from `GPUMemoryManager`.
    - [ ] Use `ImGui_ImplVulkan_AddTexture` to map the heatmap to an `ImTextureID`.
    - [ ] Render via `ImGui::GetWindowDrawList()->AddImage()` spanning the panel background.
- [ ] **2.3: Multi-Exchange Aggregation**
    - [ ] Add "Aggregated Heatmap" options in the right-click context menu of the DOM header.
    - [ ] Update the Compute Shader to sum multiple atomic depth buffers before colormapping.

---

## Phase 3: The DOM & Market Volume Profiles
**Target:** `src/components/dom_surface_panel.cpp` & `src/analytics/cluster_engine.cpp`

- [ ] **3.1: 5-Column MMT Layout**
    - [ ] Render Table: `[Buys | Asks | Price | Bids | Sells]`.
    - [ ] **Center Mode:** Mathematically lock Y-limits: `y_min = current_price - range`.
- [ ] **3.2: Cumulative Volume Columns**
    - [ ] Pull `VolumeData` from `ClusterEngine` without mutexes.
    - [ ] Render horizontal bars using `DrawList->AddRectFilled`.
    - [ ] Green bars (Buys) extend right; Red bars (Sells) extend left.

---

## Phase 4: The Order Book Widget (Liquidity Ladder)
**Target:** `src/components/orderbook_panel.cpp`

- [ ] **4.1: Atomic Unit Toggles**
    - [ ] Implement USD / COIN toggle in header.
    - [ ] If USD: Multiply `atomic_size` by `atomic_last_price` during render pass.
- [ ] **4.2: Depth Rendering**
    - [ ] Render Asks (Red) descending from top; Bids (Green) ascending from bottom.
    - [ ] Read from `MarketDataProcessor::get_atomic_snapshot` array.

---

## Phase 5: The Trades Feed (Acoustic Order Flow)
**Target:** `src/components/time_and_sales.cpp` & `src/trading/trade_command_queue.cpp`

- [ ] **5.1: Raw Trade Table**
    - [ ] Columns: `[Exchange Logo] | Price | Qty | Time`.
    - [ ] Render exchange icons from the pre-loaded texture atlas in `GPUMemoryManager`.
- [ ] **5.2: Order Flow Acoustics**
    - [ ] Integrate `miniaudio` into the `MarketDataProcessor` poll loop.
    - [ ] Trigger `ma_engine_play_sound` on new trade detection.
    - [ ] Pitch Shift: Scale pitch inversely to volume (Big trade = Deep bass).

---

## Phase 6: Chart Anatomy & Toolbar
**Target:** `src/components/chart_panel.cpp`

- [ ] **6.1: The Flush Top Toolbar**
    - [ ] Render `BeginChild("Toolbar", (0, 32))` with `NoScrollbar`.
    - [ ] Wire `Symbol Input` to `MarketDataProcessor::subscribe(new_symbol)`.
- [ ] **6.2: Quick Order Sidebar**
    - [ ] If `Mouse Trading` enabled: Render massive `BUY MKT` / `SELL MKT` buttons.
    - [ ] Display live `best_bid` / `best_ask` from atomics inside the button text.
- [ ] **6.3: Trade Command Routing**
    - [ ] Button Click -> Populate `TradeCommand` POD -> `trade_command_queue_.push()`.

---

## Phase 7: Sub-Pixel Polish & Sync
**Target:** `src/ui/font_manager.cpp` & `src/components/quant_workspace_component.cpp`

- [ ] **7.1: High-DPI Typography**
    - [ ] Load *JetBrains Mono* with `OversampleH = 4`.
    - [ ] Ensure all numeric data uses the monospaced atlas for perfect column alignment.
- [ ] **7.2: Universal Crosshair Sync**
    - [ ] Read/Write to `GlobalCrosshair` atomics defined in `quant_workspace_component.hpp`.
    - [ ] Draw 1px dashed line in all panels when `g_crosshair.active == true`.