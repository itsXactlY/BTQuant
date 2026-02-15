# TASK_ULTIMA_MMT_CLONE_MAPPING.md

**Objective:** Transform BTQ Terminal into a 144Hz institutional Order Flow platform by mapping MMT features to the existing C++26 decoupled architecture.
**Core Principle:** Data is pulled from `MarketDataProcessor` atomics; UI is composed via `LayoutManager` DockBuilder; Heavy visuals use `VulkanCore` textures.

---

## Phase 1: The Docking Matrix (Workspace Composition)
**Target:** `src/ui/layout_manager.cpp` & `src/components/quant_workspace_component.cpp`

- [x] **1.1: MMT programmatic Grid Construction**
- Inside `LayoutManager::apply_layout_preset(MODERN_TRADING)`:
- [x] Create central `ChartSuperNode`.
- [x] Split Left (3%) -> `drawing_tools_panel`.
- [x] Split Right (25%) -> `dom_surface_panel` + `orderbook_panel`.
- [x] Split Right-Bottom (40%) -> `time_and_sales` (Trades).
- [x] Split Center-Bottom (15%) -> `time_histogram_panel`.
- [x] Globally set `ImGuiStyleVar_WindowPadding` to `(0,0)` in `QuantWorkspaceComponent::render_gui`.
- [x] Set `ImGuiCol_Separator` to `rgba(94, 82, 64, 0.2)` in `unified_theme_system.cpp`.

---

## Phase 2: The GPU Heatmap Spine (LOB Engine)
**Target:** `src/components/dom_surface_panel.cpp` & `shaders/lob_heatmap.comp`

- [x] **2.1: Lock-Free Snapshot Pipeline**
- [x] In `MarketDataProcessor`, implement a ring buffer of `OrderBookSnapshot` structs (Standard Layout POD).
- [x] Use `std::atomic<uint64_t> snapshot_head` to signal `VulkanCore` that new data is ready for the Compute Shader.
- [x] **2.2: Compute-to-ImGui Bind**
- [x] Update `DomSurfacePanel` to retrieve the `VkImage` descriptor from `GPUMemoryManager`.
- [x] Use `ImGui_ImplVulkan_AddTexture` to map the heatmap to an `ImTextureID`.
- [x] Render via `ImGui::GetWindowDrawList()->AddImage()` spanning the panel background.
- [x] **2.3: Multi-Exchange Aggregation**
- [x] Add "Aggregated Heatmap" options in the right-click context menu of the DOM header.
- [x] Update the Compute Shader to sum multiple atomic depth buffers before colormapping.

---

## Phase 3: The DOM & Market Volume Profiles
**Target:** `src/components/dom_surface_panel.cpp` & `src/analytics/cluster_engine.cpp`

- [x] **3.1: 5-Column MMT Layout**
- [x] Render Table: `[Buys | Asks | Price | Bids | Sells]`.
- [x] **Center Mode:** Mathematically lock Y-limits: `y_min = current_price - range`.
- [x] **3.2: Cumulative Volume Columns**
- [x] Pull `VolumeData` from `ClusterEngine` without mutexes.
- [x] Render horizontal bars using `DrawList->AddRectFilled`.
- [x] Green bars (Buys) extend right; Red bars (Sells) extend left.

---

## Phase 4: The Order Book Widget (Liquidity Ladder)
**Target:** `src/components/orderbook_panel.cpp`

- [x] **4.1: Atomic Unit Toggles**
- [x] Implement USD / COIN toggle in header.
- [x] If USD: Multiply `atomic_size` by `atomic_last_price` during render pass.
- [x] **4.2: Depth Rendering**
- [x] Render Asks (Red) descending from top; Bids (Green) ascending from bottom.
- [x] Read from `MarketDataProcessor::get_atomic_snapshot` array.

---

## Phase 5: The Trades Feed (Acoustic Order Flow)
**Target:** `src/components/time_and_sales.cpp` & `src/trading/trade_command_queue.cpp`

- [x] **5.1: Raw Trade Table**
- [x] Columns: `[Exchange Logo] | Price | Qty | Time`.
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
- [x] **6.2: Quick Order Sidebar**
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