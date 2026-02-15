# TASK_ULTIMA_GENESIS_50.md

**Objective:** The total transformation of BTQ Terminal into an institutional, GPU-accelerated, lock-free Order Flow platform matching MMT.gg.
**Core Principle:** No standard ImGui widgets for data. No mutexes on the hot path. Programmatic docking. Sub-pixel rendering.

---

## Phase 1: The Deep Void Aesthetic (UI Engine)
**Target Files:** `src/ui/unified_theme_system.cpp`, `src/ui/font_manager.cpp`, `src/ui/ui_base.cpp`

- [x] **01.** Set `style.WindowBorderSize = 0.0f` to remove legacy 1970s window outlines.
- [x] **02.** Set `style.ChildBorderSize = 0.0f` and `style.FrameBorderSize = 0.0f`.
- [x] **03.** Set `style.WindowPadding = ImVec2(0, 0)` globally so docked panels blend seamlessly.
- [x] **04.** Define `ImGuiCol_WindowBg` as `#0B0E11` (MMT True Void).
- [x] **05.** Define `ImGuiCol_ChildBg` as `#15191E` (Panel Surface).
- [x] **06.** Define `ImGuiCol_Text` as `#D1D4DC` (Off-white anti-glare).
- [x] **07.** Define `ImGuiCol_Separator` as `rgba(94, 82, 64, 0.2)` (Subtle grid lines).
- [x] **08.** In `font_manager.cpp`, load *JetBrains Mono* or *Berkeley Mono* as the primary numeric font.
- [x] **09.** Configure `ImFontConfig::OversampleH = 4` and `OversampleV = 4` for sub-pixel anti-aliasing on numbers.
- [x] **10.** Merge *FontAwesome 6* into the font atlas for textless, icon-driven sidebars.

---

## Phase 2: The Master Docking Matrix
**Target Files:** `src/ui/layout_manager.cpp`, `src/components/quant_workspace_component.cpp`

- [x] **11.** In `apply_layout_preset(MODERN_TRADING)`, call `panels_.clear()` to eradicate duplicate panels before building.
- [x] **12.** Generate a new `ImGuiID dock_main = ImGui::GetID("WorkspaceDockSpace");`.
- [x] **13.** Apply `ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_PassthruCentralNode`.
- [x] **14.** Execute `DockBuilderSplitNode(dock_main, ImGuiDir_Left, 0.03f)` for the Drawing Tools sidebar.
- [x] **15.** Execute `DockBuilderSplitNode(dock_main, ImGuiDir_Right, 0.25f)` for the right DOM/Orderbook column.
- [x] **16.** Split the right column down (`ImGuiDir_Down, 0.40f`) to create the Time & Sales area.
- [x] **17.** Split the center node down (`ImGuiDir_Down, 0.15f`) for Time Histograms.
- [x] **18.** Dock `"Drawing Tools"` into the Left ID.
- [x] **19.** Dock `"Main Chart"` into the Center ID.
- [x] **20.** Dock `"DOM Surface"` and `"Order Book"` into the Right-Top ID (creating a tab group).
- [x] **21.** Dock `"Time & Sales"` into the Right-Bottom ID.

---

## Phase 3: Lock-Free State & Global Sync
**Target Files:** `include/market_data_processor.hpp`, `include/quant_workspace_component.hpp`

- [x] **22.** Ensure `AtomicSymbolInfo` struct is marked `alignas(64)` to prevent cache-line false sharing.
- [x] **23.** Declare `std::atomic<double> g_crosshair_price` in the workspace component.
- [x] **24.** Declare `std::atomic<uint64_t> g_crosshair_time` in the workspace component.
- [x] **25.** In `chart_panel.cpp`, if `ImPlot::IsPlotHovered()`, write mouse Y/X to `g_crosshair_price` and `g_crosshair_time` via `memory_order_relaxed`.
- [x] **26.** In `dom_surface_panel.cpp`, read `g_crosshair_price` and draw a 1px dashed horizontal line (`ImGuiCol_TextDisabled`) across the heatmap.
- [x] **27.** In `tpo_panel.cpp`, read `g_crosshair_price` and highlight the corresponding TPO letter block.
- [x] **28.** Wire the global `SymbolSelector` to atomic `active_symbol_id`. All panels must seamlessly swap data feeds when this atomic changes.

---

## Phase 4: DOM Surface & GPU Heatmap
**Target Files:** `src/components/dom_surface_panel.cpp`, `shaders/lob_heatmap.comp`, `src/system/GPUMemoryManager.cpp`

- [x] **29.** Read the rolling 100-level `OrderBookSnapshot` from the lock-free ring buffer.
- [ ] **30.** Push snapshot buffer to Vulkan SSBO.
- [ ] **31.** Dispatch `lob_heatmap.comp` to map liquidity to a Viridis/Magma color gradient into a `VkImage`.
- [ ] **32.** Retrieve the `ImTextureID` from `GPUMemoryManager` and render it via `ImGui::GetWindowDrawList()->AddImage()`.
- [ ] **33.** Add "HD/SD" resolution toggle (1 tick vs 10 ticks per row aggregation).
- [ ] **34.** Render the 5-column ImGui table: `[Buys | Asks | Price | Bids | Sells]`.
- [ ] **35.** Implement `PriceScaleMode::Center`: Mathematically lock Y-axis so `current_price` is always `(y_max + y_min) / 2`.
- [ ] **36.** Implement `PriceScaleMode::Auto`: Soft-lerp the Y-axis center only if price deviates > 25% from the middle.
- [ ] **37.** Pull `VolumeData` from `ClusterEngine` and draw solid Green rectangles extending Right for market buys.
- [ ] **38.** Draw solid Red rectangles extending Left for market sells.
- [ ] **39.** Add a right-click context menu to aggregate multiple exchange order books into one SSBO feed.

---

## Phase 5: Time & Sales (Trades) & Acoustics
**Target Files:** `src/components/time_and_sales.cpp`, `src/data/market_data_processor.cpp`

- [ ] **40.** Render a 4-column table: `[Exchange Logo] | Price | Qty | Time`.
- [ ] **41.** Read strictly from the atomic tail of `trade_ring_buffer_`. Do not copy the vector.
- [ ] **42.** Color code the Price column: `#00E676` (Neon Mint) for Ask hits, `#FF3B69` (Crimson) for Bid hits.
- [ ] **43.** Slippage Bracket: Calculate time delta between consecutive trades. If `< 50ms` and price changed, draw a 1px vertical white bracket linking the rows.
- [ ] **44.** Add `Filter` input in the header. Skip rendering any trade with `size < filter_val`.
- [ ] **45.** Integrate `miniaudio` into the data processor.
- [ ] **46.** Trigger `ma_engine_play_sound("buy.wav")` or `"sell.wav"` on incoming trades.
- [ ] **47.** Calculate audio pitch: `pitch = 1.0f - (log10(size) * scalar)`. Massive trades sound like a heavy bass thud.

---

## Phase 6: Order Book Widget
**Target Files:** `src/components/orderbook_panel.cpp`

- [ ] **48.** Render the UI header with a `USD / COIN` toggle switch.
- [ ] **49.** If USD is active, multiply atomic sizes by the atomic `current_price` on the fly during the render loop.
- [ ] **50.** Render Asks descending from top. Render Bids ascending from bottom.
- [ ] **51.** Access `snapshot_asks_` and `snapshot_bids_` via `std::memory_order_acquire`. ZERO mutexes allowed here.

---

## Phase 7: Chart Panel & SPSC Order Entry
**Target Files:** `src/components/chart_panel.cpp`, `src/trading/trading_interface.cpp`, `src/trading/trade_command_queue.cpp`

- [ ] **52.** Override ImPlot defaults: `ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0,0))` to make the chart flush with the window edges.
- [ ] **53.** Draw a borderless, floating Top Toolbar (`ImGui::SetCursorPos`) containing Symbol, Timeframe, and Chart Style.
- [ ] **54.** Implement "Snap to Last": If `ImPlot::GetPlotLimits().X.Max < latest_atomic_time`, draw a hovering arrow button that forces X-axis to jump to the live edge.
- [ ] **55.** In `trading_interface.cpp`, pull live `best_bid` and `best_ask` from atomics and display them inside massive `BUY MKT` and `SELL MKT` buttons.
- [ ] **56.** Upon clicking a trading button, construct a `TradeCommand` struct and push it into `lockfree_queue` (`trade_command_queue.cpp`). UI thread must immediately return to rendering the next frame.