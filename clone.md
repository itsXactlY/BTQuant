# TASK_BTQ_CLONE_ARCHITECTURE.md

**Objective:** Upgrade the BTQ Render Engine to achieve BTQ parity in Order Flow, Profiling, Liquidity, and visual aesthetics.
**Core Principle:** All heavy visual computation (Heatmaps, TPO aggregations) must leverage custom Vulkan textures (`ImTextureID`) or batched `ImDrawList` calls. Zero blocking in the UI thread.

---

## Phase 1: The Visual Vanguard (BTQ Aesthetics)
**Goal:** Establish the brutalist, borderless, high-contrast visual identity.

- [x] **1.1: Institutional Typography (`src/ui/font_manager.cpp`)**
    - Load a monospaced FinTech font (e.g., *JetBrains Mono*) for all numeric axes and data cells.
    - Set `ImFontConfig::OversampleH = 4` and `OversampleV = 4` to eliminate sub-pixel aliasing on small numbers inside footprint cells.
    - Merge FontAwesome 6 for all UI iconography (toolbars, layers, settings).

- [x] **1.2: Deep Void Theme (`src/ui/unified_theme_system.cpp`)**
    - Strip all ImGui padding and borders: `WindowBorderSize = 0`, `FrameBorderSize = 0`, `WindowPadding = ImVec2(0,0)`.
    - Apply BTQ Color Palette:
        - Background (`ImGuiCol_WindowBg`): `#080A0C` (True Void).
        - Panels (`ImGuiCol_ChildBg`): `#101418`.
        - Accent/Buy (`ImGuiCol_Text` or custom): `#00E676` (Neon Mint).
        - Accent/Sell (`ImGuiCol_Text` or custom): `#FF3B69` (Crimson).

---

## Phase 2: Advanced Order Flow (The Microstructure)
**Goal:** Implement the high-resolution Footprint and Aggressor visualizations.

- [x] **2.1: Gradient Footprint Rendering (`src/components/footprint_panel.cpp`)**
    - **Logic:** Bypass standard ImPlot bars. Use `ImGui::GetWindowDrawList()->AddRectFilledMultiColor(...)`.
    - **Visuals:** Calculate intensity based on `node_volume / max_volume_in_bar`. 
    - **Gradient:** If Buy Delta dominates, fade from `#00E676` (Alpha 0.7) to `#00E676` (Alpha 0.1).
    - **Text Contrast:** Dynamically set text color (`#FFFFFF` vs `#000000`) based on the calculated background luminance of the cell.

- [x] **2.2: Imbalance & Exhaustion Engine (`src/analytics/cluster_engine.cpp` & `footprint_panel.cpp`)**
    - **Calculation:** In `poll_hotspine()`, detect Diagonal Imbalances (e.g., Bid at P vs Ask at P+1 > 300%).
    - **Visuals:** Draw a stark, 2px border (e.g., Gold `#FFD700`) around imbalanced cells using `AddRect`.
    - **Exhaustion:** Highlight the top/bottom cells of a candle if volume is < 5% of the bar's average, signaling buyer/seller exhaustion.

- [x] **2.3: Aggressor Trade Bubbles (`src/components/chart_panel.cpp`)**
    - **Data:** `MarketDataProcessor` detects block trades (> X threshold) and pushes to an atomic ring buffer.
    - **Visuals:** Overlay on the main ImPlot canvas using `ImPlot::PlotScatter` or direct `DrawList->AddCircleFilled`.
    - **Math:** Radius = `log10(trade_volume) * scale_factor`. Color = Green (Ask hit) / Red (Bid hit) with 50% opacity to reveal overlapping clusters.

- [x] **2.4: CVD (Cumulative Volume Delta) Overlay**
    - Render CVD as a continuous line at the bottom of the chart panel.
    - Color the line dynamically: Green when sloping up, Red when sloping down, rendering via `ImPlot::PlotLine` using a custom getter mapped to the lock-free data spine.

---

## Phase 3: Market Profiling (TPO & Volume)
**Goal:** Institutional-grade Time Price Opportunity and Volume Profile rendering.

- [x] **3.1: The TPO Engine (`src/analytics/tpoengine.cpp`)**
    - Map 30-minute time brackets to characters (A-Z, a-z).
    - Maintain an `std::vector<std::string>` per price level for the active session.
    - **Visuals (`tpo_panel.cpp`):** Render blocks using `DrawList->AddRectFilled`. Map each letter to a subtle color gradient based on the time of day (e.g., Morning = Blue, Afternoon = Purple).

- [x] **3.2: Value Area (VA) & POC Tracking**
    - **Math:** Calculate the 70% Value Area dynamically as new trades hit the `poll_hotspine()` loop.
    - **Visuals:** Dim the opacity of all profile blocks outside the Value Area to 30%.
    - **POC:** Draw a persistent 1px horizontal line extending across the chart from the Point of Control. 

- [x] **3.3: Naked POCs / Virgin POCs (`src/components/volume_profile_panel.cpp`)**
    - Track historical POCs that have not been touched by subsequent price action.
    - Draw them as dashed rays extending rightward into the future, terminating exactly when the live price intersects them.

---

## Phase 4: Liquidity Heatmap (DOM Surface)
**Goal:** Replicate BTQ's smooth, GPU-accelerated historical depth surface.

- [x] **4.1: Rolling Depth Buffer (`src/data/orderbook_snapshot_manager.hpp`)**
    - Maintain a lock-free circular buffer of the top 100 Bid/Ask levels, snapping state every 100ms.

- [x] **4.2: Vulkan Compute Integration (`src/components/dom_surface_panel.cpp`)**
    - **Bypass ImGui Primitives:** Do not use `AddRectFilled` for the heatmap. It will kill the CPU.
    - **Pipeline:** 1. Push the rolling depth buffer to a Vulkan SSBO (Storage Buffer).
        2. Dispatch `lob_heatmap.comp` (Compute Shader) to map liquidity values to a color gradient (Dark Blue -> Crimson -> Yellow/White for highest liquidity).
        3. Output to a `VkImage`.
    - **ImGui Bind:** Register the `VkImage` with ImGui (`ImGui_ImplVulkan_AddTexture`) and render it using `ImGui::Image()`, spanning the chart background.

- [x] **4.3: Resting Liquidity Sweeps**
    - Detect when massive resting liquidity is suddenly pulled or executed.
    - Draw a distinct overlay (e.g., a hollow white circle or vector line) over the heatmap texture at the exact coordinate where the liquidity vanished.

---

## Phase 5: Global Synchronization & UX
**Goal:** Connect the panels so they behave as a single, cohesive organism.

- [x] **5.1: Global Crosshair Sync (`src/components/quant_workspace_component.cpp`)**
    - Add `std::atomic<double> global_crosshair_price` and `std::atomic<uint64_t> global_crosshair_time`.
    - **Write:** Any panel (Chart, DOM, TPO) currently hovered by the mouse writes to these atomics.
    - **Read:** All other panels read these atomics and draw a 1px dashed line (`ImGuiCol_TextDisabled`) at the corresponding X/Y coordinates.

- [x] **5.2: Floating Toolbars (`src/components/chart_panel.cpp`)**
    - Render the timeframe and drawing toolbars as borderless child windows overlapping the main canvas (`ImGuiWindowFlags_NoBackground`).
    - Use `ImGui::SetCursorPos` to float them in the top-left and right edges, preserving maximum screen real estate for the data.

- [x] **5.3: Flush DOM Ruler (`src/components/dom_surface_panel.cpp`)**
    - Anchor the live Orderbook strictly to the right edge of the Heatmap panel.
    - Render current Bids/Asks as horizontal bars extending inward from the Y-axis scale, dynamically overlaying the historical heatmap texture.