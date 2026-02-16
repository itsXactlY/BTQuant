# TASK_BTQ_V2_IMPLEMENTATION.md

**Objective:** Execute the BTQ Render Engine V2.0 pixel-perfect specification.

---

## Phase 1: Core Assets & Theming
**Target:** `src/ui/unified_theme_system.cpp`, `src/ui/font_manager.cpp`

- [ ] **1.1: Global Color Matrix Setup**
  - Implement exact hex values in `UnifiedThemeSystem`:
    - `BACKGROUND_PRIMARY`: `#1A1D2E`
    - `BACKGROUND_SECONDARY`: `#16192B`
    - `TEXT_PRIMARY`: `#E8E9ED`
    - `TEXT_SECONDARY`: `#8B8E98`
    - `BID_GREEN`: `#00D084`
    - `ASK_RED`: `#F6465D`
    - `ACCENT_BLUE`: `#00C9FF`
    - `ACCENT_YELLOW`: `#FFD700`
    - `GRID_LINE`: `#2B2F42`
    - `PANEL_BORDER`: `#3A3F56`
- [ ] **1.2: Font Integration**
  - Load `JetBrains Mono` at `11px` (Weight 400).
  - Load `JetBrains Mono` at `11px` (Weight 700 / Bold).
  - Map to ImGui font atlas.

---

## Phase 2: CMake & Asset Management
**Target:** `CMakeLists.txt`, `src/analytics/colormap_manager.cpp`

- [ ] **2.1: Miniaudio Dependency**
  - Add `FetchContent` for `miniaudio` (tag: `master`).
  - Link `miniaudio` include directories to `BTQuantTerminal`.
- [ ] **2.2: Compute Shader Compilation**
  - Add `add_custom_command` to compile `shaders/heatmap_intensity.comp` to `.spv` via `glslc`.
- [ ] **2.3: Colormap Manager**
  - Create `include/analytics/colormap_manager.hpp` & `src/analytics/colormap_manager.cpp`.
  - Load 25 colormap PNGs (Viridis, Plasma, Turbo, etc.) into a `256x32` Vulkan texture atlas.

---

## Phase 3: Acoustic Feedback Engine
**Target:** `include/analytics/audio_engine.hpp`, `src/analytics/audio_engine.cpp`

- [ ] **3.1: Engine Initialization**
  - Initialize `ma_engine` (2 channels, 48000 sample rate).
  - Load `assets/audio/buy_tone.wav` and `assets/audio/sell_tone.wav`.
- [ ] **3.2: Pitch Modulation Logic**
  - Implement inverse logarithmic pitch scaling: `pitch = MAX_PITCH - (normalized * (MAX_PITCH - MIN_PITCH))`.
  - Min: 220Hz, Max: 880Hz.
- [ ] **3.3: Data Pipeline Hook**
  - Wire `AudioEngine::play_trade_sound(size, is_buy)` directly to the Worker Thread processing loop.

---

## Phase 4: Time & Sales (Trades Feed)
**Target:** `src/components/time_and_sales.cpp`

- [ ] **4.1: Panel Frame & Header**
  - Set dimensions: `320px` width, `800px` height.
  - Render Header (Height: `40px`, BG: `#16192B`).
  - Add Filter Input (pos: `10,8`), Reset Button, USD/COIN toggle, Sound Button.
- [ ] **4.2: Trades Grid (ImGuiListClipper)**
  - Row height: `18px`.
  - Column 1: Exchange Logo (14x14px).
  - Column 2: Price (Width: `90px`, Right-align, `#00D084` or `#F6465D`).
  - Column 3: Quantity (Width: `100px`, Right-align).
  - Column 4: Time (Width: `85px`, Right-align, `#8B8E98`, Format: `HH:MM:SS.mmm`).
- [ ] **4.3: Alpha Gradient Background**
  - Calculate size percentile using `deque(maxlen=500)`.
  - Map percentile to alpha `0.05` to `0.50` over `BID_GREEN` / `ASK_RED`.

---

## Phase 5: Order Book Widget (Size & Sum Modes)
**Target:** `src/components/orderbook_panel.cpp`

- [ ] **5.1: Panel Frame & Header**
  - Set dimensions: `280px` width, `600px` height.
  - Symbol Dropdown (Width: `160px`, Height: `24px`, BG: `#0F1218`).
  - Depth Selector (Width: `20px`).
  - Currency Toggle (USD/COIN, Width: `70px`).
- [ ] **5.2: Size Mode Rendering**
  - Mid-Price Display (Height: `40px`, BG: `#0F1218`, Font: 18px Bold).
  - Ask Rows (10 rows): Price (`#F6465D`), Size, Liquidity Bar (`linear(transparent, ASK_RED alpha=0.3)`).
  - Bid Rows (10 rows): Price (`#00D084`), Size, Liquidity Bar (`linear(transparent, BID_GREEN alpha=0.3)`).
- [ ] **5.3: Sum Mode Implementation**
  - Replace `Size` column with `Sum` column.
  - Calculate `cumulative_depth[i] = sum(sizes[0..i])`.
  - Update Liquidity Bar width mapping: `(cumulative_depth[i] / total_book_depth) * 100px`.

---

## Phase 6: Footprint Chart Engine
**Target:** `include/analytics/footprint_engine.hpp`, `src/components/footprint_panel.cpp`

- [ ] **6.1: Data Structure**
  - Implement `FootprintCell` struct (price, times, buy/sell/total volume, delta, trade_count).
- [ ] **6.2: Heatmap Cell Rendering**
  - Calculate `normalized_volume = cell.total_volume / max_candle_volume`.
  - Fetch Viridis color via `ColormapManager` and render `AddRectFilled`.
- [ ] **6.3: POC & Imbalances**
  - Draw POC line (`#FFD700`, 2px thickness) at `max_volume_price` per candle.
  - Imbalance detection: If `buy/sell > 3.0` draw `#00D084` border. If `sell/buy > 3.0` draw `#F6465D` border.
- [ ] **6.4: Hover Tooltip**
  - Render tooltip at `(mouse_x + 10, mouse_y + 10)` with background `rgba(15, 18, 24, 0.95)`.

---

## Phase 7: GPU Heatmap Engine
**Target:** `shaders/heatmap_intensity.comp`, `src/analytics/heatmap_engine.cpp`, `src/analytics/heatmap_renderer.cpp`

- [ ] **7.1: Data Snapshot & Buffers**
  - Implement `on_candle_close()` trigger to write `snapshot.bids + snapshot.asks` to `2D_FLOAT_ARRAY[time_bins][price_bins]`.
  - Map bins: HD Mode (`1 tick/bin`, height `2px`), SD Mode (`5 ticks/bin`, height `10px`).
- [ ] **7.2: Compute Shader Pipeline**
  - **Stage 1 (Filter):** Zero out bins `< params.low_threshold`.
  - **Stage 2 (Intensity):** Normalize `clamp(val / params.peak_threshold, 0, 1)`.
  - **Stage 3 (Blur):** Implement `3x3` matrix convolution if `mode == SPLAT`.
  - **Stage 4 (Color):** Map intensity to `texture(colormap_atlas)`.
- [ ] **7.3: Live Extension Column**
  - Render a vertical 20px wide column on the right edge.
  - Feed with 100ms updates, lerping intensity over 200ms.

---

## Phase 8: Heatmap UI Controls
**Target:** `src/ui/heatmap_settings_panel.cpp`

- [ ] **8.1: Chart Overlay Controls**
  - Render floating bar `(400x30)` at top center of Chart.
  - Include HD/SD toggle, Intensity Slider (track: `#2B2F42`, fill: `#00C9FF`), Settings Icon.
- [ ] **8.2: Settings Window**
  - Dimensions: `330px x 500px`. Border: `1px solid #3A3F56`.
  - **Style Section:** HD Toggle, Style Dropdown (Classic/Splat/Gaussian), Colormap Dropdown.
  - **Intensity Section:** Color Scale Preview (gradient rect), Low/Peak dual-thumb slider `[0.0, 50000.0]`.
  - **Misc Section:** Zoom Tooltip toggle, Extend Heatmap toggle.
  - **Aggregate Section:** Horizontal wrap layout of Exchange Buttons (Binance, OKX, Coinbase, Bybit).