# Versteckte Funktionen im BTQuant Terminal

## Übersicht

Im Code sind **30 Panel-Typen** definiert, aber nur **~15** sind im Dashboard-Menü eingebunden.

---

## Panel-Typen Analyse

### ✅ Im Menü eingebunden (15)

| PanelType | Datei | Menü-Pfad |
|-----------|-------|-----------|
| `CHART` | quant_workspace_component.cpp:72 | "Add Chart Panel" |
| `METRICS` | quant_workspace_component.cpp:76 | "Add Metrics Panel" |
| `HEATMAP` | quant_workspace_component.cpp:80 | "Add Heatmap Panel" |
| `ORDERBOOK` | quant_workspace_component.cpp:83 | "Add Orderbook" |
| `FOOTPRINT_CHART` | quant_workspace_component.cpp:86 | "Add Footprint" |
| `TPO_PROFILE` | quant_workspace_component.cpp:90 | "Add TPO Profile" |
| `WATCHLIST` | dashboard_controls.cpp:652 | "Add Watchlist" |
| `TIME_AND_SALES` | dashboard_controls.cpp:644 | "Add Time & Sales" |
| `ALERTS` | dashboard_controls.cpp:660 | "Add Alerts" |
| `RISK_ANALYZER` | dashboard_controls.cpp:668 | "Add Risk Analyzer" |
| `VOLUME_PROFILE` | dashboard_controls.cpp:628 | "Add Volume Profile" |
| `TRADING_ORDERS` | panel_manager.cpp:1302 | Layout: MODERN_TRADING |
| `TRADING_POSITIONS` | panel_manager.cpp:1303 | Layout: MODERN_TRADING |
| `RISK_METRICS` | panel_manager.cpp:1304 | Layout: MODERN_TRADING |
| `STATUS_BAR` | panel_manager.cpp:1305 | Layout: MODERN_TRADING |

### ❌ NICHT im Menü eingebunden (15) - ABER IMPLEMENTIERT!

| PanelType | Header-Datei | Beschreibung | Status |
|-----------|--------------|--------------|--------|
| `HISTOGRAM` | histogram_panel.hpp | Histogramm-Visualisierung | ✅ Implementiert |
| `SCATTER_PLOT` | scatter_plot_panel.hpp | Streudiagramm | ✅ Implementiert |
| `TIME_SERIES` | time_series_panel.hpp | Zeitreihen-Diagramm | ✅ Implementiert |
| `SCREENER` | screener_panel.hpp | Aktienscreener | ✅ Implementiert |
| `TAPE` | tape_panel.hpp | Time & Sales Tape | ✅ Implementiert |
| `DEPTH_CHART` | depth_chart_panel.hpp | Depth-of-Market Chart | ✅ Implementiert |
| `LOG_PANEL` | log_panel.hpp | System-Logs | ✅ Implementiert |
| `PERFORMANCE_MONITOR` | performance_monitor_panel.hpp | Performance-Metriken | ✅ Implementiert |
| `TIME_STATISTICS` | time_statistics_panel.hpp | Zeitbasierte Statistiken | ✅ Implementiert |
| `TIME_HISTOGRAM` | time_histogram_panel.hpp | Zeit-Histogramm | ✅ Implementiert |
| `HISTORICAL_TIME_SALES` | historical_time_sales.hpp | Historische Time & Sales | ✅ Implementiert |
| `CHART_REPLAY` | chart_replay_panel.hpp | Chart-Wiedergabe | ✅ Implementiert |
| `STRATEGY_BUILDER` | strategy_builder.hpp | Strategie-Editor | ✅ Implementiert |
| `OPTION_ANALYTICS` | option_analytics_panel.hpp | Options-Analyse | ✅ Implementiert |

---

## Zusätzliche Panels (nicht im PanelType Enum)

Diese Panels existieren als Header-Dateien, sind aber NICHT im `PanelType` Enum definiert:

| Header-Datei | Beschreibung | Status |
|--------------|--------------|--------|
| `correlation_heatmap_component.hpp` | Korrelations-Heatmap | ⚠️ Nicht im Enum |
| `dom_surface_panel.hpp` | DOM Surface Panel | ⚠️ Nicht im Enum |
| `multi_vwap_panel.hpp` | Multi-VWAP Panel | ⚠️ Nicht im Enum |
| `technical_indicators_component.hpp` | Technische Indikatoren | ⚠️ Nicht im Enum |
| `theme_customization_component.hpp` | Theme-Anpassung | ⚠️ Nicht im Enum |
| `keyboard_shortcuts_component.hpp` | Tastaturkürzel | ⚠️ Nicht im Enum |
| `drawing_tools.hpp` | Zeichen-Tools | ⚠️ Nicht im Enum |
| `interaction_manager.hpp` | Interaktions-Manager | ⚠️ Nicht im Enum |

---

## Weitere versteckte Funktionen

### 1. Layout-Presets (nur intern genutzt)

**Datei:** [`panel_manager.cpp:1285-1331`](dependencies/BTQ_Render_Engine/src/components/panel_manager.cpp:1285)

```cpp
enum class LayoutPreset {
  DEFAULT,           // 3 Panels
  MODERN_TRADING,    // 8 Panels
  DASHBOARD_ONLY,    // 4 Panels
  CHART_FOCUS,       // 4 Panels
  RISK_MONITORING    // 5 Panels
};
```

**Problem:** Kein Menü-Eintrag zum Wechseln zwischen Layouts!

**Fix:** Menü-Eintrag hinzufügen:
```cpp
if (ImGui::BeginMenu("Layout Presets")) {
  if (ImGui::MenuItem("Default")) {
    panel_mgr->apply_layout_preset(LayoutPreset::DEFAULT);
  }
  if (ImGui::MenuItem("Modern Trading")) {
    panel_mgr->apply_layout_preset(LayoutPreset::MODERN_TRADING);
  }
  // ... etc
  ImGui::EndMenu();
}
```

### 2. Debug Overlay (F12)

**Datei:** [`main_trading_terminal.cpp:97`](dependencies/BTQ_Render_Engine/src/main_trading_terminal.cpp:97)

```cpp
if (ImGui::MenuItem("Debug Overlay (F12)")) {
  g_debug_overlay.toggle_visibility();
}
```

**Status:** ✅ Im Menü eingebunden

### 3. Performance Overlay

**Datei:** [`main_trading_terminal.cpp:92`](dependencies/BTQ_Render_Engine/src/main_trading_terminal.cpp:92)

```cpp
static bool show_perf = true;
if (ImGui::MenuItem("Performance Overlay", nullptr, &show_perf)) {
  dashboard->set_show_performance_overlay(show_perf);
}
```

**Status:** ✅ Im Menü eingebunden

### 4. Auto Arrange Panels

**Datei:** [`main_trading_terminal.cpp:102`](dependencies/BTQ_Render_Engine/src/main_trading_terminal.cpp:102)

```cpp
if (ImGui::MenuItem("Auto Arrange Panels")) {
  if (workspace && workspace->getPanelManager())
    workspace->getPanelManager()->auto_arrange_panels();
}
```

**Status:** ✅ Im Menü eingebunden

---

## Empfohlene Menü-Erweiterungen

### View → Add Panel (Erweitert)

```cpp
if (ImGui::BeginMenu("Add Panel")) {
  // Charts
  if (ImGui::MenuItem("Chart")) panel_mgr->add_panel(PanelType::CHART);
  if (ImGui::MenuItem("Footprint")) panel_mgr->add_panel(PanelType::FOOTPRINT_CHART);
  if (ImGui::MenuItem("TPO Profile")) panel_mgr->add_panel(PanelType::TPO_PROFILE);
  if (ImGui::MenuItem("Volume Profile")) panel_mgr->add_panel(PanelType::VOLUME_PROFILE);
  
  ImGui::Separator();
  
  // Market Data
  if (ImGui::MenuItem("Order Book")) panel_mgr->add_panel(PanelType::ORDERBOOK);
  if (ImGui::MenuItem("Time & Sales")) panel_mgr->add_panel(PanelType::TIME_AND_SALES);
  if (ImGui::MenuItem("Watchlist")) panel_mgr->add_panel(PanelType::WATCHLIST);
  if (ImGui::MenuItem("Heatmap")) panel_mgr->add_panel(PanelType::HEATMAP);
  
  ImGui::Separator();
  
  // Trading
  if (ImGui::MenuItem("Orders")) panel_mgr->add_panel(PanelType::TRADING_ORDERS);
  if (ImGui::MenuItem("Positions")) panel_mgr->add_panel(PanelType::TRADING_POSITIONS);
  if (ImGui::MenuItem("Alerts")) panel_mgr->add_panel(PanelType::ALERTS);
  
  ImGui::Separator();
  
  // Analysis
  if (ImGui::MenuItem("Metrics")) panel_mgr->add_panel(PanelType::METRICS);
  if (ImGui::MenuItem("Risk Metrics")) panel_mgr->add_panel(PanelType::RISK_METRICS);
  if (ImGui::MenuItem("Risk Analyzer")) panel_mgr->add_panel(PanelType::RISK_ANALYZER);
  
  ImGui::EndMenu();
}
```

### View → Layout Presets

```cpp
if (ImGui::BeginMenu("Layout Presets")) {
  if (ImGui::MenuItem("Default", "Ctrl+1")) {
    panel_mgr->apply_layout_preset(LayoutPreset::DEFAULT);
  }
  if (ImGui::MenuItem("Modern Trading", "Ctrl+2")) {
    panel_mgr->apply_layout_preset(LayoutPreset::MODERN_TRADING);
  }
  if (ImGui::MenuItem("Dashboard Only", "Ctrl+3")) {
    panel_mgr->apply_layout_preset(LayoutPreset::DASHBOARD_ONLY);
  }
  if (ImGui::MenuItem("Chart Focus", "Ctrl+4")) {
    panel_mgr->apply_layout_preset(LayoutPreset::CHART_FOCUS);
  }
  if (ImGui::MenuItem("Risk Monitoring", "Ctrl+5")) {
    panel_mgr->apply_layout_preset(LayoutPreset::RISK_MONITORING);
  }
  ImGui::EndMenu();
}
```

---

## Implementierungs-Checkliste

### Phase 1: Menü-Erweiterung (HOHE PRIORITÄT)
- [ ] `quant_workspace_component.cpp` - "Add Panel" Menü erweitern
- [ ] `main_trading_terminal.cpp` - "Layout Presets" Menü hinzufügen
- [ ] Fehlende Panel-Typen in Menü aufnehmen

### Phase 2: Panel-Implementierung (ALLE VORHANDEN!)
✅ Alle 14 "versteckten" Panel-Typen sind bereits implementiert:
- `histogram_panel.hpp`
- `scatter_plot_panel.hpp`
- `time_series_panel.hpp`
- `screener_panel.hpp`
- `tape_panel.hpp`
- `depth_chart_panel.hpp`
- `log_panel.hpp`
- `performance_monitor_panel.hpp`
- `time_statistics_panel.hpp`
- `time_histogram_panel.hpp`
- `historical_time_sales.hpp`
- `chart_replay_panel.hpp`
- `strategy_builder.hpp`
- `option_analytics_panel.hpp`

### Phase 3: Zusätzliche Panels ins Enum aufnehmen
- [ ] `correlation_heatmap_component.hpp` → `PanelType::CORRELATION_HEATMAP`
- [ ] `dom_surface_panel.hpp` → `PanelType::DOM_SURFACE`
- [ ] `multi_vwap_panel.hpp` → `PanelType::MULTI_VWAP`
- [ ] `technical_indicators_component.hpp` → `PanelType::TECHNICAL_INDICATORS`
- [ ] `theme_customization_component.hpp` → `PanelType::THEME_CUSTOMIZATION`
- [ ] `keyboard_shortcuts_component.hpp` → `PanelType::KEYBOARD_SHORTCUTS`
- [ ] `drawing_tools.hpp` → `PanelType::DRAWING_TOOLS`

### Phase 4: PanelManager::add_panel() erweitern
- [ ] Case für alle fehlenden PanelTypes in [`panel_manager.cpp`](dependencies/BTQ_Render_Engine/src/components/panel_manager.cpp) hinzufügen


# TASK_QUANTOWER_UI_ARCHITECTURE.md

**Objective:** Wire up all 30+ hidden C++ panels into a unified, lock-free ImGui interface that perfectly mimics the Quantower layout (Top Toolbar, Sidebar, Chart Area, Order Entry, Bottom Toolbar).

---

## Phase 1: The Enum & Factory Expansion (Awakening the Ghosts)
**Goal:** Your codebase has 15+ panels fully implemented in C++ but completely invisible to the UI. We must register them in the central factory.

- [ ] **1.1: Expand `PanelType` Enum**
    - **File:** `include/components/panel_manager.hpp`
    - **Action:** Add the missing panel types to the `enum class PanelType`:
      `HISTOGRAM`, `SCATTER_PLOT`, `TIME_SERIES`, `SCREENER`, `TAPE`, `DEPTH_CHART`, `LOG_PANEL`, `PERFORMANCE_MONITOR`, `TIME_STATISTICS`, `TIME_HISTOGRAM`, `HISTORICAL_TIME_SALES`, `CHART_REPLAY`, `STRATEGY_BUILDER`, `OPTION_ANALYTICS`, `CORRELATION_HEATMAP`, `DOM_SURFACE`, `MULTI_VWAP`, `TECHNICAL_INDICATORS`, `THEME_CUSTOMIZATION`, `KEYBOARD_SHORTCUTS`, `DRAWING_TOOLS`.

- [ ] **1.2: Wire the `add_panel` Factory**
    - **File:** `src/components/panel_manager.cpp`
    - **Action:** In the `PanelManager::add_panel(PanelType type)` switch statement, add `case` blocks for EVERY enum added in step 1.1.
    - **Implementation:** e.g., `case PanelType::DOM_SURFACE: panel = std::make_shared<DomSurfacePanel>(config, processor_); break;`
    - *(Note: Ensure all respective headers like `dom_surface_panel.hpp` are included at the top).*

---

## Phase 2: The Main Menu & Workspace Wiring
**Goal:** Expose the awakened panels and layout presets to the user via the top menu bar.

- [ ] **2.1: Rebuild "Add Panel" Menu Categories**
    - **File:** `src/components/quant_workspace_component.cpp` (Inside `render_gui()` -> Menu Bar)
    - **Action:** Categorize the `Add Panel` menu exactly as requested:
        - **Charts:** Chart, Footprint, TPO Profile, Volume Profile, Depth Chart.
        - **Market Data:** Order Book, Time & Sales, Tape, Watchlist, Heatmap, DOM Surface.
        - **Trading:** Orders, Positions, Alerts, Strategy Builder.
        - **Analysis:** Metrics, Risk Metrics, Risk Analyzer, Option Analytics, Multi VWAP.

- [ ] **2.2: Expose Layout Presets**
    - **File:** `src/components/quant_workspace_component.cpp`
    - **Action:** Add a `Layout Presets` menu next to `Add Panel`.
    - **Implementation:** Call `panel_manager_->apply_layout_preset(...)` for `DEFAULT`, `MODERN_TRADING`, `DASHBOARD_ONLY`, `CHART_FOCUS`, `RISK_MONITORING`.
    - **Action:** Ensure `MODERN_TRADING` is set as the default on application startup in `main_trading_terminal.cpp`.

---

## Phase 3: The Quantower Chart Panel Anatomy
**Goal:** Transform the basic `ChartPanel` into the 5-part Quantower layout.

- [ ] **3.1: Chart Top Toolbar (Main Controls)**
    - **File:** `src/components/chart_panel.cpp` (Inside `render()`)
    - **Action:** Render a horizontal ImGui bar at the top (`ImGui::BeginChild("TopBar", ImVec2(0, 30))`).
    - **Elements:**
        - Symbol Lookup (InputText).
        - Timeframe Selector (Dropdown: 1m, 5m, 1H, 1D).
        - Chart Style (Dropdown: Candle, Bar, Line, Area, Quantower).
        - Mouse Trading vs. Keyboard Trading toggle button.

- [ ] **3.2: Sidebar Menu (Tools & Objects)**
    - **File:** `src/components/chart_panel.cpp`
    - **Action:** Render a vertical toolbar on the left (`ImGui::BeginChild("Sidebar", ImVec2(40, 0))`).
    - **Elements:**
        - Icons/Buttons for: Crosshair, Drawing Tools (Lines, Fibs), Overlays, and Indicators.
        - **Favorites:** Add a "Star" toggle next to drawing tools in the context menu to pin them to this sidebar.

- [ ] **3.3: The Chart Area (Price Scale Modes)**
    - **File:** `src/components/chart_panel.cpp` (Inside the main ImPlot/Canvas area)
    - **Action:** Implement the 4 Price Centering Modes via right-click on the Y-Axis:
        - *Auto:* standard ImPlot `ImPlotAxisFlags_AutoFit`.
        - *Auto Centered:* Manually set Y limits so `(Y_max + Y_min)/2 == last_price`.
        - *Keep in View:* Only adjust Y limits if `last_price` exceeds current bounds.
        - *Manual:* Disable all auto-fitting. Triggered instantly if the user drags the chart.
    - **Action:** Implement "Snap to Last" button. Only visible if X-axis max < current time. Clicking it resets X-axis to follow live data.

- [ ] **3.4: Sidebar Order Entry (Right Side)**
    - **File:** `src/components/chart_panel.cpp`
    - **Action:** Render a vertical pane on the right side.
    - **Elements:**
        - Market Buy/Sell hot buttons displaying live Best Bid & Ask (Pulled from Atomic L2 Snapshot).
        - Input fields for Order Quantity and TIF (Time In Force).

- [ ] **3.5: Bottom Toolbar (Volume Analysis)**
    - **File:** `src/components/chart_panel.cpp`
    - **Action:** Render a horizontal bar at the bottom (`ImGui::BeginChild("BottomBar", ImVec2(0, 30))`).
    - **Elements:** Toggles for Volume Profile, Delta, and Cumulative Delta overlays.

---

## Phase 4: Lock-Free State Integration
**Goal:** Ensure the complex UI described above does not destroy performance.

- [ ] **4.1: UI Reads from Atomics**
    - **Rule:** The Quick Order Entry's Best Bid/Ask buttons MUST read from `MarketDataProcessor::get_atomic_snapshot(sym_id)`. They must NOT execute blocking calls to the exchange.
- [ ] **4.2: Trading Actions are Asynchronous**
    - **Rule:** When a user clicks "Buy Market" on the chart, the UI pushes a `TradeCommand` struct into a lock-free Single-Producer-Single-Consumer (SPSC) queue.
    - **Rule:** A background Execution Thread reads this queue and sends it to the exchange. The UI thread never waits for the HTTP/WebSocket response.