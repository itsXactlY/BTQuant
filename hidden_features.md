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

### Phase 1: Menü-Erweiterung (HOHE PRIORITÄT) ✅ ERLEDIGT
- [x] `quant_workspace_component.cpp` - "Add Panel" Menü erweitert (5 Kategorien: Charts, Market Data, Trading, Analysis, System)
- [x] `main_trading_terminal.cpp` - "Layout Presets" Menü hinzugefügt (5 Presets: Default, Modern Trading, Dashboard Only, Chart Focus, Risk Monitoring)
- [x] Fehlende Panel-Typen in Menü aufgenommen

### Phase 2: Panel-Implementierung (ALLE VORHANDEN!) ✅ ERLEDIGT
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

### Phase 3: Zusätzliche Panels ins Enum aufnehmen ✅ ERLEDIGT
- [x] `PanelType::CORRELATION_HEATMAP` - zum Enum hinzugefügt
- [x] `PanelType::DOM_SURFACE` - zum Enum hinzugefügt
- [x] `PanelType::MULTI_VWAP` - zum Enum hinzugefügt
- [x] `PanelType::TECHNICAL_INDICATORS` - zum Enum hinzugefügt
- [x] `PanelType::THEME_CUSTOMIZATION` - zum Enum hinzugefügt
- [x] `PanelType::KEYBOARD_SHORTCUTS` - zum Enum hinzugefügt
- [x] `PanelType::DRAWING_TOOLS` - zum Enum hinzugefügt

### Phase 4: PanelManager::add_panel() erweitern ✅ ERLEDIGT
- [x] Case für alle neuen PanelTypes in [`panel_manager.cpp`](dependencies/BTQ_Render_Engine/src/components/panel_manager.cpp) hinzugefügt
- [x] `get_panel_type_name()` in [`panel_base.cpp`](dependencies/BTQ_Render_Engine/src/components/panel_base.cpp) erweitert
- [x] `get_default_panel_title()` in [`panel_manager.cpp`](dependencies/BTQ_Render_Engine/src/components/panel_manager.cpp) erweitert

### Build-Status
✅ Build erfolgreich: `dependencies/BTQ_Render_Engine/build/BTQuantTerminal`
