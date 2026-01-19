# BTQ Render Engine - Archived Prototypes

This directory contains historical prototype implementations that were superseded by the current QuantWorkspace architecture.

## Archive Date: 2026-01-20

## Prototype 1: Offscreen Render System

**Files**:
- `CandlePipeline.{h,cpp}` - Direct Vulkan candle rendering pipeline
- `DashboardLayer.{h,cpp}` - Original dashboard with camera controls
- `OffscreenChartRenderer.{h,cpp}` - Vulkan offscreen framebuffer rendering

**Architecture**: Direct Vulkan rendering to offscreen framebuffers

**Era**: December 2025 - Early January 2026

**Why Archived**: Replaced by ImPlot-based rendering in QuantWorkspace, which provides better abstraction and faster development.

---

## Prototype 2: Discrete Component System

**Files**:
- `trading_components.cpp` - TradingInterfaceComponent, TapeComponent, OrderManagementComponent
- `alert_component.cpp` - Alert rule management UI
- `data_grid_component.cpp` - Generic data grid widget
- `depth_chart_component.cpp` - Order book depth visualization
- `heatmap_component.cpp` - Volume/price heatmap visualization
- `keyboard_shortcuts_component.cpp` - Keyboard shortcut configuration UI
- `log_display_component.cpp` - System log viewer
- `market_screener_component.cpp` - Symbol screener UI
- `technical_indicators_component.cpp` - Indicator configuration panel
- `theme_customization_component.cpp` - Theme editor UI
- `watchlist_component.cpp` - Symbol watchlist management

**Architecture**: 14+ discrete UIComponent subclasses, each handling a specific UI panel

**Era**: Mid January 2026

**Why Archived**: Replaced by unified QuantWorkspace which consolidates functionality into ChartManager + IndicatorRenderer. The discrete component approach had too much duplicate boilerplate and made data flow complex.

**Note**: Some implementations (OrderManager, PositionManager, RiskAssessment) from this era were valuable and were extracted into the `src/trading/` module.

---

## Current Architecture: QuantWorkspace (Prototype 3)

**Active Files**:
- `src/components/quant_workspace_component.cpp`
- `src/components/chart_manager.cpp`
- `src/components/indicator_renderer.cpp`

**Why This Won**: 
- ✅ Vulkan compute shader support for indicators
- ✅ Clean data flow: HotSpineDataBridge → MarketDataProcessor → ChartManager
- ✅ ImPlot integration for professional charting
- ✅ Modular: Chart management separate from indicator rendering
- ✅ Scalable: Easy to add new chart types and indicators

---

## Recovery Instructions

If you need to reference or restore code from these archives:

1. **View archived code**:
   ```bash
   git show HEAD:archive/prototype_1_offscreen_render/DashboardLayer.cpp
   ```

2. **Extract specific functions**:
   - Most trading logic is in `trading_components.cpp`
   - Pattern recognition is in the monolithic `vulkan_dashboard_advanced.hpp`

3. **Restore a file temporarily**:
   ```bash
   git checkout HEAD -- archive/prototype_2_discrete_components/heatmap_component.cpp
   # Copy what you need, then:
   git checkout HEAD -- .
   ```

---

## Historical Context

These prototypes represent rapid iteration during the development of BTQuant's real-time trading terminal. Each architectural shift brought improvements:

- **Prototype 1 → 2**: Needed UI modularity for trading features
- **Prototype 2 → 3**: Needed performance and simpler data flow for real-time indicators

The git commit `bca875e` notes: "huge broken mixed up prototype mess before cleanup routine" - this archive is the result of that cleanup.
