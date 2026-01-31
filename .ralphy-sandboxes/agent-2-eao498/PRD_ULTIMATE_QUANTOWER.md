# BTQ Render Engine - Complete Quantower Clone (Final Implementation)

**Objective:** Pixel-perfect clone of Quantower trading terminal with ALL features  
**Constraint:** Work ONLY in `dependencies/BTQ_Render_Engine/`  
**Current State:** Basic panels exist (Chart, Footprint, Orderbook, Time&Sales), need completion  
**Tech Stack:** C++17/20, Vulkan, ImGui, existing HotspineDataBridge  
**Reference:** https://help.quantower.com/quantower/

---

## Phase 1: Core Data Infrastructure (16 Volume Analysis Types)

- [ ] Define complete VolumeAnalysisType enum with all 16 types: Trades, BuyTrades, SellTrades, Volume, BuyVolume, SellVolume, BuyVolumePercent, SellVolumePercent, BuySellVolume, Delta, DeltaPercent, CumulativeDelta, AverageSize, AverageBuySize, AverageSellSize, MaxOneTradeVolume, FilteredVolume in `dependencies/BTQ_Render_Engine/include/data/VolumeDataTypes.h`
- [ ] Create optimized TradeData struct with timestamp (uint64_t), price (double), volume (float), side (enum Buy/Sell), exchange_id (uint8_t), flags (uint8_t bitmask) in `dependencies/BTQ_Render_Engine/include/data/TradeData.h`
- [ ] Implement ClusterCell struct in `dependencies/BTQ_Render_Engine/include/analytics/cluster_engine.hpp` storing: total_volume, buy_volume, sell_volume, trade_count, buy_trade_count, sell_trade_count, max_single_trade_volume, sum_of_volumes (for average calculations)
- [ ] Add ClusterEngine::processTrade method that atomically updates ClusterCell counters for given price level and time bucket in `dependencies/BTQ_Render_Engine/src/analytics/cluster_engine.cpp`
- [ ] Implement VolumeCalculator utility class with static methods: calculateDelta, calculateDeltaPercent, calculateBuyVolumePercent, calculateSellVolumePercent, calculateAverageSize, calculateAverageBuySize, calculateAverageSellSize, calculateMaxOneTradeVolume, calculateFilteredVolume in `dependencies/BTQ_Render_Engine/src/analytics/volume_calculator.cpp`
- [ ] Add CumulativeDeltaTracker class that maintains running sum of delta across time bars with reset functionality for session boundaries in `dependencies/BTQ_Render_Engine/src/analytics/cumulative_delta_tracker.cpp`

## Phase 2: Footprint Chart (Cluster) - Complete Implementation

- [ ] Enhance footprint_panel.cpp main render loop to iterate visible time bars and price levels, retrieve ClusterCell data, and switch on active VolumeAnalysisType to determine displayed value in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Implement diagonal imbalance detection algorithm: compare buy_volume at price P with sell_volume at price P-1, flag cells where ratio exceeds configurable threshold (default 3.0x) in `dependencies/BTQ_Render_Engine/src/analytics/cluster_engine.cpp`
- [ ] Add stacked imbalance detection: vertical analysis comparing buy/sell at same price across consecutive bars in `dependencies/BTQ_Render_Engine/src/analytics/cluster_engine.cpp`
- [ ] Implement adaptive heatmap coloring: calculate alpha based on cell_volume / max_bar_volume, support multiple color schemes (green-red gradient for delta, blue-red for buy/sell, yellow-orange for volume intensity) in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Add imbalance cell highlighting with colored borders (yellow for diagonal, cyan for stacked), thicker border width, and optional glow effect in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Render bar header summary above each cluster bar showing: total volume, net delta, cumulative delta, POC price with monospace font alignment in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Render bar footer showing: number of trades, average trade size, max single trade with smaller font below cluster grid in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Implement cell tooltip on hover displaying: exact buy volume, exact sell volume, delta, delta percent, number of buy trades, number of sell trades, max single trade, timestamp range in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Add data type selector dropdown in footprint panel header with all 16 types, update rendering immediately on selection in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Implement time aggregation selector dropdown: 1min, 5min, 15min, 30min, 1hour, 2hour, 4hour, volume-based (every N contracts), tick-based (every N ticks) in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Add price aggregation selector: 1 tick, 5 ticks, 10 ticks, 0.1%, 0.5%, 1%, custom value in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Implement filtered volume threshold slider: only display cells where volume exceeds threshold, show greyed out cells for filtered values in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Add split volume display mode showing buy volume on left half of cell, sell volume on right half with divider line in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Implement number formatting options: raw numbers, K suffix (thousands), M suffix (millions), scientific notation, custom decimal places in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Add cell size auto-adjustment based on zoom level: expand cells when zoomed in to show more detail, collapse to squares when zoomed out in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Implement Level of Detail (LOD) rendering: skip text rendering when cell height < 12px, show only heatmap colors at extreme zoom out in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`

## Phase 3: Volume Profile - All Four Types

- [ ] Add ProfileMode enum to volume_profile_panel.hpp: Step, Right, Left, Custom with ProfileSettings struct containing: vaPercent (default 70.0), tickStep, showPOC, showValueArea, colorScheme in `dependencies/BTQ_Render_Engine/include/components/volume_profile_panel.hpp`
- [ ] Implement Step Profile rendering: draw mini histogram overlay on each candlestick bar showing volume distribution for that bar's price range in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Calculate and render POC (Point of Control) line for each bar in Step Profile mode: horizontal yellow line at price with highest volume in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Implement Right Profile: aggregate all visible trades into single histogram anchored to right edge of chart, use horizontal bars extending left in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Implement Left Profile: same as Right but anchored to left edge, bars extending right in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Calculate Value Area (VA) for Right/Left profiles: find price range containing 70% of total volume centered around POC, shade this zone with semi-transparent overlay in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Render Value Area High (VAH) and Value Area Low (VAL) horizontal lines with labels in Right/Left profile modes in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Implement Custom Profile with mouse drag interaction: user drags from time A to time B, system calculates and renders profile for that specific range in `dependencies/BTQ_Render_Engine/src/components/interaction_manager.cpp`
- [ ] Add profile anchor markers for Custom Profile: vertical lines at start/end times with drag handles for adjusting range in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Implement split bar rendering in profiles: buy volume on left (green), sell volume on right (red), separated by vertical line at center in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Add profile statistics panel showing: POC price, VAH price, VAL price, total volume in value area, percentage of volume above POC in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Implement profile comparison overlay: show today's profile in full opacity, yesterday's profile in 30% opacity, highlight zones where profiles diverge significantly in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Add session profile mode: automatic reset at session boundaries (00:00 UTC, market open/close times), show separate profiles per session in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`
- [ ] Implement composite profile: aggregate multiple days into single profile showing typical price distribution patterns in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp`

## Phase 4: Time Statistics & Time Histogram

- [ ] Create TimeStatistics panel with ImGui table showing columns: Time, Open, High, Low, Close, Volume, BuyVolume, SellVolume, Delta, Trades, AvgSize, MaxTrade, dynamically show/hide columns based on user selection in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp`
- [ ] Implement sortable columns in TimeStatistics table: click column header to sort ascending/descending by that metric in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp`
- [ ] Add row highlighting in TimeStatistics: highlight row on hover, double-click row to center chart on that time bar in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp`
- [ ] Implement synchronized scrolling: scrolling TimeStatistics table scrolls chart panel to corresponding time range and vice versa in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp`
- [ ] Add color coding in TimeStatistics: green text for positive delta, red for negative, yellow for extreme values (>3 standard deviations) in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp`
- [ ] Create TimeHistogram panel rendering vertical bars below chart for each time bar in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp`
- [ ] Implement BuySellVolume histogram mode: stacked bars with buy volume (green) on top, sell volume (red) on bottom in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp`
- [ ] Implement Delta histogram mode: bars originating from zero line, positive delta extends up (green), negative delta extends down (red) in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp`
- [ ] Implement CumulativeDelta histogram: line chart overlay showing running sum of delta, color transitions from red to green as cumulative delta crosses zero in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp`
- [ ] Add histogram data type selector supporting all 16 VolumeAnalysisType values with instant update on selection in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp`
- [ ] Implement histogram bar tooltip showing exact values when hovering over any bar in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp`
- [ ] Add histogram auto-scaling: automatically adjust Y-axis range to fit visible data, option to lock scale to prevent jumping in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp`

## Phase 5: VWAP Indicators (Multiple Types)

- [ ] Implement calculateVWAP function in technical_analysis.cpp: VWAP = sum(price * volume) / sum(volume) starting from specified index in `dependencies/BTQ_Render_Engine/src/analytics/technical_analysis.cpp`
- [ ] Add calculateVWAPStandardDeviation method computing standard deviation of prices weighted by volume around VWAP in `dependencies/BTQ_Render_Engine/src/analytics/technical_analysis.cpp`
- [ ] Create AnchoredVWAP class storing anchor timestamp, VWAP values array, and standard deviation bands (SD1, SD2, SD3) in `dependencies/BTQ_Render_Engine/include/indicators/anchored_vwap.hpp`
- [ ] Implement AnchoredVWAP::calculate method that processes all bars from anchor point forward, calculating running VWAP and updating bands in `dependencies/BTQ_Render_Engine/src/indicators/anchored_vwap.cpp`
- [ ] Add click-to-anchor interaction in chart panel: user right-clicks bar and selects "Anchor VWAP Here" from context menu to create new anchored VWAP in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Render VWAP line as polyline overlay on chart using distinct color (yellow or cyan), with smooth anti-aliased rendering in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Render VWAP bands (SD1, SD2, SD3) as semi-transparent filled regions above and below VWAP line in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Create MultiVWAP panel managing multiple simultaneous VWAP instances (daily, weekly, monthly, custom anchored) in `dependencies/BTQ_Render_Engine/src/components/multi_vwap_panel.cpp`
- [ ] Implement VWAP list in MultiVWAP panel showing all active VWAPs with checkboxes to show/hide, color picker for line color, delete button in `dependencies/BTQ_Render_Engine/src/components/multi_vwap_panel.cpp`
- [ ] Add session VWAP auto-reset: automatically create new VWAP at session start (market open time), mark previous session VWAP as historical in `dependencies/BTQ_Render_Engine/src/indicators/session_vwap.cpp`
- [ ] Implement rolling VWAP: calculate VWAP for last N bars (sliding window), update continuously as new bars appear in `dependencies/BTQ_Render_Engine/src/indicators/rolling_vwap.cpp`
- [ ] Add VWAP alerts: trigger notification when price crosses VWAP, touches SD2 band, or other configurable conditions in `dependencies/BTQ_Render_Engine/src/indicators/vwap_alerts.cpp`

## Phase 6: Order Book (DOM) Panel - Advanced Features

- [ ] Enhance orderbook_panel.cpp to show full market depth with configurable number of levels (10, 20, 50, 100, 500, unlimited) in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`
- [ ] Implement order book heatmap background rendering using batched geometry: pre-calculate all bar rectangles, draw in single pass before text layer in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`
- [ ] Add liquidity bars showing cumulative volume at each level: horizontal bars extending from price column proportional to volume in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`
- [ ] Implement large order highlighting: detect orders exceeding N% of average order size, display with yellow background and bold text in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`
- [ ] Add order book imbalance indicator: calculate bid/ask ratio, display colored arrow (green for bid heavy, red for ask heavy) in header in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`
- [ ] Implement depth chart visualization mode: alternative view showing depth as area chart (bid area on left, ask area on right) in `dependencies/BTQ_Render_Engine/src/components/depth_chart_panel.cpp`
- [ ] Add order book aggregation modes: group by tick size, 0.1%, 0.5%, 1%, custom value to reduce noise in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`
- [ ] Implement order flow detection: track order additions, cancellations, executions with colored markers showing activity intensity per level in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`
- [ ] Add historical order book snapshots: capture order book state every N seconds, allow playback and comparison with current state in `dependencies/BTQ_Render_Engine/src/components/orderbook_history.cpp`
- [ ] Implement order book delta column showing net change in volume at each level over last N seconds in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`

## Phase 7: Time & Sales Panel - Complete Implementation

- [ ] Enhance time_and_sales.cpp with virtualized scrolling using ImGuiListClipper for handling 100000+ trades efficiently in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Add trade filtering options: filter by minimum size, filter by exchange, filter by time range with UI controls in panel header in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Implement trade coloring schemes: buy trades in green, sell trades in red, large trades (>avg*5) in yellow, block trades (>avg*10) in orange with bold font in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Add trade clustering detection: identify rapid sequences of trades at same price as potential algorithm activity, highlight with background color in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Implement trade size histogram showing distribution of trade sizes with logarithmic buckets in sidebar in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Add trade pace indicator: calculate trades per minute for last 1min, 5min, 15min, display as line chart in header in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Implement trade audio alerts: play sound when large trade executes, different tones for buy vs sell, configurable volume threshold in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Add trade search functionality: search by price range, size range, time range with results highlighted in list in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Implement trade export to CSV with columns: timestamp, exchange, symbol, price, size, side, custom fields in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp`
- [ ] Add historical time & sales: right-click any bar on chart to open popup showing all trades for that specific bar in `dependencies/BTQ_Render_Engine/src/components/historical_time_sales.cpp`

## Phase 8: Chart Panel - Technical Indicators Integration

- [ ] Implement complete technical indicators library in chart_panel.cpp: SMA (9,20,50,200), EMA (9,21,50,200), Bollinger Bands (20,2), RSI (14), MACD (12,26,9), Stochastic (14,3,3), ATR (14) in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Move ALL indicator calculations OUT of render loop into cached_indicators_ map updated only on new data in update() method in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Add indicator overlay panel listing all active indicators with checkboxes for visibility, color pickers, parameter inputs, delete buttons in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Implement indicator alerts: trigger when price crosses SMA, RSI enters oversold/overbought, Bollinger Band touch with notification system in `dependencies/BTQ_Render_Engine/src/indicators/indicator_alerts.cpp`
- [ ] Add multi-timeframe indicator support: display daily SMA on 1-minute chart as context, auto-update on timeframe change in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Optimize crosshair price/time lookup using std::lower_bound binary search replacing current O(n) linear scan in render_crosshair_info in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Implement drawing tools: trend lines, horizontal lines, fibonacci retracements, rectangles, text annotations stored persistently in `dependencies/BTQ_Render_Engine/src/components/drawing_tools.cpp`
- [ ] Add chart replay mode: load historical data and replay bar-by-bar at configurable speed for backtesting practice in `dependencies/BTQ_Render_Engine/src/components/chart_replay.cpp`

## Phase 9: Watchlist Panel - Full Features

- [ ] Create Watchlist panel with multi-column table showing: Symbol, Exchange, Last Price, Change%, Change$, Volume, High, Low, Open, VWAP in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Implement add/remove symbols: input field at top for entering new symbols, delete button per row for removing in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Add drag-and-drop reordering: click and drag rows to reorder watchlist, save order to config file in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Implement real-time price updates: subscribe to price feed for all watchlist symbols, update cell values smoothly with brief flash animation on change in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Add color coding: green for positive change, red for negative, intensity increases with larger changes in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Implement click-to-switch: clicking any symbol in watchlist changes all panels to display that symbol in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Add watchlist groups: create multiple named watchlists (Futures, Crypto, Stocks), tab interface to switch between groups in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Implement column customization: right-click header to show/hide columns, drag columns to reorder in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Add sorting: click column header to sort watchlist by that metric ascending/descending in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`
- [ ] Implement watchlist alerts: trigger notification when symbol price reaches specified level, shows in alerts panel in `dependencies/BTQ_Render_Engine/src/components/watchlist_alerts.cpp`

## Phase 10: Dashboard Controls & Layout Management

- [ ] Create Dashboard Controls panel with buttons to add new panels: Add Chart, Add Footprint, Add Volume Profile, Add Order Book, Add Time&Sales, Add Watchlist, Add News in `dependencies/BTQ_Render_Engine/src/components/dashboard_controls.cpp`
- [ ] Implement symbol selection dropdown in dashboard controls: search-enabled dropdown populated from exchange API, applies to all panels when changed in `dependencies/BTQ_Render_Engine/src/components/dashboard_controls.cpp`
- [ ] Add exchange selector: multi-select dropdown for choosing active exchanges, filters available symbols in `dependencies/BTQ_Render_Engine/src/components/dashboard_controls.cpp`
- [ ] Implement timeframe selector: buttons for 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w affecting all chart-based panels simultaneously in `dependencies/BTQ_Render_Engine/src/components/dashboard_controls.cpp`
- [ ] Add layout presets system: save current panel arrangement as named preset, load preset to restore layout, delete preset in `dependencies/BTQ_Render_Engine/src/ui/layout_manager.cpp`
- [ ] Implement layout quick-save: hotkeys F5-F8 to save layouts, Shift+F5-F8 to load, visual indicator showing active layout in `dependencies/BTQ_Render_Engine/src/ui/layout_manager.cpp`
- [ ] Add panel templates: predefined layouts matching common trading styles (Scalper, Day Trader, Swing Trader, Analysis) one-click load in `dependencies/BTQ_Render_Engine/src/ui/layout_manager.cpp`
- [ ] Implement workspace export/import: save entire workspace (layouts, settings, symbols) to file, share with others, load from file in `dependencies/BTQ_Render_Engine/src/ui/workspace_manager.cpp`

## Phase 11: Settings & Customization System

- [ ] Create unified settings system with categories: Appearance, Data, Performance, Alerts, Keyboard Shortcuts in `dependencies/BTQ_Render_Engine/src/ui/settings_manager.cpp`
- [ ] Implement appearance settings: color themes (Dark, Light, Custom), font selection, font sizes, panel opacity, border styles in `dependencies/BTQ_Render_Engine/src/ui/appearance_settings.cpp`
- [ ] Add data settings: default timeframe, default symbol, auto-load last used workspace, data retention period, WebSocket reconnection settings in `dependencies/BTQ_Render_Engine/src/ui/data_settings.cpp`
- [ ] Implement performance settings: FPS limiter, V-Sync toggle, LOD thresholds, caching strategy, memory limits in `dependencies/BTQ_Render_Engine/src/ui/performance_settings.cpp`
- [ ] Add alert settings: notification method (popup, sound, system tray), alert history size, alert conditions templates in `dependencies/BTQ_Render_Engine/src/ui/alert_settings.cpp`
- [ ] Create keyboard shortcuts editor: list all actions, assign/modify shortcuts, import/export shortcut profiles, reset to defaults in `dependencies/BTQ_Render_Engine/src/ui/keyboard_shortcuts.cpp`
- [ ] Implement per-panel settings: each panel type has dedicated settings modal accessible via right-click context menu or panel header button in respective panel files
- [ ] Add settings persistence: auto-save settings on change to JSON config file, load on startup, settings migration for version updates in `dependencies/BTQ_Render_Engine/src/ui/settings_manager.cpp`

## Phase 12: Advanced Data Features

- [ ] Implement data caching system: cache calculated profiles, indicators, aggregations per bar, invalidate only affected bars on new data in `dependencies/BTQ_Render_Engine/src/data/cache_manager.cpp`
- [ ] Add incremental update mechanism: process only new trades since last update, update only current bar and cumulative values in `dependencies/BTQ_Render_Engine/src/data/incremental_updater.cpp`
- [ ] Implement data compression: compress historical trade data in memory using zstd, decompress on demand for analysis in `dependencies/BTQ_Render_Engine/src/data/data_compression.cpp`
- [ ] Add data persistence: save trade data to disk in chunks, load historical data on demand, implement LRU cache for disk access in `dependencies/BTQ_Render_Engine/src/data/data_persistence.cpp`
- [ ] Create data replay system: load historical trade stream from file, replay at configurable speed (0.5x, 1x, 2x, 10x), pause/resume controls in `dependencies/BTQ_Render_Engine/src/data/data_replay.cpp`
- [ ] Implement multi-exchange aggregation: merge order books and trades from multiple exchanges, resolve price discrepancies, show composite view in `dependencies/BTQ_Render_Engine/src/data/multi_exchange_aggregator.cpp`
- [ ] Add data quality monitoring: detect gaps, duplicate trades, out-of-sequence data, log issues, show warnings in UI in `dependencies/BTQ_Render_Engine/src/data/data_quality_monitor.cpp`

## Phase 13: Performance Optimization (Critical)

- [ ] Implement viewport culling: calculate visible price range and time range, skip rendering for off-screen panels in main render loop in `dependencies/BTQ_Render_Engine/src/main_realtime_dashboard.cpp`
- [ ] Add frustum culling for chart elements: only render candles, indicators, drawings within viewport bounds in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp`
- [ ] Optimize footprint rendering with LOD: render full detail when zoomed in, reduce to heatmap-only at medium zoom, skip rendering at extreme zoom out in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp`
- [ ] Implement geometry batching for order book: pre-calculate all liquidity bar vertices, submit as single draw call using instanced rendering in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp`
- [ ] Add multi-threaded data processing: move volume calculations, profile aggregations, indicator math to worker thread pool (std::async or thread pool) in `dependencies/BTQ_Render_Engine/src/optimization/thread_pool.cpp`
- [ ] Implement lock-free data structures: replace mutex-protected queues with lock-free SPSC queues for WebSocket→UI communication in `dependencies/BTQ_Render_Engine/src/data/lockfree_queue.hpp`
- [ ] Add memory pool allocator for trade objects: pre-allocate chunks of TradeData structs, reuse to avoid malloc overhead in hot path in `dependencies/BTQ_Render_Engine/src/optimization/memory_pool.cpp`
- [ ] Optimize ImGui usage: minimize BeginTable calls, batch text rendering, use clipper for large lists, cache widget sizes in `dependencies/BTQ_Render_Engine/src/optimization/imgui_optimizer.cpp`
- [ ] Add frame pacing: measure frame time budget (16.6ms for 60fps, 6.9ms for 144fps), skip low-priority updates if over budget in `dependencies/BTQ_Render_Engine/src/optimization/frame_pacer.cpp`
- [ ] Implement adaptive quality: automatically reduce detail (LOD, update frequency) when FPS drops below target, restore when stable in `dependencies/BTQ_Render_Engine/src/optimization/adaptive_quality.cpp`

## Phase 14: Performance Monitoring & Debugging

- [ ] Enhance PerformanceMonitor to track per-panel render times: instrument each panel's render() with scoped timer, report to monitor in `dependencies/BTQ_Render_Engine/src/components/*/render()` methods
- [ ] Add frame time graph: real-time scrolling graph showing frame time history with color-coded zones (green <16ms, yellow 16-33ms, red >33ms) in `dependencies/BTQ_Render_Engine/src/monitoring/performance_overlay.cpp`
- [ ] Implement CPU profiler integration: Tracy or Optick markers in hot paths, generate flame graphs, export profile data in `dependencies/BTQ_Render_Engine/src/monitoring/profiler_integration.cpp`
- [ ] Add memory tracking: monitor Vulkan allocation sizes, track largest allocations, detect leaks with growth over time in `dependencies/BTQ_Render_Engine/src/monitoring/memory_tracker.cpp`
- [ ] Create debug overlay panel: toggle with F12, shows FPS, frame time, draw calls, vertex count, active panels, memory usage, data queue sizes in `dependencies/BTQ_Render_Engine/src/monitoring/debug_overlay.cpp`
- [ ] Add data pipeline metrics: WebSocket message rate, trade processing latency, queue depths, dropped messages counter in `dependencies/BTQ_Render_Engine/src/monitoring/pipeline_metrics.cpp`
- [ ] Implement performance regression detection: baseline measurements, alert when key metrics degrade beyond threshold in `dependencies/BTQ_Render_Engine/src/monitoring/regression_detector.cpp`

## Phase 15: UI Polish & User Experience

- [ ] Apply Quantower-style dark theme: background #0a0e1a, panel background #12171f, text #e0e0e0, borders #2a2e3a, accents cyan/magenta in `dependencies/BTQ_Render_Engine/src/ui/theme_manager.cpp`
- [ ] Load monospace font for numeric data: RobotoMono or JetBrainsMono at 13px, regular font for labels at 14px in ImGui initialization
- [ ] Implement smooth animations: panel fade-in on creation, smooth transitions for value changes, easing functions for zoom/pan in `dependencies/BTQ_Render_Engine/src/ui/animation_system.cpp`
- [ ] Add haptic feedback effects: subtle pulse animation when large trade executes, glow effect on imbalance cells, shimmer on alert trigger using shader effects
- [ ] Create loading states: skeleton screens while data loads, progress bars for long operations, spinning indicators for background tasks in respective panels
- [ ] Implement empty states: helpful messages and action buttons when panels have no data (e.g., "No symbols in watchlist - Add one above") in respective panels
- [ ] Add tooltips everywhere: informative hover tooltips on all buttons, settings, indicators explaining functionality in 1-2 sentences
- [ ] Implement context menus: right-click menus on chart (Add Indicator, Drawing Tools), order book (Set Alert), trades (Filter), etc. in respective panels
- [ ] Add quick actions toolbar: floating toolbar with most common actions (Add Chart, Switch Symbol, Change Timeframe) accessible via hotkey (Space)
- [ ] Create first-run tutorial: interactive walkthrough highlighting key features, sample layouts, optional skip in `dependencies/BTQ_Render_Engine/src/ui/tutorial_system.cpp`

## Phase 16: Testing & Quality Assurance

- [ ] Write unit tests for VolumeCalculator: test all 16 calculation methods with edge cases (zero volume, single trade, large numbers) using Google Test in `dependencies/BTQ_Render_Engine/tests/test_volume_calculator.cpp`
- [ ] Create unit tests for ClusterEngine: verify trade aggregation, cell updates, bucket boundaries with mock trade data in `dependencies/BTQ_Render_Engine/tests/test_cluster_engine.cpp`
- [ ] Add unit tests for VWAP calculations: verify formula correctness against known values, test anchored VWAP, rolling VWAP in `dependencies/BTQ_Render_Engine/tests/test_vwap.cpp`
- [ ] Write integration tests for data pipeline: inject mock WebSocket messages, verify data flows through aggregator to UI panels correctly in `dependencies/BTQ_Render_Engine/tests/test_data_pipeline.cpp`
- [ ] Create performance benchmarks: measure footprint calculation time for 10k trades, profile rendering time for 1000 candles, order book update latency in `dependencies/BTQ_Render_Engine/tests/benchmark_performance.cpp`
- [ ] Add stress tests: run with 100k trades, measure memory growth over 24 hours, verify no leaks with Valgrind in `dependencies/BTQ_Render_Engine/tests/stress_test.cpp`
- [ ] Implement visual regression tests: capture screenshots of panels, compare with baseline images, detect unintended visual changes in `dependencies/BTQ_Render_Engine/tests/visual_regression.cpp`
- [ ] Create mock data generator: generate realistic trade streams with varying patterns (trending, ranging, volatile) for testing in `dependencies/BTQ_Render_Engine/tests/mock_data_generator.cpp`
- [ ] Add automated test suite: run all tests in CI, fail build on test failure, generate coverage report (target 80%) in CMakeLists.txt

## Phase 17: Documentation

- [ ] Write comprehensive user guide covering: getting started, panel descriptions, volume analysis concepts, indicator usage, keyboard shortcuts in `dependencies/BTQ_Render_Engine/docs/USER_GUIDE.md`
- [ ] Create API documentation with Doxygen: document all public classes, methods, parameters, return values with examples in header files
- [ ] Write developer guide explaining: architecture overview, adding new panels, implementing indicators, data flow, performance considerations in `dependencies/BTQ_Render_Engine/docs/DEVELOPER_GUIDE.md`
- [ ] Document all 16 volume data types: explain calculation formulas, use cases, interpretation tips with visual examples in `dependencies/BTQ_Render_Engine/docs/VOLUME_DATA_TYPES.md`
- [ ] Create performance tuning guide: optimization strategies, profiling tools, common bottlenecks, hardware recommendations in `dependencies/BTQ_Render_Engine/docs/PERFORMANCE_TUNING.md`
- [ ] Write troubleshooting guide: common issues, solutions, FAQ, known limitations, workarounds in `dependencies/BTQ_Render_Engine/docs/TROUBLESHOOTING.md`
- [ ] Add code examples: snippets showing how to extend system with custom indicators, panels, data sources in `dependencies/BTQ_Render_Engine/docs/examples/`
- [ ] Create video tutorials: screen recordings demonstrating key features, uploaded to docs folder or linked in README

## Phase 18: Final Integration & Cleanup

- [ ] Audit entire codebase: remove commented-out code, dead functions, unused includes, fix compiler warnings in all files
- [ ] Standardize naming conventions: consistent naming across all files (snake_case for variables/functions, PascalCase for classes) in all files
- [ ] Add comprehensive logging: structured logging with levels (DEBUG, INFO, WARN, ERROR), log rotation, configurable verbosity in `dependencies/BTQ_Render_Engine/src/utils/logger.cpp`
- [ ] Implement error handling: try-catch blocks around risky operations, graceful degradation, user-friendly error messages in all components
- [ ] Add input validation: validate all user inputs (symbol names, numeric ranges, file paths), sanitize before use in all UI components
- [ ] Create release build configuration: compiler optimizations (-O3), strip debug symbols, enable LTO, minimize binary size in CMakeLists.txt
- [ ] Add crash reporting: integrate crash handler (Breakpad or custom), generate minidumps, upload to server in `dependencies/BTQ_Render_Engine/src/utils/crash_handler.cpp`
- [ ] Implement telemetry: collect anonymous usage statistics (features used, performance metrics), opt-in during first run in `dependencies/BTQ_Render_Engine/src/telemetry/telemetry_client.cpp`
- [ ] Run final QA pass: test all features, verify all panels work, check performance targets met, no crashes during extended use
- [ ] Create release package: bundle executable, required DLLs, default config, sample layouts, documentation as installer

---

## Success Criteria (Gnadenlos)

**Functionality:**
- ✅ All 16 volume data types implemented and accurate
- ✅ Footprint chart renders 500+ bars × 200+ price levels smoothly
- ✅ Volume profiles (Step, Right, Left, Custom) functional with correct POC/VA calculations
- ✅ Time Statistics and Time Histogram display all metrics correctly
- ✅ VWAP (Anchored, Multi, Session, Rolling) calculated accurately
- ✅ Order Book shows full depth with heatmap and large order detection
- ✅ Time & Sales handles 100k+ trades with filtering and search
- ✅ Chart panel has 20+ technical indicators with alerts
- ✅ Watchlist supports unlimited symbols with real-time updates
- ✅ Dashboard controls allow adding any panel type
- ✅ Layout system saves/loads configurations
- ✅ Settings cover all customization options

**Performance:**
- ✅ Maintains 144 FPS with all panels active and streaming data
- ✅ Frame time < 6.9ms (99th percentile)
- ✅ Data latency < 30ms (WebSocket message → UI update)
- ✅ Memory usage < 500MB with 100k trades loaded
- ✅ CPU usage < 25% on modern 8-core processor
- ✅ No memory leaks over 72-hour stress test
- ✅ Footprint calculation < 50ms for 10k trades
- ✅ Volume profile calculation < 10ms for 10k trades
- ✅ Order book update < 5ms for 1000-level book

**Quality:**
- ✅ Zero crashes during 48-hour soak test
- ✅ All compiler warnings resolved
- ✅ 80%+ code coverage in tests
- ✅ All unit tests passing
- ✅ Performance benchmarks meet targets
- ✅ Valgrind reports zero memory leaks
- ✅ Visual regression tests passing

**User Experience:**
- ✅ UI matches Quantower visual style (dark theme, cyan/magenta accents)
- ✅ All panels have comprehensive tooltips and help text
- ✅ Smooth animations and transitions
- ✅ Intuitive keyboard shortcuts
- ✅ Comprehensive documentation (user + developer guides)
- ✅ First-run tutorial implemented
- ✅ Settings persist across sessions

**Completeness:**
- ✅ No placeholder comments in code
- ✅ No TODO markers remaining
- ✅ All functions documented with Doxygen
- ✅ All features from Quantower reference implemented
- ✅ Release build configuration complete
- ✅ Installer/package created

---

## Notes

- Work exclusively in `dependencies/BTQ_Render_Engine/`
- Use existing Vulkan renderer, ImGui integration, HotspineDataBridge
- Prioritize performance: profile frequently, optimize hot paths
- Test incrementally: verify each feature before moving to next
- Match Quantower behavior and visuals exactly
- When in doubt, refer to Quantower documentation: https://help.quantower.com/

---

**This PRD contains 200+ detailed tasks covering every aspect of a complete Quantower clone. Execute in order, verify each task, iterate until gnadenlos fertig.**