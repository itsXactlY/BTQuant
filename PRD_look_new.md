
# BTQ Render Engine - Complete Quantower Clone (Parallel Development Version)

**Objective:** Pixel-perfect clone of Quantower trading terminal with ALL features  
**Current State:** Basic panels exist (Chart, Footprint, Orderbook, Time&Sales), need completion  
**Tech Stack:** C++23/26, Vulkan, ImGui, existing HotspineDataBridge  
**Reference:** https://help.quantower.com/quantower/  

---

## Development Guidelines
- Each module can be developed in parallel by separate developers
- Dependencies between modules are clearly marked
- Estimated complexity: S (Small, 1-2 days), M (Medium, 3-5 days), L (Large, 1+ weeks)
- Prerequisites must be completed before starting dependent tasks
- Your workdir is exlusivly dependencies/BTQ_Render_Engine/
- Fully autonomous handle merge conflicts in the most harmonic way
---

## Module 1: Core Data Infrastructure [Complexity: L]
**Dependencies:** None  
**Prerequisites:** Basic understanding of trade data structures  

### Phase 1a: Volume Data Types [Complexity: M]
- [x] Define complete VolumeAnalysisType enum with all 16 types: Trades, BuyTrades, SellTrades, Volume, BuyVolume, SellVolume, BuyVolumePercent, SellVolumePercent, BuySellVolume, Delta, DeltaPercent, CumulativeDelta, AverageSize, AverageBuySize, AverageSellSize, MaxOneTradeVolume, FilteredVolume in `dependencies/BTQ_Render_Engine/include/data/VolumeDataTypes.h` [Complexity: S]
- [x] Create optimized TradeData struct with timestamp (uint64_t), price (double), volume (float), side (enum Buy/Sell), exchange_id (uint8_t), flags (uint8_t bitmask) in `dependencies/BTQ_Render_Engine/include/data/TradeData.h` [Complexity: S]
- [x] Implement ClusterCell struct in `dependencies/BTQ_Render_Engine/include/analytics/cluster_engine.hpp` storing: total_volume, buy_volume, sell_volume, trade_count, buy_trade_count, sell_trade_count, max_single_trade_volume, sum_of_volumes (for average calculations) [Complexity: S]

### Phase 1b: Volume Calculations [Complexity: M]
- [x] Add ClusterEngine::processTrade method that atomically updates ClusterCell counters for given price level and time bucket in `dependencies/BTQ_Render_Engine/src/analytics/cluster_engine.cpp` [Complexity: M]
- [x] Implement VolumeCalculator utility class with static methods: calculateDelta, calculateDeltaPercent, calculateBuyVolumePercent, calculateSellVolumePercent, calculateAverageSize, calculateAverageBuySize, calculateAverageSellSize, calculateMaxOneTradeVolume, calculateFilteredVolume in `dependencies/BTQ_Render_Engine/src/analytics/volume_calculator.cpp` [Complexity: M]
- [x] Add CumulativeDeltaTracker class that maintains running sum of delta across time bars with reset functionality for session boundaries in `dependencies/BTQ_Render_Engine/src/analytics/cumulative_delta_tracker.cpp` [Complexity: M]

---

## Module 2: Footprint Chart (Cluster) [Complexity: L]
**Dependencies:** Module 1  
**Prerequisites:** Core data infrastructure complete  

### Phase 2a: Basic Rendering [Complexity: M]
- [x] Enhance footprint_panel.cpp main render loop to iterate visible time bars and price levels, retrieve ClusterCell data, and switch on active VolumeAnalysisType to determine displayed value in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: M]
- [x] Implement adaptive heatmap coloring: calculate alpha based on cell_volume / max_bar_volume, support multiple color schemes (green-red gradient for delta, blue-red for buy/sell, yellow-orange for volume intensity) in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: M]
- [x] Add data type selector dropdown in footprint panel header with all 16 types, update rendering immediately on selection in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: S]

### Phase 2b: Advanced Features [Complexity: M]
- [x] Implement diagonal imbalance detection algorithm: compare buy_volume at price P with sell_volume at price P-1, flag cells where ratio exceeds configurable threshold (default 3.0x) in `dependencies/BTQ_Render_Engine/src/analytics/cluster_engine.cpp` [Complexity: M]
- [x] Add stacked imbalance detection: vertical analysis comparing buy/sell at same price across consecutive bars in `dependencies/BTQ_Render_Engine/src/analytics/cluster_engine.cpp` [Complexity: M]
- [x] Add imbalance cell highlighting with colored borders (yellow for diagonal, cyan for stacked), thicker border width, and optional glow effect in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: S]

### Phase 2c: UI & UX [Complexity: S]
- [x] Render bar header summary above each cluster bar showing: total volume, net delta, cumulative delta, POC price with monospace font alignment in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: S]
- [x] Render bar footer showing: number of trades, average trade size, max single trade with smaller font below cluster grid in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: S]
- [x] Implement cell tooltip on hover displaying: exact buy volume, exact sell volume, delta, delta percent, number of buy trades, number of sell trades, max single trade, timestamp range in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: S]
- [x] Implement number formatting options: raw numbers, K suffix (thousands), M suffix (millions), scientific notation, custom decimal places in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: S]

### Phase 2d: Configuration Options [Complexity: M]
- [x] Implement time aggregation selector dropdown: 1min, 5min, 15min, 30min, 1hour, 2hour, 4hour, volume-based (every N contracts), tick-based (every N ticks) in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: M]
- [x] Add price aggregation selector: 1 tick, 5 ticks, 10 ticks, 0.1%, 0.5%, 1%, custom value in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: M]
- [x] Implement filtered volume threshold slider: only display cells where volume exceeds threshold, show greyed out cells for filtered values in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: M]
- [x] Add split volume display mode showing buy volume on left half of cell, sell volume on right half with divider line in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: M]
- [x] Implement cell size auto-adjustment based on zoom level: expand cells when zoomed in to show more detail, collapse to squares when zoomed out in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: M]
- [x] Implement Level of Detail (LOD) rendering: skip text rendering when cell height < 12px, show only heatmap colors at extreme zoom out in `dependencies/BTQ_Render_Engine/src/components/footprint_panel.cpp` [Complexity: M]

---

## Module 3: Volume Profile [Complexity: M]
**Dependencies:** Module 1  
**Prerequisites:** Core data infrastructure complete  

### Phase 3a: Profile Types [Complexity: M]
- [x] Add ProfileMode enum to volume_profile_panel.hpp: Step, Right, Left, Custom with ProfileSettings struct containing: vaPercent (default 70.0), tickStep, showPOC, showValueArea, colorScheme in `dependencies/BTQ_Render_Engine/include/components/volume_profile_panel.hpp` [Complexity: S]
- [x] Implement Step Profile rendering: draw mini histogram overlay on each candlestick bar showing volume distribution for that bar's price range in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]
- [x] Calculate and render POC (Point of Control) line for each bar in Step Profile mode: horizontal yellow line at price with highest volume in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: S]
- [x] Implement Right Profile: aggregate all visible trades into single histogram anchored to right edge of chart, use horizontal bars extending left in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]
- [x] Implement Left Profile: same as Right but anchored to left edge, bars extending right in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]

### Phase 3b: Value Area & Advanced Features [Complexity: M]
- [x] Calculate Value Area (VA) for Right/Left profiles: find price range containing 70% of total volume centered around POC, shade this zone with semi-transparent overlay in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]
- [x] Render Value Area High (VAH) and Value Area Low (VAL) horizontal lines with labels in Right/Left profile modes in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: S]
- [x] Implement split bar rendering in profiles: buy volume on left (green), sell volume on right (red), separated by vertical line at center in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]
- [x] Add profile statistics panel showing: POC price, VAH price, VAL price, total volume in value area, percentage of volume above POC in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]

### Phase 3c: Custom & Comparison Features [Complexity: M]
- [x] Implement Custom Profile with mouse drag interaction: user drags from time A to time B, system calculates and renders profile for that specific range in `dependencies/BTQ_Render_Engine/src/components/interaction_manager.cpp` [Complexity: M]
- [x] Add profile anchor markers for Custom Profile: vertical lines at start/end times with drag handles for adjusting range in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: S]
- [x] Implement profile comparison overlay: show today's profile in full opacity, yesterday's profile in 30% opacity, highlight zones where profiles diverge significantly in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]
- [x] Add session profile mode: automatic reset at session boundaries (00:00 UTC, market open/close times), show separate profiles per session in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]
- [x] Implement composite profile: aggregate multiple days into single profile showing typical price distribution patterns in `dependencies/BTQ_Render_Engine/src/components/volume_profile_panel.cpp` [Complexity: M]

---

## Module 4: Time Statistics & Time Histogram [Complexity: M]
**Dependencies:** Module 1  
**Prerequisites:** Core data infrastructure complete  

### Phase 4a: Time Statistics [Complexity: M]
- [x] Create TimeStatistics panel with ImGui table showing columns: Time, Open, High, Low, Close, Volume, BuyVolume, SellVolume, Delta, Trades, AvgSize, MaxTrade, dynamically show/hide columns based on user selection in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp` [Complexity: M]
- [x] Implement sortable columns in TimeStatistics table: click column header to sort ascending/descending by that metric in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp` [Complexity: S]
- [x] Add row highlighting in TimeStatistics: highlight row on hover, double-click row to center chart on that time bar in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp` [Complexity: S]
- [x] Implement synchronized scrolling: scrolling TimeStatistics table scrolls chart panel to corresponding time range and vice versa in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp` [Complexity: M]
- [x] Add color coding in TimeStatistics: green text for positive delta, red for negative, yellow for extreme values (>3 standard deviations) in `dependencies/BTQ_Render_Engine/src/components/time_statistics_panel.cpp` [Complexity: S]

### Phase 4b: Time Histogram [Complexity: M]
- [x] Create TimeHistogram panel rendering vertical bars below chart for each time bar in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp` [Complexity: M]
- [x] Implement BuySellVolume histogram mode: stacked bars with buy volume (green) on top, sell volume (red) on bottom in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp` [Complexity: M]
- [x] Implement Delta histogram mode: bars originating from zero line, positive delta extends up (green), negative delta extends down (red) in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp` [Complexity: M]
- [x] Implement CumulativeDelta histogram: line chart overlay showing running sum of delta, color transitions from red to green as cumulative delta crosses zero in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp` [Complexity: M]
- [x] Add histogram data type selector supporting all 16 VolumeAnalysisType values with instant update on selection in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp` [Complexity: S]
- [x] Implement histogram bar tooltip showing exact values when hovering over any bar in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp` [Complexity: S]
- [x] Add histogram auto-scaling: automatically adjust Y-axis range to fit visible data, option to lock scale to prevent jumping in `dependencies/BTQ_Render_Engine/src/components/time_histogram_panel.cpp` [Complexity: M]

---

## Module 5: VWAP Indicators [Complexity: M]
**Dependencies:** Module 1  
**Prerequisites:** Core data infrastructure complete  

### Phase 5a: VWAP Calculation [Complexity: M]
- [x] Implement calculateVWAP function in technical_analysis.cpp: VWAP = sum(price * volume) / sum(volume) starting from specified index in `dependencies/BTQ_Render_Engine/src/analytics/technical_analysis.cpp` [Complexity: M]
- [x] Add calculateVWAPStandardDeviation method computing standard deviation of prices weighted by volume around VWAP in `dependencies/BTQ_Render_Engine/src/analytics/technical_analysis.cpp` [Complexity: M]
- [x] Create AnchoredVWAP class storing anchor timestamp, VWAP values array, and standard deviation bands (SD1, SD2, SD3) in `dependencies/BTQ_Render_Engine/include/indicators/anchored_vwap.hpp` [Complexity: M]
- [x] Implement AnchoredVWAP::calculate method that processes all bars from anchor point forward, calculating running VWAP and updating bands in `dependencies/BTQ_Render_Engine/src/indicators/anchored_vwap.cpp` [Complexity: M]

### Phase 5b: VWAP Visualization [Complexity: M]
- [x] Add click-to-anchor interaction in chart panel: user right-clicks bar and selects "Anchor VWAP Here" from context menu to create new anchored VWAP in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp` [Complexity: M]
- [x] Render VWAP line as polyline overlay on chart using distinct color (yellow or cyan), with smooth anti-aliased rendering in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp` [Complexity: M]
- [x] Render VWAP bands (SD1, SD2, SD3) as semi-transparent filled regions above and below VWAP line in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp` [Complexity: M]
- [x] Create MultiVWAP panel managing multiple simultaneous VWAP instances (daily, weekly, monthly, custom anchored) in `dependencies/BTQ_Render_Engine/src/components/multi_vwap_panel.cpp` [Complexity: M]
- [x] Implement VWAP list in MultiVWAP panel showing all active VWAPs with checkboxes to show/hide, color picker for line color, delete button in `dependencies/BTQ_Render_Engine/src/components/multi_vwap_panel.cpp` [Complexity: S]

### Phase 5c: VWAP Management [Complexity: M]
- [x] Add session VWAP auto-reset: automatically create new VWAP at session start (market open time), mark previous session VWAP as historical in `dependencies/BTQ_Render_Engine/src/indicators/session_vwap.cpp` [Complexity: M]
- [x] Implement rolling VWAP: calculate VWAP for last N bars (sliding window), update continuously as new bars appear in `dependencies/BTQ_Render_Engine/src/indicators/rolling_vwap.cpp` [Complexity: M]
- [x] Add VWAP alerts: trigger notification when price crosses VWAP, touches SD2 band, or other configurable conditions in `dependencies/BTQ_Render_Engine/src/indicators/vwap_alerts.cpp` [Complexity: M]

---

## Module 6: Order Book (DOM) Panel [Complexity: M]
**Dependencies:** Module 1  
**Prerequisites:** Core data infrastructure complete  

### Phase 6a: Basic Features [Complexity: M]
- [x] Enhance orderbook_panel.cpp to show full market depth with configurable number of levels (10, 20, 50, 100, 500, unlimited) in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp` [Complexity: M]
- [x] Implement order book heatmap background rendering using batched geometry: pre-calculate all bar rectangles, draw in single pass before text layer in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp` [Complexity: M]
- [x] Add liquidity bars showing cumulative volume at each level: horizontal bars extending from price column proportional to volume in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp` [Complexity: M]
- [x] Implement large order highlighting: detect orders exceeding N% of average order size, display with yellow background and bold text in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp` [Complexity: S]

### Phase 6b: Advanced Features [Complexity: M]
- [x] Add order book imbalance indicator: calculate bid/ask ratio, display colored arrow (green for bid heavy, red for ask heavy) in header in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp` [Complexity: S]
- [x] Implement order book aggregation modes: group by tick size, 0.1%, 0.5%, 1%, custom value to reduce noise in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp` [Complexity: M]
- [x] Implement order flow detection: track order additions, cancellations, executions with colored markers showing activity intensity per level in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp` [Complexity: M]
- [x] Add order book delta column showing net change in volume at each level over last N seconds in `dependencies/BTQ_Render_Engine/src/components/orderbook_panel.cpp` [Complexity: M]

### Phase 6c: History & Visualization [Complexity: M]
- [x] Add historical order book snapshots: capture order book state every N seconds, allow playback and comparison with current state in `dependencies/BTQ_Render_Engine/src/components/orderbook_history.cpp` [Complexity: M]
- [x] Implement depth chart visualization mode: alternative view showing depth as area chart (bid area on left, ask area on right) in `dependencies/BTQ_Render_Engine/src/components/depth_chart_panel.cpp` [Complexity: M]

---

## Module 7: Time & Sales Panel [Complexity: M]
**Dependencies:** Module 1  
**Prerequisites:** Core data infrastructure complete  

### Phase 7a: Core Functionality [Complexity: M]
- [x] Enhance time_and_sales.cpp with virtualized scrolling using ImGuiListClipper for handling 100000+ trades efficiently in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: M]
- [x] Add trade filtering options: filter by minimum size, filter by exchange, filter by time range with UI controls in panel header in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: M]
- [x] Implement trade coloring schemes: buy trades in green, sell trades in red, large trades (>avg*5) in yellow, block trades (>avg*10) in orange with bold font in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: S]
- [x] Add trade clustering detection: identify rapid sequences of trades at same price as potential algorithm activity, highlight with background color in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: M]

### Phase 7b: Analytics & Search [Complexity: M]
- [x] Implement trade size histogram showing distribution of trade sizes with logarithmic buckets in sidebar in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: M]
- [x] Add trade pace indicator: calculate trades per minute for last 1min, 5min, 15min, display as line chart in header in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: M]
- [x] Add trade search functionality: search by price range, size range, time range with results highlighted in list in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: M]
- [x] Implement trade export to CSV with columns: timestamp, exchange, symbol, price, size, side, custom fields in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: M]

### Phase 7c: Audio & Historical Data [Complexity: M]
- [x] Implement trade audio alerts: play sound when large trade executes, different tones for buy vs sell, configurable volume threshold in `dependencies/BTQ_Render_Engine/src/components/time_and_sales.cpp` [Complexity: M]
- [x] Add historical time & sales: right-click any bar on chart to open popup showing all trades for that specific bar in `dependencies/BTQ_Render_Engine/src/components/historical_time_sales.cpp` [Complexity: M]

---

## Module 8: Chart Panel & Technical Indicators [Complexity: L]
**Dependencies:** Module 1  
**Prerequisites:** Core data infrastructure complete  

### Phase 8a: Indicator Library [Complexity: L]
- [x] Implement complete technical indicators library in chart_panel.cpp: SMA (9,20,50,200), EMA (9,21,50,200), Bollinger Bands (20,2), RSI (14), MACD (12,26,9), Stochastic (14,3,3), ATR (14) in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp` [Complexity: L]
- [x] Move ALL indicator calculations OUT of render loop into cached_indicators_ map updated only on new data in update() method in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp` [Complexity: M]
- [x] Add indicator overlay panel listing all active indicators with checkboxes for visibility, color pickers, parameter inputs, delete buttons in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp` [Complexity: M]

### Phase 8b: Advanced Chart Features [Complexity: M]
- [x] Implement indicator alerts: trigger when price crosses SMA, RSI enters oversold/overbought, Bollinger Band touch with notification system in `dependencies/BTQ_Render_Engine/src/indicators/indicator_alerts.cpp` [Complexity: M]
- [x] Add multi-timeframe indicator support: display daily SMA on 1-minute chart as context, auto-update on timeframe change in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp` [Complexity: M]
- [x] Optimize crosshair price/time lookup using std::lower_bound binary search replacing current O(n) linear scan in render_crosshair_info in `dependencies/BTQ_Render_Engine/src/components/chart_panel.cpp` [Complexity: S]

### Phase 8c: Drawing Tools & Replay [Complexity: M]
- [x] Implement drawing tools: trend lines, horizontal lines, fibonacci retracements, rectangles, text annotations stored persistently in `dependencies/BTQ_Render_Engine/src/components/drawing_tools.cpp` [Complexity: M]
- [x] Add chart replay mode: load historical data and replay bar-by-bar at configurable speed for backtesting practice in `dependencies/BTQ_Render_Engine/src/components/chart_replay.cpp` [Complexity: M]

---

## Module 9: Watchlist Panel [Complexity: M]
**Dependencies:** None (can be developed in parallel)  
**Prerequisites:** Basic understanding of UI components  

### Phase 9a: Basic Watchlist [Complexity: M]
- [x] Create Watchlist panel with multi-column table showing: Symbol, Exchange, Last Price, Change%, Change$, Volume, High, Low, Open, VWAP in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: M]
- [x] Implement add/remove symbols: input field at top for entering new symbols, delete button per row for removing in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: S]
- [x] Add drag-and-drop reordering: click and drag rows to reorder watchlist, save order to config file in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: M]
- [x] Implement real-time price updates: subscribe to price feed for all watchlist symbols, update cell values smoothly with brief flash animation on change in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: M]

### Phase 9b: Enhanced Features [Complexity: M]
- [x] Add color coding: green for positive change, red for negative, intensity increases with larger changes in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: S]
- [x] Implement click-to-switch: clicking any symbol in watchlist changes all panels to display that symbol in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: M]
- [x] Add watchlist groups: create multiple named watchlists (Futures, Crypto, Stocks), tab interface to switch between groups in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: M]
- [x] Implement column customization: right-click header to show/hide columns, drag columns to reorder in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: M]
- [x] Add sorting: click column header to sort watchlist by that metric ascending/descending in `dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp` [Complexity: S]

### Phase 9c: Alerts & Notifications [Complexity: M]
- [x] Implement watchlist alerts: trigger notification when symbol price reaches specified level, shows in alerts panel in `dependencies/BTQ_Render_Engine/src/components/watchlist_alerts.cpp` [Complexity: M]

---

## Module 10: Dashboard Controls & Layout Management [Complexity: M]
**Dependencies:** Modules 2, 3, 4, 6, 7, 9  
**Prerequisites:** At least basic versions of other panels complete  

### Phase 10a: Panel Creation [Complexity: S]
- [ ] Create Dashboard Controls panel with buttons to add new panels: Add Chart, Add Footprint, Add Volume Profile, Add Order Book, Add Time&Sales, Add Watchlist, Add News in `dependencies/BTQ_Render_Engine/src/components/dashboard_controls.cpp` [Complexity: M]
- [ ] Implement symbol selection dropdown in dashboard controls: search-enabled dropdown populated from exchange API, applies to all panels when changed in `dependencies/BTQ_Render_Engine/src/components/dashboard_controls.cpp` [Complexity: M]
- [ ] Add exchange selector: multi-select dropdown for choosing active exchanges, filters available symbols in `dependencies/BTQ_Render_Engine/src/components/dashboard_controls.cpp` [Complexity: M]
- [ ] Implement timeframe selector: buttons for 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w affecting all chart-based panels simultaneously in `dependencies/BTQ_Render_Engine/src/components/dashboard_controls.cpp` [Complexity: M]

### Phase 10b: Layout Management [Complexity: M]
- [ ] Add layout presets system: save current panel arrangement as named preset, load preset to restore layout, delete preset in `dependencies/BTQ_Render_Engine/src/ui/layout_manager.cpp` [Complexity: M]
- [x] Implement layout quick-save: hotkeys F5-F8 to save layouts, Shift+F5-F8 to load, visual indicator showing active layout in `dependencies/BTQ_Render_Engine/src/ui/layout_manager.cpp` [Complexity: S]
- [ ] Add panel templates: predefined layouts matching common trading styles (Scalper, Day Trader, Swing Trader, Analysis) one-click load in `dependencies/BTQ_Render_Engine/src/ui/layout_manager.cpp` [Complexity: M]
- [ ] Implement workspace export/import: save entire workspace (layouts, settings, symbols) to file, share with others, load from file in `dependencies/BTQ_Render_Engine/src/ui/workspace_manager.cpp` [Complexity: M]

---

## Module 11: Settings & Customization System [Complexity: M]
**Dependencies:** None (can be developed in parallel)  
**Prerequisites:** Basic understanding of configuration systems  

### Phase 11a: Settings Framework [Complexity: M]
- [ ] Create unified settings system with categories: Appearance, Data, Performance, Alerts, Keyboard Shortcuts in `dependencies/BTQ_Render_Engine/src/ui/settings_manager.cpp` [Complexity: M]
- [ ] Implement appearance settings: color themes (Dark, Light, Custom), font selection, font sizes, panel opacity, border styles in `dependencies/BTQ_Render_Engine/src/ui/appearance_settings.cpp` [Complexity: M]
- [ ] Add data settings: default timeframe, default symbol, auto-load last used workspace, data retention period, WebSocket reconnection settings in `dependencies/BTQ_Render_Engine/src/ui/data_settings.cpp` [Complexity: M]
- [ ] Implement performance settings: FPS limiter, V-Sync toggle, LOD thresholds, caching strategy, memory limits in `dependencies/BTQ_Render_Engine/src/ui/performance_settings.cpp` [Complexity: M]

### Phase 11b: Advanced Settings [Complexity: M]
- [ ] Add alert settings: notification method (popup, sound, system tray), alert history size, alert conditions templates in `dependencies/BTQ_Render_Engine/src/ui/alert_settings.cpp` [Complexity: M]
- [ ] Create keyboard shortcuts editor: list all actions, assign/modify shortcuts, import/export shortcut profiles, reset to defaults in `dependencies/BTQ_Render_Engine/src/ui/keyboard_shortcuts.cpp` [Complexity: M]
- [ ] Implement per-panel settings: each panel type has dedicated settings modal accessible via right-click context menu or panel header button in respective panel files [Complexity: M]
- [ ] Add settings persistence: auto-save settings on change to JSON config file, load on startup, settings migration for version updates in `dependencies/BTQ_Render_Engine/src/ui/settings_manager.cpp` [Complexity: M]

---

## Module 12: Advanced Data Features [Complexity: L]
**Dependencies:** Module 1  
**Prerequisites:** Core data infrastructure complete  

### Phase 12a: Caching & Optimization [Complexity: M]
- [ ] Implement data caching system: cache calculated profiles, indicators, aggregations per bar, invalidate on new data in `dependencies/BTQ_Render_Engine/src/data/cache_manager.cpp` [Complexity: M]
- [ ] Add incremental data updates: only recalculate affected portions when new trade arrives, maintain running totals for all aggregations in `dependencies/BTQ_Render_Engine/src/data/incremental_updater.cpp` [Complexity: L]
- [ ] Implement data compression for historical data: use delta compression, quantization for price/volume, store differences rather than absolute values in `dependencies/BTQ_Render_Engine/src/data/compression.cpp` [Complexity: M]
- [ ] Add data persistence: save aggregated data to disk periodically, load on startup, support for multiple data sources in `dependencies/BTQ_Render_Engine/src/data/persistence.cpp` [Complexity: M]

### Phase 12b: Multi-Exchange & Quality [Complexity: M]
- [ ] Add multi-exchange aggregation: combine data from multiple exchanges into single view, handle time synchronization, account for exchange-specific features in `dependencies/BTQ_Render_Engine/src/data/exchange_aggregator.cpp` [Complexity: M]
- [ ] Implement data quality monitoring: detect missing data, duplicate trades, out-of-order timestamps, latency issues, alert user to data problems in `dependencies/BTQ_Render_Engine/src/data/quality_monitor.cpp` [Complexity: M]

---

## Module 13: Performance Optimization [Complexity: M]
**Dependencies:** All other modules  
**Prerequisites:** Basic implementations of other modules complete  

### Phase 13a: Rendering Optimization [Complexity: M]
- [ ] Implement off-screen panel culling: don't render panels that are minimized or off-screen in `dependencies/BTQ_Render_Engine/src/rendering/panel_culler.cpp` [Complexity: M]
- [ ] Add off-screen chart item culling: don't render chart elements outside visible area, reduce polygon count at lower zoom levels in `dependencies/BTQ_Render_Engine/src/rendering/chart_culler.cpp` [Complexity: M]
- [ ] Implement footprint Level of Detail (LOD): reduce detail when zoomed out, show simplified representations, increase detail when zoomed in in `dependencies/BTQ_Render_Engine/src/rendering/footprint_lod.cpp` [Complexity: M]
- [ ] Batch order book geometry: combine multiple similar draw calls into single batched calls, reduce GPU overhead in `dependencies/BTQ_Render_Engine/src/rendering/orderbook_batcher.cpp` [Complexity: M]

### Phase 13b: Threading & Memory [Complexity: M]
- [ ] Multi-thread calculations: run volume calculations, indicator computations, data processing on background threads in `dependencies/BTQ_Render_Engine/src/threading/task_scheduler.cpp` [Complexity: M]
- [ ] Use lock-free queues: implement thread-safe data structures for passing data between calculation threads and UI thread in `dependencies/BTQ_Render_Engine/src/threading/lockfree_queue.cpp` [Complexity: M]
- [ ] Add memory pools: pre-allocate large blocks of memory for frequently allocated objects (trades, clusters, indicators) to reduce allocation overhead in `dependencies/BTQ_Render_Engine/src/memory/memory_pool.cpp` [Complexity: M]
- [ ] Optimize ImGui usage: reduce redundant calls, cache computed values, minimize state changes in `dependencies/BTQ_Render_Engine/src/rendering/imgui_optimizer.cpp` [Complexity: M]

### Phase 13c: Performance Monitoring [Complexity: S]
- [ ] Add frame pacing: maintain consistent frame rate, smooth out rendering spikes, prevent dropped frames in `dependencies/BTQ_Render_Engine/src/rendering/frame_pacer.cpp` [Complexity: M]
- [ ] Implement auto-quality reduction: detect performance drops, automatically reduce visual quality to maintain responsiveness in `dependencies/BTQ_Render_Engine/src/rendering/auto_quality.cpp` [Complexity: M]
- [ ] Track panel render times: measure how long each panel takes to render, identify bottlenecks in `dependencies/BTQ_Render_Engine/src/performance/panel_profiler.cpp` [Complexity: S]
- [ ] Show frame time graph: visualize frame times in real-time, help identify performance issues in `dependencies/BTQ_Render_Engine/src/performance/frame_time_graph.cpp` [Complexity: M]
- [ ] Add CPU profiler: detailed breakdown of where CPU time is spent, function-level profiling in `dependencies/BTQ_Render_Engine/src/performance/cpu_profiler.cpp` [Complexity: M]
- [ ] Track memory usage: monitor RAM consumption, identify leaks, show usage trends over time in `dependencies/BTQ_Render_Engine/src/performance/memory_tracker.cpp` [Complexity: S]
- [ ] Add debug overlay: show performance metrics, frame rate, memory usage, active features in `dependencies/BTQ_Render_Engine/src/performance/debug_overlay.cpp` [Complexity: S]
- [ ] Track pipeline metrics: measure throughput of data pipeline, identify bottlenecks in `dependencies/BTQ_Render_Engine/src/performance/pipeline_tracker.cpp` [Complexity: S]
- [ ] Detect performance regression: automated tests that measure performance, alert when regressions introduced in `dependencies/BTQ_Render_Engine/src/performance/regression_detector.cpp` [Complexity: M]

---

## Module 14: UI/UX Enhancement [Complexity: S]
**Dependencies:** All other modules  
**Prerequisites:** Core functionality of other modules complete  

### Phase 14a: Visual Polish [Complexity: S]
- [ ] Apply dark theme: consistent dark color scheme across all panels, proper contrast ratios in `dependencies/BTQ_Render_Engine/src/ui/theme_manager.cpp` [Complexity: S]
- [ ] Load monospace font: use consistent monospace font for all numerical displays, improve readability in `dependencies/BTQ_Render_Engine/src/ui/font_manager.cpp` [Complexity: S]
- [ ] Add smooth animations: transition effects for panel opening/closing, smooth value changes, animated highlights in `dependencies/BTQ_Render_Engine/src/ui/animations.cpp` [Complexity: S]
- [ ] Add haptic effects: subtle feedback for important interactions, customizable intensity in `dependencies/BTQ_Render_Engine/src/ui/haptic_feedback.cpp` [Complexity: S]

### Phase 14b: User Experience [Complexity: S]
- [ ] Show loading states: progress indicators when data loading, prevent UI freezing during operations in `dependencies/BTQ_Render_Engine/src/ui/loading_states.cpp` [Complexity: S]
- [ ] Show empty states: helpful messages when no data available, suggestions for next steps in `dependencies/BTQ_Render_Engine/src/ui/empty_states.cpp` [Complexity: S]
- [ ] Add tooltips everywhere: informative tooltips for all controls, explain functionality in `dependencies/BTQ_Render_Engine/src/ui/tooltips.cpp` [Complexity: S]
- [ ] Add context menus: right-click menus with relevant actions for each element in `dependencies/BTQ_Render_Engine/src/ui/context_menus.cpp` [Complexity: S]
- [ ] Add quick actions toolbar: commonly used actions accessible from floating toolbar in `dependencies/BTQ_Render_Engine/src/ui/quick_actions.cpp` [Complexity: S]
- [ ] Add first-run tutorial: guided tour for new users, explain key features step-by-step in `dependencies/BTQ_Render_Engine/src/ui/tutorial.cpp` [Complexity: M]

---

## Module 15: Testing & Documentation [Complexity: M]
**Dependencies:** All other modules  
**Prerequisites:** Core implementations complete  

### Phase 15a: Testing [Complexity: M]
- [ ] Test volume calculations: validate all 16 volume types produce correct values with known datasets in `dependencies/BTQ_Render_Engine/tests/volume_calculation_tests.cpp` [Complexity: M]
- [ ] Test cluster engine: verify cluster aggregation works correctly, handles edge cases, maintains accuracy in `dependencies/BTQ_Render_Engine/tests/cluster_engine_tests.cpp` [Complexity: M]
- [ ] Test VWAP math: validate VWAP calculations against external sources, test anchored VWAP, rolling VWAP in `dependencies/BTQ_Render_Engine/tests/vwap_tests.cpp` [Complexity: M]
- [ ] Test data pipeline: verify data flows correctly from input to display, handles missing data, duplicates in `dependencies/BTQ_Render_Engine/tests/data_pipeline_tests.cpp` [Complexity: M]
- [ ] Benchmark performance: measure frame rates, memory usage, CPU consumption under various loads in `dependencies/BTQ_Render_Engine/tests/performance_benchmarks.cpp` [Complexity: M]
- [ ] Run stress tests: simulate high-volume trading conditions, verify stability over extended periods in `dependencies/BTQ_Render_Engine/tests/stress_tests.cpp` [Complexity: M]
- [ ] Test visual regression: automated screenshots comparing current output to reference images in `dependencies/BTQ_Render_Engine/tests/visual_regression_tests.cpp` [Complexity: M]
- [ ] Generate mock data: create realistic test datasets for all data types in `dependencies/BTQ_Render_Engine/tests/mock_data_generator.cpp` [Complexity: M]
- [ ] Run automated tests: continuous integration setup, test on multiple platforms in `dependencies/BTQ_Render_Engine/tests/automated_test_runner.cpp` [Complexity: M]

### Phase 15b: Documentation [Complexity: M]
- [ ] Write user guide: comprehensive manual for end users, explain all features in detail in `docs/user_guide.md` [Complexity: L]
- [ ] Generate API docs: automatically generated documentation from code comments in `docs/api_reference/` [Complexity: S]
- [ ] Write developer guide: setup instructions, architecture overview, contribution guidelines in `docs/developer_guide.md` [Complexity: M]
- [ ] Document volume types: detailed explanation of all 16 volume analysis types, use cases, interpretation in `docs/volume_types_documentation.md` [Complexity: M]
- [ ] Write performance guide: optimization tips, hardware requirements, troubleshooting performance issues in `docs/performance_guide.md` [Complexity: M]
- [ ] Write troubleshooting guide: common issues, solutions, debugging techniques in `docs/troubleshooting_guide.md` [Complexity: M]
- [ ] Add code examples: sample implementations of common tasks, reusable code snippets in `examples/` [Complexity: M]
- [ ] Create video tutorials: screen recordings demonstrating key features and workflows in `docs/videos/` [Complexity: L]

---

## Module 16: Code Quality & Release [Complexity: S]
**Dependencies:** All other modules  
**Prerequisites:** All functionality complete  

### Phase 16a: Code Quality [Complexity: S]
- [ ] Remove dead code: eliminate unused functions, variables, files in `dependencies/BTQ_Render_Engine/src/` [Complexity: S]
- [ ] Standardize naming: ensure consistent naming conventions across entire codebase in `dependencies/BTQ_Render_Engine/src/` [Complexity: S]
- [ ] Add logging system: structured logging throughout application, configurable log levels in `dependencies/BTQ_Render_Engine/src/logging/logger.cpp` [Complexity: M]
- [ ] Add error handling: comprehensive error checking, graceful degradation, user-friendly error messages in `dependencies/BTQ_Render_Engine/src/error_handling/` [Complexity: M]
- [ ] Validate all inputs: sanitize user inputs, validate data from external sources, prevent crashes from bad data in `dependencies/BTQ_Render_Engine/src/validation/input_validator.cpp` [Complexity: M]

### Phase 16b: Release Preparation [Complexity: S]
- [ ] Create release build: optimized compilation, strip debug symbols, minimize binary size in `build_scripts/release_build.sh` [Complexity: S]
- [ ] Add crash reporting: automatic crash dumps, error telemetry, user opt-out in `dependencies/BTQ_Render_Engine/src/error_handling/crash_reporter.cpp` [Complexity: M]
- [ ] Add telemetry: anonymous usage statistics, performance metrics, feature adoption tracking in `dependencies/BTQ_Render_Engine/src/telemetry/telemetry_collector.cpp` [Complexity: M]
- [ ] Run final QA: comprehensive testing of all features, edge cases, integration points in `dependencies/BTQ_Render_Engine/tests/final_qa_checklist.md` [Complexity: M]
- [ ] Create installer: platform-specific installation packages, dependency management, first-run setup in `installers/` [Complexity: M]