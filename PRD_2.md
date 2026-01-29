# BTQ Render Engine - Refactoring & Optimization

**Context:** Resolve namespace conflict (BTQ vs BTQuant), decouple logic from rendering, optimize hot paths.  
**Target:** `dependencies/BTQ_Render_Engine/src/`

---

## Phase 1: Architecture Consolidation (Namespace Merge)

- [x] Audit legacy footprint logic: port DataType enum and ImGui::BeginCombo selector from `src/widgets/FootprintChart.cpp` to `src/components/footprint_panel.cpp`
- [x] Migrate heatmap coloring: merge getHeatmapColor method from `src/widgets/FootprintChart.cpp` into getCellColor in `src/components/footprint_panel.cpp` to support Delta gradient (green/red)
- [ ] Delete legacy footprint artifacts: remove `src/widgets/FootprintChart.cpp` and `include/widgets/FootprintChart.h` after migration verification to eliminate BTQ namespace

## Phase 2: Render-Loop Decoupling (Performance Critical)

- [ ] Extract indicator calculations from render loop: move calculate_sma, calculate_ema, calculate_rsi out of render() in `src/components/chart_panel.cpp` (lines 185-350) into cached map updated only on new data
- [ ] Optimize crosshair search: replace linear O(n) candle search with std::lower_bound binary search O(log n) in render_crosshair_info method (line 436) in `src/components/chart_panel.cpp`

## Phase 3: Orderbook Rendering Optimization

- [ ] Batch orderbook geometry drawing: use ImDrawList::ChannelsSplit to batch liquidity bar backgrounds before text in `src/components/orderbook_panel.cpp` render_orderbook_ladder method
- [ ] Flatten orderbook UI structure: pre-calculate heatmap geometry and draw as single primitive pass behind table instead of row-by-row cells in `src/components/orderbook_panel.cpp`

## Phase 4: Volume Profile Memory Management

- [ ] Implement incremental volume profile updates: stop clearing volume_profile_ vector on every tick, instead check if trade fits existing bucket and only rebuild if price outside range in `src/components/volume_profile_panel.cpp` build_volume_profile method

## Phase 5: System Observability

- [ ] Instrument main render loop: add performance_monitor.start_frame() and end_frame() calls in `src/main_realtime_dashboard.cpp` ImGui loop
- [x] Add widget-specific telemetry: insert scoped std::chrono timers in render() methods of `src/components/chart_panel.cpp` and `src/components/footprint_panel.cpp` to report individual panel render times to PerformanceMonitor

---

## Success Criteria

- BTQ namespace completely eliminated
- Indicator calculations no longer run in render loop
- Orderbook geometry batched (reduced draw calls by >80%)
- Volume profile uses incremental updates (no full rebuild per tick)
- Performance monitor shows per-widget render times