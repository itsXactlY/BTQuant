
**Tech Stack:** C++23/26, existing HotspineDataBridge  

---

## Development Guidelines
- Each module can be developed in parallel by separate developers
- Dependencies between modules are clearly marked
- Estimated complexity: S (Small, 1-2 days), M (Medium, 3-5 days), L (Large, 1+ weeks)
- Prerequisites must be completed before starting dependent tasks
- Your workdir is exlusivly /home/alca/projects/PubBTQuant/dependencies/ccapi/example/src/market_data_collector/
- Fully autonomous handle conflicts in the most harmonic way with C++23/26 only.
---


# Phase 43: Producer-Side HotSpine Alignment (Market Data Collector)
**Objective:** convert `market_data_collector` into a pure, low-latency "Feed Handler" that does strictly one thing: normalize trades and push them to the Shared Memory Ring Buffer.

## 43.1: Shared Memory Layout Alignment [Critical]
- [x] **Import Layout Header:** Ensure `market_data_collector` uses the exact same `hotspine_layout_v3.hpp` as the Terminal.
- [x] **Fix Ring Buffer Size:**
  - **Current Bug:** `market_data_processor.cpp:330` uses `current_head % 1024`.
  - **Fix:** Use `current_head & (HotSpine::V3::RING_BUFFER_SIZE - 1)`.
  - **Requirement:** Ensure `RING_BUFFER_SIZE` is defined as `1048576` (2^20) in the shared header.

## 43.2: The "Raw Feed" Refactor (Performance)
**Context:** The Collector is currently running `engine_.process_trade()` and `snapshot_to_viewport()`. This is **WRONG**. The Collector should write *Raw Trades*, and the Terminal (Consumer) should calculate Clusters/Viewports.

- [x] **Remove Analytics from Collector:**
  - Delete `Analytics::ClusterEngine engine_;` from `MarketDataProcessor`.
  - Remove `engine_.process_trade(t)` and `engine_.snapshot_to_viewport(...)` calls.
- [x] **Implement Raw Write:**
  - Modify the SHM write block in `handleTradeMessage` to write a `HotspineData` struct (Raw Trade) instead of a `Viewport`.
  - **Fields:** `timestamp`, `price`, `volume`, `flags` (Buy/Sell/Liquidation).

## 43.3: Hot Path Optimization (C++26 Style)
- [x] **Remove Mutexes:**
  - In `handleTradeMessage`, remove `std::lock_guard<std::mutex> lock(buffer_mutex_);`.
  - **Logic:** If `enable_exclusive_hotspine_` is true, **SKIP** all `trade_buffer_.push_back` (DB buffering). The DB path is too slow for the HotSpine.
- [x] **Fast Parsing:**
  - Replace `safeParseDouble` (which uses `std::stod` and `try-catch`) with `std::from_chars` (no exceptions, zero allocation).
  - Use `std::string_view` for all parsing helpers to avoid `std::string` copies.

## 43.4: Producer-Side Flow Control
- [x] **Overflow Handling:**
  - Before writing, check: `write_head - read_tail > RING_BUFFER_SIZE`.
  - **Action:** If full, increment a `dropped_packet_count` atomic and **yield** (`std::this_thread::yield()`). Do not overwrite unread data blindly unless in "Turbo Mode".
- [x] **Affinity & Priority:**
  - Verify `threadAffinityCheck` pins to an Isolated Core (e.g., Core 3).
  - Set thread priority to `SCHED_FIFO` (Real-time) if running as root/admin.

## 43.5: Validation
- [x] **Verification Step:**
  - Start Collector.
  - Start Terminal.
  - **Expected:** Terminal should receive raw trades and build its own Heatmap/Clusters.
  - **Metric:** Latency (Collector Ingest -> Terminal Render) should be < 50 microseconds.

# Phase 44: Build System Repair & Type Alignment
**Objective:** Fix breaking changes in the CCAPI library headers and align the HotSpine Reader with the current Shared Memory Layout V3.

## 44.1: Fix CCAPI Library Internals (Critical)
**Context:** The compiler cannot find `UtilString::safeParseDouble`. This suggests a partial optimization was applied to `ccapi_util_private.h`.

- [x] **Define `safeParseDouble` in `UtilString`:**
  - Open `dependencies/ccapi/include/ccapi_cpp/ccapi_util_private.h`.
  - Locate the `class UtilString` definition.
  - Add the static method implementation (using `std::from_chars` for C++17/26 compliance and speed):
    ```cpp
    static double safeParseDouble(std::string_view sv) {
        double result = 0.0;
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);
        if (ec != std::errc()) return 0.0; // Fallback
        return result;
    }
    // Overload for std::string if needed by legacy code
    static double safeParseDouble(const std::string& s) {
        return safeParseDouble(std::string_view(s));
    }
    ```
  - **Verification:** This needs to be inside the `struct UtilString` or `class UtilString` scope to satisfy `UtilString::safeParseDouble`.

## 44.2: Fix HotSpine Reader Type Mismatch
**Context:** `hotspine_reader.cpp` is referencing `ClusterColumn`, which likely doesn't exist in your new V3 layout.

- [x] **Identify Correct Type:**
  - Check `include/hotspine_layout_v3.hpp`. Look for the main struct definition (likely `HotspineData` or `Viewport`).
- [x] **Update Header (`src/hotspine/hotspine_reader.hpp`):**
  - Replace `bool pollLatestViewport(HotSpine::V3::ClusterColumn& out_viewport);`
  - With: `bool pollLatestViewport(HotSpine::V3::Viewport& out_viewport);` (or whatever the actual struct name is in `hotspine_layout_v3.hpp`).
- [x] **Update Source (`src/hotspine/hotspine_reader.cpp`):**
  - Update the function signature to match the header.
  - Update any internal member access (e.g., if `ClusterColumn` had `price_levels` and `Viewport` has `clusters`, update the mapping).

## 44.3: Market Data Collector Cleanup
**Context:** Once 44.1 is fixed, `market_data_collector` might still have issues if it repeats the `safeParseDouble` mistake in its own `utilities.h`.

- [x] **Check `market_data_collector/utilities.h`:**
  - Ensure `safeParseDouble` is defined `inline` to avoid ODR (One Definition Rule) violations if included in multiple translation units.
- [x] **Sync Includes:**
  - Ensure `market_data_processor.cpp` includes the corrected `hotspine_layout_v3.hpp`.

## 44.4: Validation
- [x] **Rebuild CCAPI Targets:**
  - Run `ninja src/market_data_simple_request/market_data_simple_request` to verify the CCAPI fix.
- [x] **Rebuild HotSpine:**
  - Run `ninja src/hotspine/hotspine_reader` to verify the type fix.
- [x] **Full Build:**
  - Run `ninja` to ensure all targets link correctly.