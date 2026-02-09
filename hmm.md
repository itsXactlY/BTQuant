
# BTQ Render Engine - Complete Quantower Clone (Parallel Development Version)

**Tech Stack:** C++23/26, Vulkan, ImGui, existing HotspineDataBridge  
**Reference:** https://help.quantower.com/quantower/  

---

## Development Guidelines
- Each module can be developed in parallel by separate developers
- Dependencies between modules are clearly marked
- Estimated complexity: S (Small, 1-2 days), M (Medium, 3-5 days), L (Large, 1+ weeks)
- Prerequisites must be completed before starting dependent tasks
- Your workdir is exlusivly dependencies/BTQ_Render_Engine/
- Never do or build TEST files - work only on livecode
- Fully autonomous handle conflicts in the most harmonic way with C++23/26 only.
---

# Phase 40: Zero-Lock Modernization (C++26)
**Objective:** Eliminate all `std::mutex`, `std::shared_mutex`, and `std::condition_variable` usage in the hot path. Migrate to C++26 Hazard Pointers and Lock-Free Queues.

## 40.1: Immediate Startup Fixes [Critical]
- [x] **Fix Vulkan Font Upload:**
  - In `src/system/VulkanCore.cpp` -> `init_imgui()`:
  - Create a single-time command buffer (`begin_single_time_commands()`).
  - Call `ImGui_ImplVulkan_CreateFontsTexture()`.
  - Submit and destroy font upload objects (`ImGui_ImplVulkan_DestroyFontUploadObjects()`).
- [x] **Verify Shader Loading:** Check `BTQ_Render_Engine/shaders/spirv/` paths in `main_trading_terminal.cpp` relative to the execution directory.

## 40.2: Lock-Free Task Scheduler (The Engine)
**Goal:** Replace the mutex-heavy scheduler with a high-throughput, wait-free implementation.
- [x] **Replace Queue:** Remove `std::queue` and `std::mutex queue_mutex_`.
  - Integrate `moodycamel::ConcurrentQueue<Task>` (already in dependencies) or build a custom Ring Buffer using `std::atomic<size_t>` head/tail.
- [x] **Atomic Signaling (C++20/26):**
  - Remove `std::condition_variable`.
  - Use `std::atomic<uint32_t>::notify_one()` and `std::atomic<uint32_t>::wait()` for worker sleep/wake cycles.
  - **Why:** Removes kernel-level locking overhead during task dispatch.
- [x] **Stop Token:** Replace `stop_mutex_` with `std::atomic_flag` or `std::stop_source` (C++20).

## 40.3: Hazard Pointer Memory Reclamation (The Infinite Canvas)
**Goal:** Allow the "Infinite Canvas" (ClusterEngine) to grow and prune without stopping readers (Renderers).
- [x] **Implement Hazard Pointers (`<hazard_pointer>` C++26):**
  - **Readers (Renderers):** When rendering a viewport, acquire a hazard pointer to the `ClusterChunk` being drawn. This guarantees the data remains valid even if the processor tries to delete it.
  - **Writer (MarketDataProcessor):** When pruning old data (>4 hours), call `retire()` on the chunk.
  - **Reclamation:** The system automatically frees the memory *only* when no hazard pointers reference it. No locks required.
- [x] **RCU for Configuration (`<rcu>` C++26):**
  - Use `std::rcu_obj_base` for `active_pairs_` and configuration maps.
  - Readers access data via `std::rcu_read_lock`.
  - Updates happen via `synchronize_rcu()`, ensuring zero contention for readers.

## 40.4: Atomic Data Ingestion (The Pipeline)
**Goal:** Ingest 1M+ trades/sec without locking the UI.
- [x] **Double-Buffered State:**
  - In `MarketDataProcessor`, replace `std::vector` buffers with a **Swap-Buffer** architecture using `std::atomic<State*>`.
  - **Writer:** Fills the "Back" buffer. When full, atomically swaps the pointer to make it the "Front" buffer.
  - **Reader:** Grabs the "Front" buffer pointer atomically to process/render.
- [x] **Parallel Processing (`<execution>`):**
  - In `ClusterEngine::process_trade_batch`, use `std::for_each(std::execution::par_unseq, ...)` to vectorize volume calculations across the batch before merging.

## 40.5: Validation
- [x] **Benchmark:** Run `MarketDataProcessor` with 1M messages/sec replay.
  - **Expectation:** CPU usage should be high (processing) but uniform across cores. No "spikes" or "stalls".
- [x] **Leak Check:** Verify Hazard Pointers correctly reclaim memory after the 4-hour window moves.
- [x] Run `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build` and confirm `BTQuantTerminal` links successfully without the previous type / namespace errors.

# Phase 40: Zero-Lock Modernization (C++26)
**Objective:** Eliminate all `std::mutex`, `std::shared_mutex`, and `std::condition_variable` usage in the hot path. Migrate to C++26 Hazard Pointers and Lock-Free Queues.

## 40.1: Immediate Startup Fixes [Critical]
- [x] **Fix Vulkan Font Upload:**
  - In `src/system/VulkanCore.cpp` -> `init_imgui()`:
  - Create a single-time command buffer (`begin_single_time_commands()`).
  - Call `ImGui_ImplVulkan_CreateFontsTexture()`.
  - Submit and destroy font upload objects (`ImGui_ImplVulkan_DestroyFontUploadObjects()`).
- [x] **Verify Shader Loading:** Check `BTQ_Render_Engine/shaders/spirv/` paths in `main_trading_terminal.cpp` relative to the execution directory.

## 40.2: Lock-Free Task Scheduler (The Engine)
**Goal:** Replace the mutex-heavy scheduler with a high-throughput, wait-free implementation.
- [x] **Replace Queue:** Remove `std::queue` and `std::mutex queue_mutex_`.
  - Integrate `moodycamel::ConcurrentQueue<Task>` (already in your dependencies) or build a custom Ring Buffer using `std::atomic<size_t>` head/tail.
- [x] **Atomic Signaling (C++20/26):**
  - Remove `std::condition_variable`.
  - Use `std::atomic<uint32_t>::notify_one()` and `std::atomic<uint32_t>::wait()` for worker sleep/wake cycles.
  - **Why:** Removes kernel-level locking overhead during task dispatch.
- [x] **Stop Token:** Replace `stop_mutex_` with `std::atomic_flag` or `std::stop_source` (C++20).

## 40.3: Hazard Pointer Memory Reclamation (The Infinite Canvas)
**Goal:** Allow the "Infinite Canvas" (ClusterEngine) to grow and prune without stopping readers (Renderers).
- [x] **Implement Hazard Pointers (`<hazard_pointer>` C++26):**
  - **Readers (Renderers):** When rendering a viewport, acquire a hazard pointer to the `ClusterChunk` being drawn. This guarantees the data remains valid even if the processor tries to delete it.
  - **Writer (MarketDataProcessor):** When pruning old data (>4 hours), call `retire()` on the chunk.
  - **Reclamation:** The system automatically frees the memory *only* when no hazard pointers reference it. No locks required.
- [x] **RCU for Configuration (`<rcu>` C++26):**
  - Use `std::rcu_obj_base` for `active_pairs_` and configuration maps.
  - Readers access data via `std::rcu_read_lock`.
  - Updates happen via `synchronize_rcu()`, ensuring zero contention for readers.

## 40.4: Atomic Data Ingestion (The Pipeline)
**Goal:** Ingest 1M+ trades/sec without locking the UI.
- [x] **Double-Buffered State:**
  - In `MarketDataProcessor`, replace `std::vector` buffers with a **Swap-Buffer** architecture using `std::atomic<State*>`.
  - **Writer:** Fills the "Back" buffer. When full, atomically swaps the pointer to make it the "Front" buffer.
  - **Reader:** Grabs the "Front" buffer pointer atomically to process/render.
- [x] **Parallel Processing (`<execution>`):**
  - In `ClusterEngine::process_trade_batch`, use `std::for_each(std::execution::par_unseq, ...)` to vectorize volume calculations across the batch before merging.

###  40.5: Validation
- [x] **Benchmark:** Run `MarketDataProcessor` with 1M messages/sec replay.
  - **Expectation:** CPU usage should be high (processing) but uniform across cores. No "spikes" or "stalls".
- [x] **Leak Check:** Verify Hazard Pointers correctly reclaim memory after the 4-hour window moves.
- [x] Run `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build` and confirm `BTQuantTerminal` links successfully without the previous type / namespace errors.


# Phase 42: The "Sapir" Lock-Free HotSpine Pipeline
**Objective:** Eliminate IPC latency and UI locking by converting the data pipeline from a Push/Queue model to a Shared Memory Ring Buffer + Atomic Polling model.
**Philosophy:** "The UI never waits. The Data never stops."

## 42.1: Shared Memory Core Infrastructure
**Context:** Leveraging existing `HotspineData` and Layout files to enforce cache-aligned, lock-free structures.

- [x] **42.1.1: Canonical Event Definition (`include/trading/HotspineData.h`)**
    - Refactor `HotspineData` struct to be `standard_layout` and trivially copyable.
    - **Fields:** Ensure explicit padding to 64 bytes (cache line size).
    - **Flags:** Add `uint8_t flags` field (Bit 0: `IS_WARMUP`, Bit 1: `IS_SNAPSHOT`).
    - **Alignment:** Add `alignas(64)` to the struct definition.

- [x] **42.1.2: Ring Buffer Layout (`include/hotspine_layout_v3.hpp`)**
    - Modify `HotSpineLayoutV3` struct to implement a raw ring buffer header.
    - **Indices:** Add `alignas(64) std::atomic<uint64_t> write_head;` and `alignas(64) std::atomic<uint64_t> read_tail;`.
    - **Buffer:** Define the data area as a flexible array member or fixed offset calculation, not a `std::vector`.
    - **Logic:** Add inline helper methods `get_next_write_slot()` and `commit_write()` directly in the header using `std::memory_order_release`.

- [x] **42.1.3: Shared Memory Manager (`include/hotspine_data_bridge.hpp`)**
    - Refactor `HotSpineDataBridge` to manage the raw memory mapping.
    - **Consumer Mode:** Ensure `mmap` is read-only or read-write (depending on consumer feedback needs) but strictly non-blocking.
    - **Validation:** Add a startup check in `connect()` to verify the "Magic Number" and Version in `HotSpineLayoutV3` match exactly.

## 42.2: Producer Integration (Market Data Collector)
**Context:** Modifying the ingestion path in `market_data_collector/` to write directly to SHM without intermediate queues.

- [x] **42.2.1: Bypass Internal Queues (`market_data_collector/src/data/exchange_aggregator.cpp`)**
    - Identify `on_trade_update` and `on_depth_update` callbacks.
    - **Action:** Instead of pushing to an internal `std::queue` or `concurrentqueue`, call `HotSpineDataBridge::write_direct()`.
    - **Constraint:** Zero allocation in the callback. Use stack-allocated `HotspineData` structs only.

- [x] **42.2.2: Ring Buffer Writer (`market_data_collector/src/data/hotspine_data_bridge.cpp`)**
    - Implement `write_direct(const HotspineData& event)`.
    - **Logic:**
        1. Load `write_head` (relaxed).
        2. Calculate slot index: `idx = write_head & (RING_SIZE - 1)`.
        3. `memcpy` event to slot.
        4. Atomic store `write_head` (release).
    - **Overflow:** If `write_head - read_tail > RING_SIZE`, increment a `dropped_count` atomic (diagnostic only) and overwrite (circular) or yield. *Decision: Overwrite for HFT.*

- [x] **42.2.3: Warm-Up Generator (`market_data_collector/src/main.cpp`)**
    - Create a simple timer loop in the main thread (or dedicated thread).
    - **Action:** Every 100ms, inject a `HotspineData` event with `flags |= IS_WARMUP`.
    - **Purpose:** Keep the CPU cache lines of the ring buffer and processor hot during low-volume periods.

## 42.3: Consumer Path (Render Engine / MarketDataProcessor)
**Context:** Converting `MarketDataProcessor` from a queue consumer to a ring buffer poller.

- [x] **42.3.1: Atomic Storage (`include/market_data_processor.hpp`)**
    - Replace `std::unordered_map<uint32_t, SymbolData>` with `std::vector<AtomicSymbolInfo>`.
    - **Struct:** Define `AtomicSymbolInfo` inside the header:
      ```cpp
      struct alignas(64) AtomicSymbolInfo {
          std::atomic<double> price;
          std::atomic<double> volume;
          std::atomic<double> daily_change;
          std::atomic<uint64_t> last_ts;
      };
      ```
    - **Access:** Add `get_atomic_snapshot(uint32_t symbol_id)` returning a const pointer.

- [x] **42.3.2: Polling Ingestion Loop (`src/data/market_data_processor.cpp`)**
    - Refactor `process_events()` to poll the SHM ring buffer.
    - **Batching:** Process up to `MAX_BATCH_SIZE` (e.g., 4096) events per cycle.
    - **Logic:**
        1. Read `write_head` from SHM (acquire).
        2. Iterate from `local_read_tail` to `write_head`.
        3. For each event:
           - Check `IS_WARMUP`. If true, skip side effects, but touch memory.
           - If real, update `atomic_storage_[symbol_id]` fields using `std::memory_order_relaxed`.
    - **Zero Locks:** Ensure strictly NO `mutex` usage in this loop.

- [x] **42.3.3: Cluster & Analytics Bypass (`src/analytics/cluster_engine.cpp`)**
    - **Current State:** Likely waits for `MarketDataProcessor` to push data.
    - **New State:** The ingestion loop calls `ClusterEngine::ingest_batch(start_ptr, count)` directly.
    - **Threading:** If `ClusterEngine` runs on a separate thread, use a *Single-Producer Single-Consumer* (SPSC) pointer pass-through, NOT a copying queue.

## 42.4: UI Pull Model (Decoupled Rendering)
**Context:** Rewriting UI panels to pull data from `MarketDataProcessor`'s atomic storage.

- [x] **42.4.1: Watchlist De-Queueing (`src/components/watchlist_panel.cpp`)**
    - **Delete:** `pending_subscriptions_` queue and `on_market_data_update` callback.
    - **Refactor `render()`:**
        - Iterate strictly over *visible* rows (ImGui Clipper).
        - For each symbol, call `processor->get_atomic_snapshot(id)`.
        - Read atomic doubles/uints directly into local variables for `ImGui::Text`.
    - **Benefit:** Rendering cost becomes O(Visible Rows), not O(Market Activity).

- [x] **42.4.2: Orderbook Polling (`src/components/orderbook_panel.cpp`)**
    - Similar to Watchlist. Instead of processing delta updates in the UI thread:
    - **Logic:** `MarketDataProcessor` maintains the L2 Book in a lock-free/double-buffered structure.
    - **Render:** `OrderbookPanel` requests a "Reader Snapshot" pointer. If the pointer is valid, read and render. If invalid (writer swapping), skip update for one frame (imperceptible at 60FPS).

- [x] **42.4.3: Frame Budgeting (`src/vulkan_dashboard_advanced.cpp`)**
    - In `render_frame()`, verify that the `MarketDataProcessor::poll()` call happens *before* the UI render pass.
    - Ensure the polling duration is capped (e.g., 2ms) to prevent frame drops during market storms.

## 42.6: Static Data & Optimization
**Context:** Removing dynamic lookups from the hot path.

- [x] **42.6.1: Static Symbol Map (`include/symbol_registry.hpp`)**
    - Replace `std::unordered_map` with a sorted `std::vector<SymbolEntry>` + `std::binary_search` for ID lookups during configuration/startup.
    - **Hot Path:** Ensure `symbol_id` (integer) is used exclusively in the hot path. No string comparisons (`"BTC-USDT"`) during ingestion.

- [x] **42.6.2: Pre-Allocation Strategy (`src/system/memory_optimizer.cpp`)**
    - Ensure `AtomicSymbolInfo` vector in `MarketDataProcessor` is `reserve()`d to `MAX_SYMBOLS` at startup.
    - Verify `ClusterEngine` pools are pre-allocated.

## 42.7: Global Concurrency Audit (Search & Destroy)
- [x] **42.7.1: Mutex Audit**
    - Grep for `std::lock_guard` and `std::unique_lock` in `src/data/` and `src/components/`.
    - **Action:**
        - If in `WatchlistPanel`: Replace with Atomic Pull (Task 42.4.1).
        - If in `MarketDataProcessor`: Replace with Atomic Storage (Task 42.3.1).
        - If in `Logger`: Ensure it's using the new `recursive_mutex` fix, but verify logging is DISABLED in the hot loop.

- [x] **42.7.2: Condition Variable Purge**
    - Grep for `std::condition_variable`.
    - **Rule:** Allowed *only* in `TaskScheduler` (worker sleep) and `LoadingStateManager`.
    - **Ban:** Strictly prohibited in `HotSpineDataBridge` or `MarketDataProcessor`. Replace with busy-spin/yield logic for HFT components.

## 42.8: Validation & Verification
- [x] **42.8.1: Latency Test**
    - Instrument the pipeline to record timestamps: `T0 (Ingest)` -> `T1 (Atomic Write)` -> `T2 (UI Read)`.
    - **Goal:** T1 - T0 < 10 microseconds (High performance).

- [x] **42.8.2: Soak Test**
    - Run the terminal connected to the `market_data_collector` (Producer).
    - Simulate 1 million events/sec (using the Warmer or Replay).
    - **Pass Criteria:** UI remains interactive (mouse hover works, buttons click instantly).

- [x] **42.8.3: Memory Stability**
    - Enhanced bounds checking in `HotSpineDataBridge::write_direct()` and `MarketDataProcessor::pollSharedMemoryRingBuffer()` to prevent buffer overflows and out-of-bounds reads.
    - Added integer overflow protection in pointer arithmetic calculations.
    - Enabled AddressSanitizer (ASAN) compilation flags for memory stability testing.
    - Created enhanced memory stability test with memory usage monitoring.
- [x] Run `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build` and confirm `BTQuantTerminal` links successfully without the previous type / namespace errors.