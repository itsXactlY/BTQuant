# TASK_ARCHITECT.md: The Zero-Lock Protocol

**Objective:** A flawless, clean, cutting edge, high-frequency, lock-free rendering engine.
**Core Principle:** "Data Flows, UI Polls." No blocking queues. No mutexes in the hot path.
**Tech Stack:** C++26 (std::atomic, std::span, std::from_chars), Vulkan, Shared Memory (Ring Buffer).

---

## Module 0: Critical Repairs (The "Stop the Bleeding" Phase)
**Dependencies:** None. Must be done first to fix the build.

### 0.1: HotSpine Type Alignment
- [ ] **Sync Data Structures:**
    - In `include/hotspine_layout_v3.hpp`, define the `HotspineData` struct as strictly POD (Plain Old Data), `alignas(64)`.
    - **Remove:** Any `std::vector`, `std::string`, or complex types from shared memory structs.
    - **Fix:** Update `src/hotspine/hotspine_reader.cpp` to use `HotSpine::V3::Viewport` (or `HotspineData`) instead of the non-existent `ClusterColumn`.

---

## Module 1: The Data Spine (Shared Memory Architecture)
**Dependencies:** Module 0. Defines the rigid contract between Collector and Terminal.

### 1.1: Lock-Free Ring Buffer Layout
- [ ] **Define `RingBufferHeader` (`include/hotspine_layout_v3.hpp`):**
    - Implement struct with `std::atomic<uint64_t> write_head` and `std::atomic<uint64_t> read_tail` (aligned to 64 bytes).
    - Hardcode `RING_BUFFER_SIZE = 1048576` (2^20) for bitwise masking operations.
- [ ] **Implement Inline Accessors:**
    - Add `load_acquire()` and `store_release()` helpers directly in the header to enforce memory ordering.

### 1.2: Shared Memory Manager (`include/hotspine_data_bridge.hpp`)**
- [ ] **Refactor Bridge:** Remove all `std::queue` and `std::mutex`.
- [ ] **Pointer Arithmetic:** Map the SHM segment to a raw `HotspineData*` pointer. Indexing becomes `base_ptr[index & mask]`.

---

## Module 2: The Producer (Market Data Collector)
**Dependencies:** Module 1. Converts Collector into a pure, dumb, fast feed handler.

### 2.1: Stripping the Fat
- [ ] **Remove Analytics:** In `market_data_collector/src/data/market_data_processor.cpp`, delete `ClusterEngine`, `Database`, and `snapshot_to_viewport` logic. The Collector must *never* calculate clusters.
- [ ] **Zero-Copy Ingestion:**
    - In `exchange_aggregator.cpp`, modify callbacks to construct `HotspineData` on the stack.
    - Call `HotSpineDataBridge::write_direct()` immediately.
    - **Constraint:** Zero heap allocations in the callback path.

### 2.2: Producer Flow Control
- [ ] **Yield-on-Full:** Implement logic: `if (write_head - read_tail > SIZE) std::this_thread::yield();`.
- [ ] **Affinity:** Ensure the Collector thread is pinned to a specific CPU core (e.g., Core 2) to avoid context switches.

---

## Module 3: The Consumer (Terminal Data Core)
**Dependencies:** Module 1. The "Heart" of the Render Engine.

### 3.1: Atomic Storage Infrastructure (`include/market_data_processor.hpp`)
- [ ] **Create Atomic Registry:**
    - Replace `std::map` with `std::vector<AtomicSymbolInfo> atomic_storage_` (pre-allocated to 100,000 slots).
    - `struct AtomicSymbolInfo { std::atomic<double> price; std::atomic<double> volume; ... };`
- [ ] **Zero-Lock Access:** Implement `get_atomic_snapshot(uint32_t id)` that returns a const pointer to the atomic struct.

### 3.2: The Polling Loop (`src/data/market_data_processor.cpp`)
- [ ] **Implement `poll_hotspine()`:**
    - A tight loop that reads from SHM `write_head`.
    - Updates `atomic_storage_` using `std::memory_order_relaxed`.
    - **Batching:** Process max 50,000 events per frame to ensure UI responsiveness.
- [ ] **Cluster Engine Integration:**
    - Feed raw trades into `ClusterEngine` *after* updating the atomic price.
    - **Optimization:** Use a thread-local buffer for `ClusterEngine` updates to avoid locking the main atomic storage.

---

## Module 4: UI Refactor (The Pull Model)
**Dependencies:** Module 3. Converting all Panels to "Poll" instead of "Push".

### 4.1: Watchlist & Dashboard ("Lobotomy")
- [ ] **Remove Queues:** Delete `pending_updates_` queue and `on_market_data_update` callback from `WatchlistPanel`.
- [ ] **Polling Render:** In `WatchlistPanel::render()`, iterate visible rows and read directly from `processor_->get_atomic_snapshot(id)`.
- [ ] **Result:** Rendering cost becomes `O(Visible_Rows)`, decoupling FPS from Market Rate.

### 4.2: Orderbook Double-Buffering
- [ ] **Implement Snapshotting:**
    - `MarketDataProcessor` maintains the live L2 book.
    - Once per frame (start of frame), `MarketDataProcessor` atomically swaps a "Reader Snapshot" pointer.
- [ ] **Render:** `OrderbookPanel` reads strictly from the "Reader Snapshot". No mutexes required during render.

---

## Module 5: Advanced Analytics (Restoring Quantower Features)
**Dependencies:** Module 3 & 4. Re-enabling features on the new architecture.

### 5.1: Footprint & TPO
- [ ] **Direct Memory Access:** Modify `FootprintPanel` to read `ClusterEngine` data structures directly via `const` pointers.
- [ ] **LOD System:** Implement Level-of-Detail rendering (Module 13a) that skips text/details when zooming out, reading directly from the cluster vectors.

### 5.2: Technical Indicators
- [ ] **Calculation Thread:** Move VWAP/SMA/EMA calculations to a background thread (`TaskScheduler`).
- [ ] **Atomic Result:** The background thread writes results to `AtomicIndicatorValues` in `MarketDataProcessor`. The Chart Panel simply polls these values.

---

## Module 6: Cleanup & Final Polish
**Dependencies:** All previous modules.

### 6.1: The Purge
- [ ] **Delete Mutexes:** Grep for `std::mutex` in `src/data/` and `src/components/`. Delete them. Redesign flow if this causes race conditions (use ownership transfer or atomics).
- [ ] **Delete CVs:** Remove `std::condition_variable` usage in the hot path.

### 6.2: Compiler Optimization
- [ ] **CMake Update:** Ensure flags `-std=c++26`, `-O3`, `-march=native`, `-flto` are set in `CMakeLists.txt`.
- [ ] **Verify:** Check that `Release` build strips symbols and optimizes loops.