# TASK_SURGERY.md: The Core Refactor

**Objective:** Eliminate the "Hydra" (multiple UI managers), remove all hot-path mutexes, and implement atomic polling.
**Tech Stack:** C++26, Vulkan, Lock-Free Ring Buffer.

---

## Phase 1: Decapitation (Kill the Zombies)
**Goal:** Remove conflicting UI managers. `PanelManager` becomes the sole owner of panels. `QuantWorkspaceComponent` becomes the sole owner of the layout.

- [x] **1.1: Delete Legacy Dashboard Component**
    - **Delete File:** `src/components/realtime_dashboard_component.cpp`
    - **Delete File:** `include/components/realtime_dashboard_component.hpp`
    - **Update CMake:** Open `CMakeLists.txt` and remove lines referencing `realtime_dashboard_component`.

- [x] **1.2: Cleanse `VulkanDashboard`**
    - **File:** `src/vulkan_dashboard_advanced.cpp`
    - **Action:** Delete `std::shared_ptr<RealtimeDashboardComponent> realtime_dashboard_;` from the class/header.
    - **Action:** In `init_components()`, delete `realtime_dashboard_ = std::make_shared...`.
    - **Action:** In `render_frame()`, delete `realtime_dashboard_->render()`.
    - **Verify:** Only `workspace_->render_gui()` and `panel_manager_->render_panels()` should remain in the render loop.

- [x] **1.3: Fix Main Entry Point**
    - **File:** `src/main_trading_terminal.cpp`
    - **Action:** Remove any `panel_mgr->add_panel(...)` calls inside `main()`.
    - **Action:** Ensure instantiation order: 
        1. `MarketDataProcessor`
        2. `PanelManager` (takes processor)
        3. `QuantWorkspaceComponent` (takes panel_manager)
        4. `VulkanDashboard`
    - **Bootstrap:** Add `workspace->set_layout(LayoutPreset::PRO_QUANT);` before entering the main loop.

---

## Phase 2: The Atomic Core (Data Storage)
**Goal:** Create a lock-free memory bank that the UI can read from instantly.

- [x] **2.1: Define Atomic Structures**
    - **File:** `include/market_data_processor.hpp`
    - **Add Struct:**
      ```cpp
      struct alignas(64) AtomicSymbolInfo {
          std::atomic<double> price{0.0};
          std::atomic<double> volume_24h{0.0};
          std::atomic<double> change_24h{0.0};
          std::atomic<double> high_24h{0.0};
          std::atomic<double> low_24h{0.0};
          std::atomic<uint64_t> last_update_ts{0};
      };
      ```
    - **Add Storage:** `std::vector<AtomicSymbolInfo> atomic_store_;` (Resize to `MAX_SYMBOLS` in constructor).
    - **Add Accessor:** `const AtomicSymbolInfo* get_atomic_snapshot(uint32_t symbol_id) const;`

- [x] **2.2: Implement Polling Loop**
    - **File:** `src/data/market_data_processor.cpp`
    - **Create Function:** `void poll_hotspine_updates();`
    - **Logic:**
        1. Read `write_head` from `hotspine_bridge_` (`std::memory_order_acquire`).
        2. Loop from `local_read_tail` to `write_head`.
        3. For each event:
           - Calculate `symbol_id`.
           - Update fields in `atomic_store_[symbol_id]` using `std::memory_order_relaxed`.
        4. Update `local_read_tail`.
    - **Constraint:** NO MUTEXES. NO ALLOCATIONS.

---

## Phase 3: The UI Lobotomy (Watchlist)
**Goal:** Stop the Watchlist from processing data. It should only *display* data.

- [x] **3.1: Strip Logic from Header**
    - **File:** `include/components/watchlist_panel.hpp`
    - **Remove:** `std::mutex watchlist_mutex_`.
    - **Remove:** `std::queue pending_updates_` (and `ConcurrentQueue`).
    - **Remove:** `on_market_data_update(...)` callback declaration.

- [x] **3.2: Refactor Render Implementation**
    - **File:** `src/components/watchlist_panel.cpp`
    - **Action:** Delete `update(float dt)` body (or leave empty).
    - **Action:** Rewrite `render()` to use `ImGuiListClipper`:
      ```cpp
      ImGuiListClipper clipper;
      clipper.Begin(watchlist_.size());
      while(clipper.Step()) {
          for(int i=clipper.DisplayStart; i<clipper.DisplayEnd; i++) {
              uint32_t sym_id = get_symbol_at(i);
              auto* snapshot = processor_->get_atomic_snapshot(sym_id);
              // Render snapshot->price.load(std::memory_order_relaxed);
          }
      }
      ```

---

## Phase 4: Producer Alignment (Collector)
**Goal:** Ensure the data source writes to the ring buffer correctly without blocking.

- [x] **4.1: Fix Shared Layout**
    - **File:** `include/hotspine_layout_v3.hpp`
    - **Define:** `constexpr size_t RING_BUFFER_SIZE = 1048576;` (Power of 2).
    - **Struct:** Ensure `RingBufferHeader` uses `std::atomic<uint64_t>` for `write_head` and `read_tail`.

- [x] **4.2: Dumb Writer Implementation**
    - **File:** `market_data_collector/src/data/market_data_processor.cpp`
    - **Modify:** `handleTradeMessage`.
    - **Logic:**
        1. Parse JSON to `HotspineData` struct (Stack allocation only).
        2. Call `shm_bridge_->write_direct(data)`.
    - **Remove:** All `trade_buffer_.push_back` calls. All `ClusterEngine` usage in the collector.

---

## Phase 5: Build & Verification
**Goal:** Compile a clean release build.

- [x] **5.1: Clean Build**
    - **Command:** `rm -rf build && ./build_integration.sh`
    - **Check:** Verify no linker errors regarding `RealtimeDashboardComponent`.

- [x] **5.2: Sanity Check**
    - Run the terminal.
    - Confirm the UI is responsive (buttons click instantly) even when connected to the data feed.