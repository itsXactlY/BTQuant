# TASK_SURGERY_FINAL.md: The Pure Quantower Clone (C++26)

**Objective:** Clean the Hydra. One Brain (PanelManager + QuantWorkspaceComponent). One Data Spine (Lock-free HotSpine). Zero Mutexes on the hot path.

---

## Phase 1: The Great Purge (Amputation)
**Goal:** Physically delete the conflicting brains, test files, and data source logic that do not belong in a Render Engine.

- [x] **1.1: Delete the UI Hydras (Legacy Dashboards & Layouts)**
    - Delete `src/components/realtime_dashboard_component.cpp` & `include/components/realtime_dashboard_component.hpp`.
    - Delete `src/layout/dashboard_layout_manager.cpp` & `include/layout/dashboard_layout_manager.hpp`.
    - *Reason:* `QuantWorkspaceComponent` and `ui/layout_manager.cpp` are the only UI/Layout brains allowed to live.

- [x] **1.2: Delete the Data Hydras (Exchange Connections in the Renderer)**
    - Delete `src/data/exchange_aggregator.cpp`, `.bak`, `.bak2` & `include/data/exchange_aggregator.hpp`.
    - Delete `src/data/unified_data_pipeline.cpp` & `include/data/unified_data_pipeline.hpp`.
    - *Reason:* The Render Engine does NOT talk to Binance/OKX. It ONLY talks to the HotSpine shared memory.

- [x] **1.3: Delete the Optimization Hydra**
    - Delete the entire `src/optimization/` folder (contains `performance_optimizer.cpp`).
    - *Reason:* Conflicting thread priority assignments. Keep only `src/system/system_optimizer.cpp`.

- [x] **1.4: Delete the Old Renderer Monolith**
    - Delete `src/components/MarketMicrostructureRenderer.cpp` & `include/components/MarketMicrostructureRenderer.h`.
    - *Reason:* Panels now draw themselves. A monolithic renderer causes dual-pass GPU overhead.

- [x] **1.5: Eradicate All Test Files from the Build Tree**
    - Delete `src/test_candlestick.cpp`.
    - Delete `src/data/test_quality_monitor.cpp`.
    - Delete `src/ui/test_tooltips.cpp`.
    - Delete `src/ui/test_quick_actions.cpp`.
    - Delete `src/error_handling/test_*.cpp` and `main_test.cpp`.
    - Delete `src/logging/test_structured_logging.cpp`.
    - Delete `src/memory/test_*.cpp`.
    - *Reason:* Production C++26 code does not ship with scattered test translation units.

- [x] **1.6: Sanitize CMake**
    - Open `CMakeLists.txt`. Remove EVERY file you just deleted from the source lists.

---

## Phase 2: The Core Transplant (Single Brain Entry)
**Goal:** Wire `main_trading_terminal.cpp` to use strictly the modern dashboard by default.

- [x] **2.1: Rewire `src/main_trading_terminal.cpp`**
    - **Remove:** Any `panel_mgr->add_panel(...)` loops or legacy initializations.
    - **Implement Strict Sequence:**
        1. Initialize `SystemOptimizer` & `Logger`.
        2. Initialize `MarketDataProcessor` (The Data Core).
        3. Initialize `PanelManager` (Pass `MarketDataProcessor`).
        4. Initialize `QuantWorkspaceComponent` (Pass `PanelManager`).
        5. Initialize `VulkanDashboard` (Pass `QuantWorkspaceComponent` and `PanelManager`).
    - **Set Default View:** Add `workspace->set_layout(BTQuant::PanelManager::LayoutPreset::MODERN_TRADING);` right before the main Vulkan loop starts.

- [x] **2.2: Cleanse `VulkanDashboard`**
    - Open `src/vulkan_dashboard_advanced.cpp` and `include/vulkan_dashboard_advanced.hpp`.
    - **Remove:** All references, includes, and pointers to `RealtimeDashboardComponent`.
    - **Verify Render Loop:** Inside `VulkanDashboard::render_frame()`, ensure ONLY `workspace_->render_gui()` and `panel_manager_->render_panels()` are executed.

---

## Phase 3: The C++26 Lock-Free Data Spine
**Goal:** Rework `MarketDataProcessor` to use Atomic Snapshots instead of mutex-locked queues.

- [x] **3.1: Define Atomic Storage (`include/market_data_processor.hpp`)**
    - **Add Struct:**
      ```cpp
      struct alignas(64) AtomicSymbolInfo {
          std::atomic<double> price{0.0};
          std::atomic<double> volume{0.0};
          std::atomic<double> change_pct{0.0};
          std::atomic<uint64_t> timestamp{0};
      };
      ```
    - **Add Field:** `std::vector<AtomicSymbolInfo> atomic_snapshots_;` (Resize this to `100000` in the constructor to avoid reallocation).
    - **Add Method:** `const AtomicSymbolInfo* get_atomic_snapshot(uint32_t symbol_id) const;`

- [x] **3.2: Implement High-Frequency Poller (`src/data/market_data_processor.cpp`)**
    - Remove the queue-processing logic (`pending_market_data_updates_`).
    - **Implement `poll_hotspine()`:** - A lock-free `while` loop that reads `hotspine_bridge_->read_tail` to `write_head`.
      - For each event, update `atomic_snapshots_[symbol_id]` fields using `std::memory_order_relaxed`.
      - Execute this poll function at the start of the `VulkanDashboard` frame render.

---

## Phase 4: UI Decoupling (The Pull Model)
**Goal:** Remove all mutexes from the UI panels so they render at maximum FPS regardless of data throughput.

- [x] **4.1: Watchlist Lobotomy (`include/components/watchlist_panel.hpp` & `src/components/watchlist_panel.cpp`)**
    - **Delete:** `std::recursive_mutex watchlist_mutex_`.
    - **Delete:** `moodycamel::ConcurrentQueue pending_subscriptions_`.
    - **Delete:** The `on_market_data_update` callback logic.
    - **Refactor `render()`:**
      - Iterate over the visible list using `ImGuiListClipper`.
      - For each `symbol_id`, retrieve `processor_->get_atomic_snapshot(symbol_id)`.
      - Render the `price.load(std::memory_order_relaxed)`.

- [x] **4.2: Orderbook Lock-Free Read (`src/components/orderbook_panel.cpp`)**
    - Ensure the Orderbook Panel does NOT update its own state incrementally via callbacks.
    - Modify it to request a read-only pointer to the `MarketDataProcessor`'s "Front Buffer" L2 Book snapshot every frame.

---

## Phase 5: Verification & Compilation
**Goal:** Prove the Hydra is dead.

- [x] **5.1: Clean & Build**
    - Run: `rm -rf build && ./build_integration.sh`
    - Verify 100% successful linking. No missing references to `exchange_aggregator` or `RealtimeDashboard`.
- [x] **5.2: Sanity Check**
    - Launch the terminal. It should immediately open into the `MODERN_TRADING` layout.
    - Verify FPS is locked to the monitor's refresh rate (e.g., 60/144Hz) with absolutely zero stutter, as the UI is no longer waiting on data locks.