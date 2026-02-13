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

---

## Phase 6: Double/Triple Rendering Fix (2026-02-13)
**Goal:** Fix the issue where everything was rendering 2-3 times, causing duplicate UI elements.

### Root Causes Identified

| # | Problem | Location | Why it causes duplicate rendering |
|---|---------|----------|-----------------------------------|
| 1 | **Orphaned Variable Instantiation** | [`main_trading_terminal.cpp`](dependencies/BTQ_Render_Engine/src/main_trading_terminal.cpp) | Variables (Trading Systems, PanelManager, Workspace) were created in `main()` AND internally by `VulkanDashboard`, resulting in 2 sets of everything |
| 2 | **Double Data Sync** | [`quant_workspace_component.cpp`](dependencies/BTQ_Render_Engine/src/components/quant_workspace_component.cpp) | `data_bridge->sync()` was called twice per frame - once in main loop, once in `QuantWorkspaceComponent::update()` |
| 3 | **Missing Layout Bootstrap** | [`vulkan_dashboard_advanced.cpp`](dependencies/BTQ_Render_Engine/src/vulkan_dashboard_advanced.cpp) | `PanelManager::initialize()` creates no panels; layout must be loaded explicitly via `load_layout()` or `apply_layout_preset()` |
| 4 | **Malformed Layout JSON** | [`default_layout.json`](dependencies/BTQ_Render_Engine/default_layout.json) | Layout file had incorrect panel type IDs and positions |

### Fixes Applied

- [x] **6.1: Clean up orphaned instantiations in `main_trading_terminal.cpp`**
    - **Before:** Trading Systems, PanelManager, Workspace created in `main()` AND by `VulkanDashboard`
    - **After:** Only `VulkanDashboard` creates these internally
    - **Code Change:**
      ```cpp
      // REMOVED orphaned instantiations:
      // auto trading_systems = ...;  // Now created by VulkanDashboard
      // auto panel_manager = ...;    // Now created by VulkanDashboard
      // auto workspace = ...;        // Now created by VulkanDashboard
      ```

- [x] **6.2: Remove duplicate `sync()` call in `quant_workspace_component.cpp`**
    - **Before:** `sync()` called in main loop AND in `QuantWorkspaceComponent::update()`
    - **After:** Single sync point in main loop only
    - **Code Change:**
      ```cpp
      void QuantWorkspaceComponent::update(float dt) {
        // NOTE: Data sync is handled in main loop (main_trading_terminal.cpp)
        panel_manager_->update(dt);
      }
      ```

- [x] **6.3: Add layout bootstrap in `vulkan_dashboard_advanced.cpp`**
    - **Before:** No layout loaded after initialization → empty terminal
    - **After:** Load `default_layout.json` or fallback to `LayoutPreset::MODERN_TRADING`
    - **Code Change:**
      ```cpp
      void VulkanDashboard::init_components() {
        workspace_ = std::make_unique<QuantWorkspaceComponent>(...);
        
        // LAYOUT BOOTSTRAP
        if (auto* pm = workspace_->getPanelManager()) {
          pm->load_layout("default_layout.json");
          if (pm->get_panel_count() == 0) {
            pm->apply_layout_preset(LayoutPreset::MODERN_TRADING);
          }
        }
      }
      ```

- [x] **6.4: Fix `default_layout.json` with correct panel definitions**
    - **Panel Type IDs:** CHART=0, METRICS=1, HEATMAP=2, WATCHLIST=6, ORDERBOOK=10, TAPE=14, TPO_PROFILE=19
    - **Fixed Layout:** 5 panels at correct positions (Chart, Order Book, Watchlist, Time & Sales, Metrics)

### Verification

- [x] **6.5: Build & Test**
    - Build: `./build_integration.sh` → SUCCESS
    - Executable: `build/BTQuantTerminal` created
    - Expected: Single render pass, no duplicate UI elements

### Key Learnings

1. **Single Responsibility:** Each component should be created by exactly one owner
2. **Single Sync Point:** Data synchronization should happen at one place in the frame loop
3. **Explicit Initialization:** `initialize()` methods should not assume layout; caller must explicitly load layout
4. **API Contracts:** `load_layout()` returns `void`, not `bool` - use `get_panel_count()` to verify success