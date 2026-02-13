# TASK_CLEANUP.md: The Pure Quantower Amputation

**Objective:** Ruthlessly delete all rogue clones, legacy hydras, Qt-infected files, and test parasites. Leave ONLY the modern `snake_case` C++26 ImGui/Vulkan architecture.

---

## Phase 1: The Clone Wars (Delete Duplicates)
**Rule:** Keep `snake_case.cpp/.hpp`. Delete flat lowercase and `.h`.

- [x] **1.1: Purge TPO Clones & Qt Poison**
    - `rm src/components/tpoprofilepanel.cpp`
    - `rm include/components/tpoprofilepanel.h`
    - `rm include/components/moc_tpoprofilepanel.cpp*` (Kill the Qt garbage)
    - *(Action: KEEP `tpo_panel.cpp` and `tpo_panel.hpp`)*

- [x] **1.2: Purge DOM Surface Clones**
    - `rm src/components/domsurfacepanel.cpp`
    - `rm include/components/domsurfacepanel.h`
    - *(Action: KEEP `dom_surface_panel.cpp` and `dom_surface_panel.hpp`)*

- [x] **1.3: Purge Options Panel Clones**
    - `rm src/components/optionanalyticspanel.cpp`
    - `rm include/components/optionanalyticspanel.hpp`
    - *(Action: KEEP `option_analytics_panel.cpp` and `option_analytics_panel.hpp`)*

- [x] **1.4: Purge Orderbook History Clones**
    - `rm src/data/orderbookhistory.cpp`
    - `rm src/data/orderbookhistory.h`
    - *(Action: KEEP `src/components/orderbook_history.cpp` and `include/components/orderbook_history.hpp`)*

- [x] **1.5: Purge Price Statistic Clones (Qt Poison)**
    - `rm src/components/pricestatisticpanel.cpp`
    - `rm include/components/pricestatisticpanel.h`
    - `rm include/components/moc_pricestatisticpanel.cpp*`

---

## Phase 2: The Hydra Purge (Architecture Violations)
**Rule:** The Render Engine ONLY reads shared memory. It does not manage exchange APIs, legacy layouts, or conflicting OS optimizers.

- [ ] **2.1: Kill the Data/Exchange Hydras**
    - `rm src/data/exchange_aggregator.*` (including `.bak`, `.bak2`)
    - `rm include/data/exchange_aggregator.hpp`
    - `rm src/data/unified_data_pipeline.cpp`
    - `rm include/data/unified_data_pipeline.hpp`

- [ ] **2.2: Kill the Legacy Layout Hydra**
    - `rm -rf src/layout/` (Deletes `dashboard_layout_manager.cpp` & `layout_presets.cpp`)
    - `rm -rf include/layout/`
    - *(Action: KEEP `src/ui/layout_manager.cpp` and `src/ui/workspace_manager.cpp`)*

- [ ] **2.3: Kill the Conflicting Optimization Hydra**
    - `rm -rf src/optimization/` (Deletes `performance_optimizer.cpp`)
    - *(Action: KEEP `src/system/system_optimizer.cpp`)*

- [ ] **2.4: Kill the Legacy Monolith Renderer**
    - `rm src/components/MarketMicrostructureRenderer.cpp`
    - `rm include/components/MarketMicrostructureRenderer.h`

- [ ] **2.5: Kill the Zombie Dashboard**
    - `rm src/components/realtime_dashboard_component.cpp`
    - `rm include/components/realtime_dashboard_component.hpp`

---

## Phase 3: The Parasite Purge (Test Files in Production Source)
**Rule:** No `test_*.cpp` files are allowed in the `src/` directory. They break CMake globbing and pollute the production build.

- [ ] **3.1: Eradicate Test Files**
    - `rm src/test_candlestick.cpp`
    - `rm src/data/test_quality_monitor.cpp`
    - `rm src/ui/test_tooltips.cpp`
    - `rm src/ui/test_quick_actions.cpp`
    - `rm src/error_handling/test_error_handling.cpp`
    - `rm src/error_handling/test_crash_reporter.cpp`
    - `rm src/error_handling/main_test.cpp`
    - `rm src/logging/test_structured_logging.cpp`
    - `rm src/memory/test_new_memory_pools.cpp`
    - `rm src/memory/test_memory_pool.cpp`
    - `rm src/memory/test_extended_memory_pool.cpp`

---

## Phase 4: CMake Sanitization & Verification
**Rule:** The build system must reflect the amputation.

- [ ] **4.1: Update CMakeLists.txt**
    - Open `CMakeLists.txt`.
    - **Crucial:** Remove EVERY file you just deleted from the `add_executable` or `add_library` source lists. If you miss one, CMake will fail stating "Cannot find source file".

- [ ] **4.2: Clean Build**
    - Run: `rm -rf build && ./build_integration.sh`
    - Verify that CMake generates the build files and Ninja compiles without complaining about missing `.cpp` files or conflicting Qt `moc` definitions.