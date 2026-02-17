# TASK_ULTIMA_GENESIS_MMT_CLONE.md

**Objective:** Absolute architectural alignment with the MMT (Market Monkey Terminal) specification.
**Core Directives:** Zero `std::mutex` on the hot path. Zero runtime allocations. Sub-millisecond ImGui presentation. Pure GPU compute for liquidity surfaces.

---

## PHASE 1: THE GREAT PURGE (ERADICATION OF BLOAT)
**Objective:** Strip the codebase to a flawless, disgusting bare bone. Delete UI hand-holding and legacy bridges.

- [x] **01. Destroy Legacy Bridges:** Delete `src/data/hotspine_data_bridge.cpp` and `include/hotspine_data_bridge.hpp`. The architecture is moving to direct memory-mapped SSBOs; legacy polling is dead.
- [x] **02. Purge UI Fluff:** Delete `src/ui/tutorial.cpp`, `src/ui/loading_states.cpp`, `src/ui/empty_states.cpp`, and `src/ui/haptic_feedback.cpp`. A high-performance terminal renders the void when data is absent; it does not render loading spinners.
- [x] **03. Eradicate Theme Editors:** Delete `src/components/theme_customization_panel.cpp`. The MMT aesthetic is hardcoded. Users do not get to choose their colors.
- [x] **04. Strip ImGui Overlays:** Open `src/ui/tooltips.cpp` and `src/ui/context_menus.cpp`. Remove all `ImGui::PushStyleVar` calls that add rounding, borders, or drop shadows.
- [x] **05. Standardize Panel Inheritance:** Audit all panels in `src/components/` (e.g., `dom_surface_panel.cpp`, `tape_panel.cpp`). Ensure they strictly inherit from `PanelBase` and implement ONLY `render_content()`.

---

## PHASE 2: MEMORY & CONCURRENCY (THE ZERO-LATENCY SPINE)
**Objective:** Guarantee O(1) memory access and zero OS-level context switches during data ingestion.

- [x] **06. Trivial Struct Alignment:** Open `include/data/core_types.hpp` and `include/data/data_types.hpp`. Force `TradeData`, `PriceLevel`, and `VolumeProfileNode` to be `std::is_trivial_v`. Add `alignas(64)` to prevent cache-line false sharing across CPU cores.
- [x] **07. Monolithic Memory Arena:** In `src/memory/memory_pool.cpp`, replace vector allocations with a single 1GB `mmap` (Linux) or `VirtualAlloc` (Windows) block on startup.
- [x] **08. Lock-Free Slab Allocator:** Implement a lock-free free-list stack in `memory_pool.cpp` using `std::atomic<Node*>`. Ensure `acquire()` and `release()` operate in constant time.
- [x] **09. Ring Buffer Implementation:** In `src/threading/lockfree_queue.cpp`, rewrite the queue as a strict Single-Producer/Single-Consumer (SPSC) ring buffer.
- [x] **10. Bitwise Modulo:** Force the `LockFreeQueue` capacity to be a power of 2 (e.g., 65536) and replace `% capacity` with bitwise AND (`& (capacity - 1)`) for microsecond routing speed.
- [x] **11. Strip Mutexes:** Open `src/data/market_data_processor.cpp`. Find and delete every `std::mutex` and `std::lock_guard`.
- [x] **12. Atomic State Routing:** In `market_data_processor.cpp`, implement atomic double-buffered pointers for the Best Bid/Offer (BBO) state using `std::memory_order_acquire` (UI thread) and `std::memory_order_release` (Network thread).

---

## PHASE 3: THE MMT AESTHETIC & THE DEEP VOID
**Objective:** Sub-pixel typography and absolute zero-border integration.

- [x] **13. The Deep Void Palette:** In `src/ui/unified_theme_system.cpp`, hardcode the MMT palette:
    - `ImGuiCol_WindowBg` = `#0B0E11`
    - `ImGuiCol_ChildBg` = `#15191E`
    - `ImGuiCol_Text` = `#D1D4DC`
    - `ImGuiCol_Separator` = `rgba(94, 82, 64, 0.2)`
- [x] **14. Border Decimation:** Set `WindowBorderSize = 0.0f`, `ChildBorderSize = 0.0f`, `FrameBorderSize = 0.0f`, and `WindowRounding = 0.0f` globally.
- [x] **15. Font Rasterization:** In `src/ui/font_manager.cpp`, configure `ImFontConfig` for JetBrains Mono (or Berkeley Mono) with `OversampleH = 4` and `OversampleV = 4`. Set `PixelSnapH = false`.
- [x] **16. Glyph Merging:** Load FontAwesome 6 icons into the same font atlas as the primary text to prevent texture swapping during render passes.
- [x] **17. Strict Docking:** In `src/ui/layout_manager.cpp`, override user `.ini` files. Programmatically split the root `ImGuiDockNode` into the 5-region MMT layout (Tools Left 3%, Charts Center, DOM/OB Right-Top, Tape Right-Bottom).
- [x] **18. Global Crosshair:** In `src/components/quant_workspace_component.cpp`, initialize `std::atomic<double> g_crosshair_price`. All charting panels must read this and draw a synchronized `1px` dashed line.

---

## PHASE 4: VULKAN COMPUTE & GPU MEMORY
**Objective:** Offload all order book aggregation and heatmap generation to the GPU.

- [x] **19. Buffer Alignment:** In `src/vulkan/ssbo_snapshot_updater.cpp`, enforce `std430` alignment for the `OrderBookSnapshot` payload. Ensure it maps perfectly to an array of `vec4` in the compute shader.
- [x] **20. Persistent Mapping:** Refactor `ssbo_snapshot_updater.cpp` to use `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`. Keep the memory persistently mapped to avoid `vkMapMemory` overhead on the hot path.
- [x] **21. Shader Dispatch:** In `src/vulkan/lob_heatmap_compute_pipeline.cpp`, configure the workgroup size to `local_size_x = 16, local_size_y = 16`.
- [x] **22. Image Transitions:** Insert explicit `vkCmdPipelineBarrier` calls in the compute pipeline to transition the heatmap texture from `VK_IMAGE_LAYOUT_GENERAL` to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` before handing it to ImGui.
- [x] **23. ImGui Texture Hook:** In `src/system/GPUMemoryManager.cpp`, bind the computed `VkImageView` to ImGui using `ImGui_ImplVulkan_AddTexture` and cache the descriptor set.
- [x] **24. Dirty Compute Flag:** Implement an atomic check in `vulkan_dashboard_advanced.cpp` to only dispatch the heatmap compute shader if `incremental_updater.cpp` signals a state change.

---

## PHASE 5: RENDER BATCHING & CULLING
**Objective:** Prevent ImGui from choking on massive data sets.

- [x] **25. Vertex Pre-allocation:** In `src/rendering/imgui_optimizer.cpp`, override ImGui's buffer sizing. Force `VtxBuffer.reserve(250000)` and `IdxBuffer.reserve(500000)` on startup.
- [x] **26. AABB Spatial Culling:** In `src/rendering/chart_culler.cpp`, fetch `ImPlot::GetPlotLimits()`. If an incoming footprint cell or trade circle is outside these bounds, execute an immediate `continue`. Emit zero vertices.
- [x] **27. DOM Hardware Instancing:** In `src/rendering/orderbook_batcher.cpp`, bypass standard `ImGui::AddRectFilled`. Construct a contiguous vertex array for the Bids/Asks liquidity bars and submit via a single `draw_list->AddDrawCmd()`.
- [x] **28. Frame Pacing:** In `src/rendering/frame_pacer.cpp`, use `std::chrono::high_resolution_clock`. If the render finishes in `< 6.94ms` (for 144Hz), sleep the UI thread to yield CPU cache back to the network ingestion thread.
- [x] **29. Footprint LOD:** In `src/rendering/footprint_lod.cpp`, monitor vertical chart zoom. If a TPO block or footprint cell compresses to `< 4px` in height, suppress text rendering completely. Draw only the intensity quad.

---

## PHASE 6: ORDER FLOW & ANALYTICS PANELS
**Objective:** Construct the high-fidelity visualizations.

- [x] **30. The Tape (Time & Sales):** In `src/components/time_and_sales.cpp`, use `ImGuiListClipper`. Read directly from the atomic tail of the trade ring buffer.
- [x] **31. Tape Alpha Mapping:** Calculate dynamic background alpha (`0.05` to `0.50`) based on the trade size percentile within a trailing 500-trade window.
- [x] **32. Slippage Brackets:** In the tape, if consecutive trades execute `< 50ms` apart at different prices, draw a `1px` white bracket linking them.
- [x] **33. Orderbook DOM:** In `src/components/orderbook_panel.cpp`, split the panel strictly 50/50. Asks descending, Bids ascending. Render a 40px distinct mid-price block.
- [x] **34. Liquidity Surface:** In `src/components/dom_surface_panel.cpp`, push a transparent `ImGuiCol_ChildBg`. Render the `GPUMemoryManager` texture behind a 5-column `ImGuiTable` (`[Buys|Asks|Price|Bids|Sells]`).
- [x] **35. Auto-Centering:** Implement `PriceScaleMode::Auto` in the DOM. Check if `abs(current_price - center_price) > threshold`. If true, smoothly interpolate the Y-axis center to follow price.
- [x] **36. Cluster Engine O(1):** In `src/analytics/cluster_engine.cpp`, implement constant-time price binning: `bin_index = (trade.price - session_low) / tick_size`.
- [x] **37. CAS Volume Accumulation:** In `cluster_engine.cpp`, use Compare-And-Swap (CAS) atomic operations to increment `buyVolume` and `sellVolume` on the active `CandleCluster`.
- [x] **38. Imbalance Detection:** In `src/components/footprint_panel.cpp`, compare Bid volume at level N with Ask volume at N+1. If the ratio > 3.0, draw a `2px` colored boundary box.

---

## PHASE 7: EXECUTION & RISK SUBSYSTEM
**Objective:** Zero-latency routing from the UI click to the network socket.

- [x] **39. Quick Order Buttons:** In `src/components/chart_panel.cpp`, overlay massive `BUY MKT` and `SELL MKT` buttons directly inside the chart boundaries reading the live atomic mid-price.
- [x] **40. Command SPSC Queue:** In `src/trading/trade_command_queue.cpp`, allocate a dedicated SPSC ring buffer specifically for outbound `TradeCommand` structs.
- [x] **41. O(1) Risk Assessment:** Before routing to the exchange API, pass the order through `src/trading/risk_assessment.cpp`. Implement atomic checks for Daily Loss Limit and Max Position Size without acquiring any locks.
- [x] **42. Rejection Routing:** If a risk invariant fails, push an error code to a reverse `LockFreeQueue<ExecutionReport>`. Render this immediately in `src/components/alerts_panel.cpp` in `ASK_RED`.
- [x] **43. Zero-Latency PnL:** In `src/trading/position_manager.cpp`, calculate live Mark-to-Market PnL strictly using the atomic `best_bid`/`best_ask` pointers supplied by the `market_data_processor`. Ensure rendering in `trading_positions_panel.cpp` is V-Sync locked to prevent tearing.












































## Phase -1: Bugfixing.
- [x] Build and run ./build_integration.sh confirm it builds. Run ./build/BTQTerminal with an 15second timeout to confirm it starts, not segfaulting, or worser. As soon an pair gets selected, it segfaults.

 ╭─alca@alca in repo: PubBTQuant/dependencies/BTQ_Render_Engine on  0.0.2-1 [$x!?] via △ v4.2.1 as 🧙 took 2m12s
[🔴] × ./build/BTQuantTerminal
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763845000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763845000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763845000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: [WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.071991.07, Timestamp: , Timestamp: 17712817638450001771281763845000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763845000

[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763845000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763845000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: [WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763853000
1991.07, Timestamp: 1771281763853000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763853000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281763853000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764018000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764018000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764018000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764018000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764105000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764517000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764517000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764517000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.07, Timestamp: 1771281764517000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.06, Timestamp: 1771281765380000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.06, Timestamp: 1771281765380000
[WatchlistPanel] Market data update received for [WatchlistPanel] Market data update received for BNB-USDT (ID: BNB-USDT (ID: 1001210012). New price: ). New price: 1991.061991.06, Timestamp: 1771281765380000, Timestamp: 
1771281765380000
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.06, Timestamp: 1771281765380000
double free or corruption (!prev)
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.06, Timestamp: 1771281765380000
double free or corruption (!prev)
[WatchlistPanel] Market data update received for BNB-USDT (ID: 10012). New price: 1991.06, Timestamp: 1771281765380000
double free or corruption (!prev)
fish: Job 1, './build/BTQuantTerminal' terminated by signal SIGABRT (Abbruch)



fish: Job 1, './build/BTQuantTerminal' terminated by signal SIGABRT (Abbruch)
# TASK_ULTIMA_GENESIS: MMT CLONE (2000+ FPS ARCHITECTURE)

**Objective:** Rebuild BTQ Terminal from Ground Zero. Institutional order flow, zero page-faults, 2000+ FPS rendering loop, lock-free memory spine.
**Constraints:** No `std::mutex`. No `new`/`delete` after initialization. `std::is_trivial_v` for all payloads. Uncapped Vulkan swapchain.

---

## PHASE 1: SCORCHED EARTH & COMPILER OVERDRIVE
**Objective:** Wipe the decaying logic and optimize the compiler output for bare-metal execution.

- [ ] **01. File Purge:** Delete the contents of `src/data/`, `src/components/`, `src/analytics/`, and `src/threading/`. We are starting with empty text files.
- [ ] **02. CMake Optimization:** Open `CMakeLists.txt`. Force aggressive optimizations.
  - Add `-O3 -march=native -flto -fno-rtti` to `CMAKE_CXX_FLAGS_RELEASE`.
  - Enable Link-Time Optimization (LTO) globally.
- [ ] **03. Vulkan Swapchain Uncapping:** Open `src/system/VulkanCore.cpp` (or your glfw/vulkan init).
  - Find `VkSwapchainCreateInfoKHR`. Set `presentMode` to `VK_PRESENT_MODE_IMMEDIATE_KHR` (uncapped) or `VK_PRESENT_MODE_MAILBOX_KHR` (fastest tear-free). Do NOT use `FIFO`.

---

## PHASE 2: THE ZERO-ALLOCATION SPINE
**Objective:** Create the foundational data structures and the monolithic memory arena. No `double free` is possible if we never call `free()`.

- [ ] **04. Trivial Structs:** In `include/data/core_types.hpp`, define strict Standard-Layout structs.
  - `TradeData`: `alignas(64)` with `uint64_t timestamp`, `double price`, `double size`, `bool is_buyer_maker`.
  - `PriceLevel`: `double price`, `double size`.
  - `OrderBookSnapshot`: `alignas(64)` containing `uint64_t timestamp`, `PriceLevel bids[100]`, `PriceLevel asks[100]`.
  - Add `static_assert(std::is_trivial_v<TradeData>);`
- [ ] **05. Monolithic Arena:** In `include/memory/memory_pool.hpp`, implement `class MemoryArena`.
  - On startup, execute `std::malloc(1024 * 1024 * 500)` (500MB contiguous block).
- [ ] **06. Lock-Free Free-List:** Subdivide the 500MB block into fixed chunks matching the size of `OrderBookSnapshot`. 
  - Push all chunk pointers onto an `std::atomic<uintptr_t>` stack.
  - Implement `void* allocate()` using an O(1) `std::atomic_compare_exchange_weak` pop.
  - Implement `void deallocate(void* ptr)` using an atomic push.

---

## PHASE 3: THE SPSC ATOMIC ROUTER
**Objective:** Microsecond routing from the Network Ingestion thread to the Engine UI thread.

- [ ] **07. Bitwise Ring Buffer:** In `include/threading/lockfree_queue.hpp`, implement a template `SpscRingBuffer<T, Capacity>`.
  - `Capacity` MUST be a power of 2 (e.g., `65536`).
  - Use `index & (Capacity - 1)` instead of modulo `%` for extreme indexing speed.
- [ ] **08. Memory Ordering:** Implement `push()` using `std::memory_order_release` when updating the tail. Implement `pop()` using `std::memory_order_acquire` when reading the tail.
- [ ] **09. Atomic L2 State:** In `include/market_data_processor.hpp`, declare an array of atomic pointers: `std::atomic<OrderBookSnapshot*> current_book[MAX_SYMBOLS];`.
- [ ] **10. Pointer Swapping:** When the network thread builds a new book, pop memory from `MemoryArena`, populate it, and execute `current_book[sym].store(new_ptr, std::memory_order_release)`.
- [ ] **11. UI Consumption:** The UI panels read the book strictly via `OrderBookSnapshot* active = current_book[sym].load(std::memory_order_acquire)`.

---

## PHASE 4: 2000+ FPS RENDER OPTIMIZATIONS
**Objective:** Modify ImGui and your rendering logic to prevent the CPU from stalling during draw calls.

- [ ] **12. Buffer Pre-allocation:** In `src/rendering/imgui_optimizer.cpp`, override ImGui's dynamic vector growth. 
  - Call `ImGui::GetIO().BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;`.
  - On frame start, ensure `draw_list->VtxBuffer.reserve(500000)` and `IdxBuffer.reserve(1000000)` to prevent runtime reallocations during volatile market drops.
- [ ] **13. Hardware Instancing (DOM Bars):** In `src/rendering/orderbook_batcher.cpp`, stop calling `ImGui::GetWindowDrawList()->AddRectFilled()` for every DOM level.
  - Pre-calculate the vertices for all 200 DOM bars.
  - Push them directly as a raw array into `draw_list->VtxBuffer` and `draw_list->IdxBuffer`.
  - Add a single `ImDrawCmd` to the draw list.
- [ ] **14. AABB Spatial Culling:** In `src/rendering/chart_culler.cpp`, implement a fast Axis-Aligned Bounding Box intersection test.
  - Before rendering any Footprint cell or Trade circle, check if its Price/Time falls strictly within `ImPlot::GetPlotLimits()`. 
  - If it is off-screen, instantly `continue` the loop. Emit 0 vertices.

---

## PHASE 5: MMT VISUAL IDENTITY (THE DEEP VOID)
**Objective:** Pixel-perfect MMT aesthetic implementation. Zero borders. Sub-pixel text.

- [ ] **15. MMT Color Matrix:** In `src/ui/unified_theme_system.cpp`, override `ImGuiStyleColorsDark()`.
  - `ImGuiCol_WindowBg` = `#0B0E11`
  - `ImGuiCol_ChildBg` = `#15191E`
  - `ImGuiCol_Text` = `#D1D4DC`
  - `ImGuiCol_Separator` = `rgba(94, 82, 64, 0.2)`
  - Set `WindowBorderSize`, `ChildBorderSize`, `FrameBorderSize`, and `WindowRounding` to `0.0f`.
- [ ] **16. Sub-Pixel Typography:** In `src/ui/font_manager.cpp`, load *JetBrains Mono* (or Berkeley Mono) at size 11px.
  - Configure `ImFontConfig::OversampleH = 4` and `OversampleV = 4`.
  - Disable pixel snapping: `config.PixelSnapH = false;`.
- [ ] **17. Strict Docking Matrix:** In `src/ui/layout_manager.cpp`, define a single master `ImGuiID`.
  - Use `ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_PassthruCentralNode`.
  - Slicing: `Left` (3% for icons), `Center` (Main Chart), `Right Top` (DOM/OB), `Right Bottom` (Tape).
- [ ] **18. Global Sync Crosshair:** In `quant_workspace_component.hpp`, define `std::atomic<double> g_crosshair_price`.
  - All chart and DOM panels read this via `memory_order_relaxed` to draw a synchronized 1px dashed line.

---

## PHASE 6: VULKAN GPU COMPUTE (LIQUIDITY HEATMAP)
**Objective:** Process the massive L2 historical arrays purely on the GPU.

- [ ] **19. STD430 Alignment:** In `src/vulkan/ssbo_snapshot_updater.cpp`, guarantee your historical DOM array exactly matches `std430` shader layout rules (groups of 16 bytes / `vec4`).
- [ ] **20. Persistent Mapped Buffers:** Allocate the SSBO using `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`. Keep it mapped permanently.
- [ ] **21. Shader Dispatch:** In `src/vulkan/lob_heatmap_compute_pipeline.cpp`, dispatch the compute shader with `vkCmdDispatch` using `local_size_x = 16, local_size_y = 16`.
- [ ] **22. Image Barrier:** Execute `vkCmdPipelineBarrier` to transition the heatmap `VkImage` from `GENERAL` to `SHADER_READ_ONLY_OPTIMAL`.
- [ ] **23. ImGui Binding:** Extract the `ImTextureID` via `ImGui_ImplVulkan_AddTexture()` and render it behind the DOM using `draw_list->AddImage()`.

---

## PHASE 7: DOM & TAPE PANEL RENDERING
**Objective:** The institutional order flow widgets.

- [ ] **24. The DOM Structure:** In `src/components/dom_surface_panel.cpp`, push `ImGuiCol_ChildBg` to completely transparent.
  - Setup a 5-column `ImGuiTable`: `[Buys | Asks | Price | Bids | Sells]`.
  - Draw the Vulkan heatmap texture as the background under the table.
- [ ] **25. Auto-Center Logic:** Implement mathematical auto-centering in the DOM. If `abs(current_price - center_price) > threshold`, smoothly interpolate the Y-axis center to follow the live price.
- [ ] **26. The Tape (Time & Sales):** In `src/components/tape_panel.cpp`, instantiate an `ImGuiListClipper`.
  - Bind the clipper row count strictly to the atomic elements in the `SpscRingBuffer<TradeData>`.
  - Calculate row background alpha (`0.05` to `0.50`) dynamically based on trade size percentile relative to the last 500 trades.
- [ ] **27. Slippage Brackets:** In the Tape, track `trade[i].timestamp - trade[i-1].timestamp`. If the delta is `< 50ms` and the price changed, use `draw_list->AddLine()` to draw a 1px white bracket on the left edge connecting the two rows.