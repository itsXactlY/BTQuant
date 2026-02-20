# TASK_ULTIMA_MMT_GENESIS.md

**Objective:** The absolute, feature-for-feature reconstruction of Market Monkey Terminal (MMT).
**Architecture:** Zero-allocation C++ memory spine. Lock-free SPSC routing. Vulkan compute for liquidity matrices. ImGui borderless presentation. Only GPU ONLY rendering for DOM and Charts. No CPU-side copies of market data. Sub-millisecond UI latency. No CPU-side rendering of DOM or Charts. The Tape is the only component with direct CPU access to market data, and it must be strictly O(1) via a lock-free ring buffer. No CPU-side rendering of the DOM or Charts; all visual elements must be GPU-accelerated. The UI must be a perfect clone of MMT's deep void aesthetic, including the exact color palette, font rendering, and layout. The project will be executed in 8 distinct phases, each with specific technical milestones and target files for implementation.

---

## PHASE 0: THE BARE METAL SPINE (O(1) ROUTING)
**Target Files:** `include/data/core_types.hpp`, `include/threading/lockfree_queue.hpp`, `include/memory/memory_pool.hpp`

- [ ] **0.1. Trivial Data Pods:** Define `TradeData`, `PriceLevel`, and `OrderBookSnapshot` as strictly `std::is_trivial_v` and `std::is_standard_layout_v`.
- [ ] **0.2. Cache-Line Alignment:** Pad `TradeData` and `OrderBookSnapshot` with `alignas(64)` to completely eliminate false sharing across CPU cores.
- [ ] **0.3. The Monolithic Arena:** Implement `MemoryArena` using a single 1GB OS-level allocation (`mmap` / `VirtualAlloc`) on startup. Purge all `new`/`delete` calls across the entire codebase.
- [ ] **0.4. Lock-Free Free-List:** Build an O(1) stack of pointers for `MemoryArena::acquire()` and `MemoryArena::release()` using `std::atomic_compare_exchange_weak`.
- [ ] **0.5. SPSC Ring Buffer:** Implement a Single-Producer Single-Consumer ring buffer with a power-of-2 capacity. Use bitwise `& (capacity - 1)` for routing ticks from Network to UI.
- [ ] **0.6. Atomic Memory Barriers:** Enforce `std::memory_order_release` for the writer and `std::memory_order_acquire` for the reader in the ring buffer.

---

## PHASE 1: VULKAN COMPUTE & GPU HEATMAP (THE LIQUIDITY MATRIX)
**Target Files:** `src/vulkan/lob_heatmap_compute_pipeline.cpp`, `src/vulkan/ssbo_snapshot_updater.cpp`, `src/system/GPUMemoryManager.cpp`

- [ ] **1.1. Persistent SSBO Mapping:** Map the Vulkan Shader Storage Buffer Object (SSBO) with `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`. Keep it mapped permanently for zero-copy L2 data uploads.
- [ ] **1.2. STD430 Alignment:** Guarantee the C++ `OrderBookSnapshot` array perfectly matches `std430` GPU memory alignment rules (16-byte `vec4` strides).
- [ ] **1.3. Compute Shader Dispatch:** Configure `lob_heatmap.comp` workgroup sizes to strictly `local_size_x = 16, local_size_y = 16`. Dispatch via `vkCmdDispatch`.
- [ ] **1.4. Heatmap Color Math:** In the compute shader, map liquidity depth to an MMT-style gradient (Black/Void for empty, Green/Orange for high resting volume).
- [ ] **1.5. Pipeline Barriers:** Execute explicit `vkCmdPipelineBarrier` transitions from `VK_IMAGE_LAYOUT_GENERAL` to `SHADER_READ_ONLY_OPTIMAL` before handing the texture to ImGui.
- [ ] **1.6. ImGui Texture Hook:** Register the `VkImageView` via `ImGui_ImplVulkan_AddTexture()` and expose the descriptor to the UI layer.

---

## PHASE 2: THE DEEP VOID AESTHETIC & DOCKING
**Target Files:** `src/ui/unified_theme_system.cpp`, `src/ui/font_manager.cpp`, `src/ui/layout_manager.cpp`

- [ ] **2.1. Eradicate Borders:** Override `ImGuiStyle`: Set `WindowBorderSize = 0.0f`, `ChildBorderSize = 0.0f`, `FrameBorderSize = 0.0f`, and `WindowRounding = 0.0f`.
- [ ] **2.2. MMT Color Palette:** Map `ImGuiCol_WindowBg` to `#0B0E11` and `ImGuiCol_ChildBg` to `#15191E`.
- [ ] **2.3. Sub-Pixel Rasterization:** Load JetBrains Mono at 11px using `OversampleH = 4`, `OversampleV = 4`, and `PixelSnapH = false`. Merge FontAwesome 6 glyphs directly into the atlas.
- [ ] **2.4. Programmatic Docking:** Slice the root `ImGuiDockNode` strictly: Left (Sidebar), Center (Charts), Right Top (DOM/Orderbook), Right Bottom (Tape).

---

## PHASE 3: THE TAPE (TIME & SALES / FLUX DES TRADES)
**Target Files:** `src/components/tape_panel.cpp`, `src/components/historical_time_sales.cpp`

- [ ] **3.1. Ring Buffer Binding:** Map the Tape strictly to the `SpscRingBuffer<TradeData>`. Never copy the data; read via `std::memory_order_consume`.
- [ ] **3.2. Virtualized List:** Implement `ImGuiListClipper`. Render only the currently visible rows to maintain sub-millisecond draw times.
- [ ] **3.3. Dynamic Alpha Mapping:** Calculate the current trade size percentile relative to the last 500 trades. Map this to a background row alpha (`0.05` to `0.50`).
- [ ] **3.4. Aggression Coloring:** Hardcode Market Buys (Ask hits) to Neon Mint/Green, Market Sells (Bid hits) to Crimson/Red.
- [ ] **3.5. Slippage / Sweep Brackets:** Compare `trade[i]` and `trade[i-1]`. If `delta_time < 50ms` and `price_diff != 0`, draw a vertical `1px` white bracket linking the executions.
- [ ] **3.6. Size Filtering:** Implement a UI threshold slider. Skip rendering any execution below the integer filter size.

---

## PHASE 4: THE DOM (DEPTH OF MARKET)
**Target Files:** `src/components/dom_surface_panel.cpp`, `src/components/orderbook_panel.cpp`

- [ ] **4.1. DOM Table Layout:** Implement a transparent 5-column `ImGuiTable`: `[Buys | Asks | Price | Bids | Sells]`.
- [ ] **4.2. Liquidity Surface Background:** Render the Vulkan Heatmap texture directly behind the transparent DOM table using `draw_list->AddImage()`.
- [ ] **4.3. Auto-Centering Mechanism:** Implement mathematical centering: If `abs(live_price - center_y) > threshold`, smoothly interpolate the Y-scroll to lock the live price back to the center.
- [ ] **4.4. Hardware Instanced Volume Bars:** Bypass `ImGui::AddRectFilled()` loops. Construct a raw contiguous vertex array for Bid/Ask liquidity bars and submit via a single `draw_list->AddDrawCmd()`.
- [ ] **4.5. Live Trade Execution Bubbles:** Pull data from the Tape and overlay transient circle markers on the DOM Price column where the latest hits occurred.

---

## PHASE 5: FOOTPRINT & VOLUME PROFILE (ORDERFLOW)
**Target Files:** `src/analytics/cluster_engine.cpp`, `src/components/footprint_panel.cpp`, `src/components/volume_profile_panel.cpp`

- [ ] **5.1. O(1) Cluster Binning:** Implement constant-time binning for incoming trades: `bin_index = (trade.price - day_low) / tick_size`.
- [ ] **5.2. CAS Volume Accumulation:** Use Compare-And-Swap (`std::atomic_compare_exchange_weak`) to safely update `buyVolume` and `sellVolume` on active footprint clusters without locking.
- [ ] **5.3. Diagonal Imbalance Detection:** Compare Bid volume at level N with Ask volume at level N+1. If ratio > 3.0, draw a bold `2px` boundary box.
- [ ] **5.4. Dynamic POC (Point of Control):** Maintain a running maximum `totalVolume` pointer natively within the CAS loop to avoid sorting the array upon every tick.
- [ ] **5.5. Cumulative Volume Delta (CVD):** Track net delta globally via an `std::atomic<int64_t>`, adding Ask hits and subtracting Bid hits.

---

## PHASE 6: MARKET PROFILE (TPO)
**Target Files:** `src/analytics/tpoengine.cpp`, `src/components/tpo_panel.cpp`

- [ ] **6.1. ASCII Time Brackets:** Map 30-minute epochs to ASCII characters (A-Z, a-z).
- [ ] **6.2. Profile Accumulation:** Build a distribution matrix mapping `price_level` -> `std::vector<char>`. 
- [ ] **6.3. Raw Text Rendering:** Iterate the TPO matrix and render characters horizontally using raw `draw_list->AddText()` for maximum efficiency.
- [ ] **6.4. Single Print Highlighting:** Identify price levels with strictly one TPO character, bounded by levels with > 1. Highlight with an accent background (e.g., `#4A90FF`).
- [ ] **6.5. Value Area (VA) Math:** Implement outward iterative expansion from the POC until 68% of the total daily characters are enclosed. Highlight the VA zone.

---

## PHASE 7: GLOBAL SYNC & PERFORMANCE
**Target Files:** `src/components/quant_workspace_component.cpp`, `src/rendering/frame_pacer.cpp`, `src/performance/regression_detector.cpp`

- [ ] **7.1. Atomic Crosshair:** Maintain `std::atomic<double> g_crosshair_price`. Read across all Charts, DOMs, and TPOs to draw a synchronized `1px` dashed line.
- [ ] **7.2. Microsecond Frame Pacing:** Use `std::chrono::high_resolution_clock`. If rendering completes faster than the display V-Sync (< 6.94ms), yield the CPU thread back to the ingestion engine.
- [ ] **7.3. Adaptive LOD (Level of Detail):** Monitor chart zoom. If Footprint or TPO cells compress to < 4px height, suppress text rendering completely.
- [ ] **7.4. Bare-Metal Telemetry:** Utilize TSC intrinsics (`__rdtsc()`) to measure microsecond latency between network ingress and UI render completion. If the 99th percentile frame exceeds 10ms, trigger an alert.