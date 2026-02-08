
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
- [ ] **Parallel Processing (`<execution>`):**
  - In `ClusterEngine::process_trade_batch`, use `std::for_each(std::execution::par_unseq, ...)` to vectorize volume calculations across the batch before merging.

## 40.5: Validation
- [ ] **Benchmark:** Run `MarketDataProcessor` with 1M messages/sec replay.
  - **Expectation:** CPU usage should be high (processing) but uniform across cores. No "spikes" or "stalls".
- [ ] **Leak Check:** Verify Hazard Pointers correctly reclaim memory after the 4-hour window moves.