
**Tech Stack:** C++23/26, existing HotspineDataBridge  

---

## Development Guidelines
- Each module can be developed in parallel by separate developers
- Dependencies between modules are clearly marked
- Estimated complexity: S (Small, 1-2 days), M (Medium, 3-5 days), L (Large, 1+ weeks)
- Prerequisites must be completed before starting dependent tasks
- Your workdir is exlusivly /home/alca/projects/PubBTQuant/dependencies/ccapi/
- Never do or build TEST files - work only on livecode
- Fully autonomous handle conflicts in the most harmonic way with C++23/26 only.
---


# Phase 43: C++26 Lock-Free HotSpine Migration [Complexity: L]

**C++26 Requirements:** `<hazard_pointer>`, `<rcu>`, `std::atomic::wait/notify`, `std::for_each(std::execution::par_unseq)`, zero `std::mutex`/`std::condition_variable`/`std::shared_mutex` in hot path.

## 43.1: C++26 Lock-Free Infrastructure

- [ ] **43.1.1: C++26 Hazard Pointer Ring Buffer** [M]
  - `include/hotspine/c26_lockfree_ring.hpp`: Use `<hazard_pointer>` for safe reclamation of retired events
  - `std::hazard_ptr_registry` for producer/consumer nodes
  - `std::hazard_ptr_retire` when overwriting old slots

- [ ] **43.1.2: RCU Shared Memory Manager** [M]
  - `include/hotspine/c26_rcu_shm.hpp`: `<rcu>` for shared memory configuration
  - `std::rcu_obj_base` for ring buffer metadata
  - `std::rcu_read_lock()` in consumer, `synchronize_rcu()` on writer updates

- [ ] **43.1.3: Atomic Wait/Notify Signaling** [S]
  - `std::atomic<uint64_t> generation_{0}` → `generation_.wait(old_val)` / `generation_.notify_one()`
  - Replace all `std::condition_variable` in health monitoring

## 43.2: ExchangeConnectionManager → Pure Atomic Producer

- [ ] **43.2.1: Eliminate Session Queues** [M]
  - `exchange_connection_manager.cpp`: Remove `maxEventQueueSize`, direct atomic writes
  - CCAPI callbacks → `ring_->hazard_produce(ev)` (hazard-safe)

- [ ] **43.2.2: Par_Unseq Event Processing** [M]
  - `std::for_each(std::execution::par_unseq, batch.begin(), batch.end(), [](auto& ev){ atomic_snapshots_[ev.symbol_id].update(ev); })`

## 43.3: MarketDataProcessor → C++26 Consumer

- [ ] **43.3.1: RCU Consumer Polling** [M]
  - `std::rcu_read_lock()` around `ring_->consume(batch, MAX_BATCH)`
  - `std::for_each(std::execution::par_unseq)` for batch updates

- [ ] **43.3.2: Hazard Pointer Snapshots** [M]
  - `AtomicSnapshot` → `std::hazard_ptr<Snapshot>` for safe reader access
  - Producer retires old snapshots via `hazard_ptr_retire()`

## 43.4: C++26 Cache Warming + Branchless

- [ ] **43.4.1: Branchless Accumulators** [S]
  - `std::array<std::atomic<double>, 2> real_volume_{};` → `real_volume_[ev.is_warming()].store(+= qty, relaxed)`

- [ ] **43.4.2: Atomic Wait Warmer** [S]
  - `std::atomic<uint64_t> warmup_gen_{0}` → `warmup_gen_.wait(0); inject_dummy(); warmup_gen_.notify_one()`

## 43.5: CMake C++26 Flags

- [ ] **43.5.1: C++26 Standard** [S]
  - `CMakeLists.txt`: `set(CMAKE_CXX_STANDARD 26)` + `-std=c++26 -stdlib=libstdc++`
  - Link `-lhazardptr -lrcu` if separate libs needed

## 43.6: C++26 Validation

- [ ] **43.6.1: Zero Lock Audit** [S]
  - `grep -r "std::mutex\|std::condition_variable\|std::shared_mutex" src/data/ src/hotspine/` → 0 results

- [ ] **43.6.2: Hazard/RCU Coverage** [M]
  - All shared mutable state uses `<hazard_pointer>` or `<rcu>`
  - Producer/consumer nodes registered in global `hazard_ptr_registry`

- [ ] **43.6.3: Par_Unseq Scaling** [M]
  - Benchmark: `std::execution::par_unseq` scales to all cores on 1M events/sec

## Critical C++26 Targets

- [ ] **Zero Traditional Locks**: No `std::mutex`/`std::shared_mutex`/`std::condition_variable` anywhere
- [ ] **Hazard Pointers Everywhere**: All dynamic deallocation uses `<hazard_pointer>`
- [ ] **RCU for Config**: All shared config uses `<rcu>`
- [ ] **Atomic Wait/Notify**: All signaling uses `std::atomic::wait/notify_*`
- [ ] **Par Unseq Batches**: All event processing uses `std::execution::par_unseq`
