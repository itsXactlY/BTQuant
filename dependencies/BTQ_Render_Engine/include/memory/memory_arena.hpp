#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace BTQuant {

/// CPU-side memory arena with O(1) bump allocation and CAS-based free-list.
/// Single 1GB mmap reservation on construction. Thread-safe for concurrent acquire/release.
/// Does NOT replace GPUMemoryManager (VkDeviceMemory) — this is for TradeData arrays,
/// OrderBookSnapshot buffers, and analytics structs.
class MemoryArena {
 public:
  explicit MemoryArena(size_t size_bytes = 1ULL << 30) : capacity_(size_bytes) {
#ifdef _WIN32
    base_ = static_cast<uint8_t*>(
        VirtualAlloc(nullptr, size_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    base_ = static_cast<uint8_t*>(mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    if (base_ == MAP_FAILED) base_ = nullptr;
#endif
  }

  ~MemoryArena() {
    if (!base_) return;
#ifdef _WIN32
    VirtualFree(base_, 0, MEM_RELEASE);
#else
    munmap(base_, capacity_);
#endif
  }

  // Non-copyable, non-movable
  MemoryArena(const MemoryArena&) = delete;
  MemoryArena& operator=(const MemoryArena&) = delete;

  /// O(1) bump allocator. Tries free-list first (CAS pop), then atomic bump.
  /// alignment must be power-of-2. Returns nullptr if arena is exhausted.
  [[nodiscard]] void* acquire(size_t bytes, size_t alignment = 64) noexcept {
    if (!base_ || bytes == 0) return nullptr;

    // 1) Try free-list first
    FreeNode* node = free_head_.load(std::memory_order_acquire);
    while (node) {
      if (node->size >= bytes) {
        if (free_head_.compare_exchange_weak(node, node->next, std::memory_order_release,
                                             std::memory_order_acquire)) {
          return static_cast<void*>(node);
        }
        // CAS failed, retry with updated node
        continue;
      }
      break;  // Free-list block too small, fall through to bump
    }

    // 2) Atomic bump allocation
    size_t current = offset_.load(std::memory_order_relaxed);
    size_t aligned;
    size_t new_offset;
    do {
      aligned = (current + alignment - 1) & ~(alignment - 1);
      new_offset = aligned + bytes;
      if (new_offset > capacity_) return nullptr;  // Exhausted
    } while (!offset_.compare_exchange_weak(current, new_offset, std::memory_order_release,
                                            std::memory_order_relaxed));

    return static_cast<void*>(base_ + aligned);
  }

  /// Returns memory to the lock-free free-list (O(1) CAS push).
  /// Only call with pointers originally returned from acquire().
  void release(void* ptr, size_t bytes) noexcept {
    if (!ptr || bytes < sizeof(FreeNode)) return;

#ifndef NDEBUG
    std::memset(ptr, 0xDE, bytes);  // Poison on debug builds
#endif

    auto* node = static_cast<FreeNode*>(ptr);
    node->size = bytes;
    FreeNode* old_head = free_head_.load(std::memory_order_relaxed);
    do {
      node->next = old_head;
    } while (!free_head_.compare_exchange_weak(old_head, node, std::memory_order_release,
                                               std::memory_order_relaxed));
  }

  size_t used_bytes() const noexcept { return offset_.load(std::memory_order_acquire); }

  size_t capacity_bytes() const noexcept { return capacity_; }

  bool is_valid() const noexcept { return base_ != nullptr; }

 private:
  uint8_t* base_ = nullptr;
  size_t capacity_ = 0;
  std::atomic<size_t> offset_ = 0;  // Bump pointer

  // Free-list: intrusive stack via CAS
  struct FreeNode {
    FreeNode* next;
    size_t size;
  };
  std::atomic<FreeNode*> free_head_ = nullptr;
};

/// Global singleton — initialized in main() before any subsystem.
/// Uses MAP_NORESERVE so the 1GB virtual reservation doesn't consume
/// physical RAM until pages are touched.
inline MemoryArena g_arena;

}  // namespace BTQuant
