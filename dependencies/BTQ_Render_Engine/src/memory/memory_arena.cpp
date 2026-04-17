#include "memory/memory_arena.hpp"

#include <sys/mman.h>
#include <cstdlib>
#include <cstring>

namespace BTQuant {

MemoryArena::MemoryArena(size_t size_bytes) {
    void* ptr = mmap(nullptr, size_bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);
    if (ptr == MAP_FAILED) {
        base_ = nullptr;
        capacity_ = 0;
        return;
    }
    base_ = static_cast<uint8_t*>(ptr);
    capacity_ = size_bytes;
    offset_.store(0, std::memory_order_relaxed);
    free_head_.store(nullptr, std::memory_order_relaxed);
}

MemoryArena::~MemoryArena() {
    if (base_ && capacity_ > 0) {
        munmap(base_, capacity_);
        base_ = nullptr;
        capacity_ = 0;
    }
}

void* MemoryArena::acquire(size_t bytes, size_t alignment) noexcept {
    // Try free-list first: pop a suitable block
    while (true) {
        FreeNode* head = free_head_.load(std::memory_order_acquire);
        if (!head) {
            break;
        }

        if (head->size >= bytes) {
            if (free_head_.compare_exchange_weak(
                    head, head->next,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
#ifndef NDEBUG
                std::memset(head, 0, bytes);
#endif
                return head;
            }
        } else {
            // Too small, skip it — look at next node
            // For simplicity, we fall through to bump allocator
            break;
        }
    }

    // Bump allocator path
    while (true) {
        size_t current = offset_.load(std::memory_order_relaxed);
        size_t aligned = (current + alignment - 1) & ~(alignment - 1);

        if (aligned + bytes > capacity_) {
            return nullptr;
        }

        if (offset_.compare_exchange_weak(
                current, aligned + bytes,
                std::memory_order_release,
                std::memory_order_relaxed)) {
            void* ptr = base_ + aligned;
#ifndef NDEBUG
            std::memset(ptr, 0, bytes);
#endif
            return ptr;
        }
    }
}

void MemoryArena::release(void* ptr, size_t bytes) noexcept {
    if (!ptr) return;

    // Push onto free-list using CAS
    FreeNode* node = static_cast<FreeNode*>(ptr);
    FreeNode* head = free_head_.load(std::memory_order_relaxed);
    node->next = head;
    node->size = bytes;

    while (!free_head_.compare_exchange_weak(
            node->next, node,
            std::memory_order_release,
            std::memory_order_relaxed)) {
        // retry — node->next was reloaded by CAS
    }
}

size_t MemoryArena::used_bytes() const noexcept {
    return offset_.load(std::memory_order_relaxed);
}

size_t MemoryArena::capacity_bytes() const noexcept {
    return capacity_;
}

MemoryArena g_arena;

} // namespace BTQuant
