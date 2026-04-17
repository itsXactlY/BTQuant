#ifndef BTQ_MEMORY_ARENA_HPP
#define BTQ_MEMORY_ARENA_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace BTQuant {

class MemoryArena {
public:
    explicit MemoryArena(size_t size_bytes = 1ULL << 30);
    ~MemoryArena();

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    [[nodiscard]] void* acquire(size_t bytes, size_t alignment = 64) noexcept;
    void release(void* ptr, size_t bytes) noexcept;

    size_t used_bytes() const noexcept;
    size_t capacity_bytes() const noexcept;

private:
    uint8_t* base_ = nullptr;
    size_t capacity_ = 0;
    std::atomic<size_t> offset_{0};

    struct FreeNode {
        FreeNode* next;
        size_t size;
    };

    std::atomic<FreeNode*> free_head_{nullptr};
};

extern MemoryArena g_arena;

} // namespace BTQuant

#endif // BTQ_MEMORY_ARENA_HPP
