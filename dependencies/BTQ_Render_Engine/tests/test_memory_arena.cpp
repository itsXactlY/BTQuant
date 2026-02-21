/**
 * Unit tests for MemoryArena
 *
 * Tests verify that:
 * - MemoryArena is constructed with 1GB capacity
 * - used_bytes() starts at 0
 * - acquire() and release() work correctly
 */

#include <memory/memory_arena.hpp>
#include <cassert>
#include <cstdio>
#include <cstring>

using namespace BTQuant;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) void name()
#define RUN_TEST(name) do { \
    printf("Running %s... ", #name); \
    name(); \
    printf("PASSED\n"); \
    tests_passed++; \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("FAILED: %s (%zu != %zu)\n", msg, (size_t)(a), (size_t)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        printf("FAILED: %s\n", msg); \
        tests_failed++; \
        return; \
    } \
} while(0)

// Test: MemoryArena constructed with 1GB capacity
TEST(test_arena_1gb_capacity) {
    MemoryArena arena(1ULL << 30);  // 1GB
    ASSERT_EQ(arena.capacity_bytes(), 1ULL << 30, "Capacity should be 1GB");
}

// Test: g_arena.used_bytes() starts at 0
TEST(test_arena_used_bytes_starts_at_zero) {
    MemoryArena arena;
    ASSERT_EQ(arena.used_bytes(), 0, "used_bytes() should start at 0");
}

// Test: acquire() increases used_bytes
TEST(test_arena_acquire_increases_used) {
    MemoryArena arena;
    
    size_t alloc_size = 1024;
    void* ptr = arena.acquire(alloc_size, 64);
    ASSERT_TRUE(ptr != nullptr, "acquire() should return valid pointer");
    
    // used_bytes should be at least alloc_size (may be more due to alignment)
    size_t used = arena.used_bytes();
    ASSERT_TRUE(used >= alloc_size, "used_bytes should be >= allocated size");
}

// Test: release() returns memory to free-list (used_bytes unchanged, but memory reusable)
TEST(test_arena_release_returns_to_freelist) {
    MemoryArena arena;
    
    size_t alloc_size = 1024;
    void* ptr1 = arena.acquire(alloc_size, 64);
    ASSERT_TRUE(ptr1 != nullptr, "First acquire should succeed");
    
    size_t used_after_first = arena.used_bytes();
    
    // Release the memory
    arena.release(ptr1, alloc_size);
    
    // used_bytes should remain the same (bump pointer doesn't go back)
    ASSERT_EQ(arena.used_bytes(), used_after_first, "used_bytes unchanged after release");
    
    // But we should be able to acquire again (from free-list)
    void* ptr2 = arena.acquire(alloc_size, 64);
    ASSERT_TRUE(ptr2 != nullptr, "Second acquire should succeed after release");
}

// Test: global g_arena singleton
TEST(test_global_arena_singleton) {
    // g_arena should be valid (mmap succeeded)
    ASSERT_TRUE(g_arena.is_valid(), "g_arena should be valid");
    
    // g_arena should start with 0 used bytes
    ASSERT_EQ(g_arena.used_bytes(), 0, "g_arena used_bytes should start at 0");
    
    // g_arena should have 1GB capacity
    ASSERT_EQ(g_arena.capacity_bytes(), 1ULL << 30, "g_arena capacity should be 1GB");
}

// Test: multiple allocations with alignment
TEST(test_arena_multiple_aligned_allocations) {
    MemoryArena arena;
    
    const size_t alignment = 64;
    void* ptrs[10];
    
    for (int i = 0; i < 10; ++i) {
        ptrs[i] = arena.acquire(128, alignment);
        ASSERT_TRUE(ptrs[i] != nullptr, "Allocation should succeed");
        
        // Verify alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptrs[i]);
        ASSERT_TRUE((addr % alignment) == 0, "Pointer should be 64-byte aligned");
    }
    
    // All allocations should have increased used_bytes
    ASSERT_TRUE(arena.used_bytes() > 0, "used_bytes should be > 0 after allocations");
}

int main() {
    printf("=== MemoryArena Unit Tests ===\n\n");

    RUN_TEST(test_arena_1gb_capacity);
    RUN_TEST(test_arena_used_bytes_starts_at_zero);
    RUN_TEST(test_arena_acquire_increases_used);
    RUN_TEST(test_arena_release_returns_to_freelist);
    RUN_TEST(test_global_arena_singleton);
    RUN_TEST(test_arena_multiple_aligned_allocations);

    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
