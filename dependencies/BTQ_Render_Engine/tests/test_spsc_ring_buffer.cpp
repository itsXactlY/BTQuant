/**
 * Unit tests for SpscRingBuffer::peek()
 * 
 * Tests verify that peek():
 * - Returns last N items without modifying tail
 * - Works correctly with wrap-around
 * - Is thread-safe for single-reader scenario
 */

#include <threading/lockfree_queue.hpp>
#include <cassert>
#include <cstdio>
#include <cstring>

using namespace BTQuant;

// Simple test structure
struct TestData {
    int32_t id;
    int32_t value;
};

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
        printf("FAILED: %s (%s != %s)\n", msg, #a, #b); \
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

// Test: peek() on empty buffer returns 0
TEST(test_peek_empty) {
    SpscRingBuffer<TestData, 16> buffer;
    TestData out[10];
    
    size_t count = buffer.peek(10, out);
    ASSERT_EQ(count, 0, "Empty buffer peek should return 0");
}

// Test: peek() returns items without consuming them
TEST(test_peek_non_consuming) {
    SpscRingBuffer<TestData, 16> buffer;
    
    // Push 3 items
    TestData items[] = {{1, 100}, {2, 200}, {3, 300}};
    for (const auto& item : items) {
        bool ok = buffer.push(item);
        ASSERT_TRUE(ok, "Push should succeed");
    }
    
    // Peek 3 items
    TestData out[10];
    size_t count = buffer.peek(3, out);
    ASSERT_EQ(count, 3, "Should peek 3 items");
    
    // Verify data
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(out[i].id, items[i].id, "ID should match");
        ASSERT_EQ(out[i].value, items[i].value, "Value should match");
    }
    
    // Verify buffer still has 3 items (peek doesn't consume)
    ASSERT_EQ(buffer.size(), 3, "Size should still be 3 after peek");
    
    // Pop and verify same data
    TestData popped;
    for (int i = 0; i < 3; ++i) {
        bool ok = buffer.pop(popped);
        ASSERT_TRUE(ok, "Pop should succeed");
        ASSERT_EQ(popped.id, items[i].id, "Popped ID should match");
        ASSERT_EQ(popped.value, items[i].value, "Popped value should match");
    }
    
    ASSERT_EQ(buffer.size(), 0, "Buffer should be empty after pops");
}

// Test: peek() with n < available returns n items (most recent)
TEST(test_peek_partial) {
    SpscRingBuffer<TestData, 16> buffer;
    
    // Push 5 items
    TestData items[] = {{1, 100}, {2, 200}, {3, 300}, {4, 400}, {5, 500}};
    for (const auto& item : items) {
        buffer.push(item);
    }
    
    // Peek only 3 items (returns the LAST 3 = most recent = items[2,3,4])
    TestData out[10];
    size_t count = buffer.peek(3, out);
    ASSERT_EQ(count, 3, "Should peek 3 items");
    
    // out[0] = oldest of the returned 3 = items[2], out[2] = most recent = items[4]
    ASSERT_EQ(out[0].id, 3, "First peeked = oldest of last 3");
    ASSERT_EQ(out[1].id, 4, "Second peeked");
    ASSERT_EQ(out[2].id, 5, "Third peeked = most recent of last 3");
}

// Test: peek() with n > available returns all available
TEST(test_peek_more_than_available) {
    SpscRingBuffer<TestData, 16> buffer;
    
    // Push 2 items
    buffer.push({1, 100});
    buffer.push({2, 200});
    
    // Try to peek 10 items
    TestData out[10];
    size_t count = buffer.peek(10, out);
    ASSERT_EQ(count, 2, "Should return only available items");
    ASSERT_EQ(out[0].id, 1, "First item");
    ASSERT_EQ(out[1].id, 2, "Second item");
}

// Test: peek() with wrap-around (ring buffer circular behavior)
TEST(test_peek_wrap_around) {
    SpscRingBuffer<TestData, 8> buffer;  // Small buffer for easy wrap
    
    // Fill and partially drain to cause wrap
    for (int i = 0; i < 6; ++i) {
        buffer.push({i, i * 100});
    }
    
    // Pop 4 items (head moves forward)
    TestData dummy;
    for (int i = 0; i < 4; ++i) {
        buffer.pop(dummy);
    }
    
    // Now push 4 more (will wrap around)
    for (int i = 6; i < 10; ++i) {
        buffer.push({i, i * 100});
    }
    
    // Buffer should have items [4, 5, 6, 7, 8, 9] (6 items)
    ASSERT_EQ(buffer.size(), 6, "Should have 6 items");
    
    // Peek all 6
    TestData out[10];
    size_t count = buffer.peek(6, out);
    ASSERT_EQ(count, 6, "Should peek all 6 items");
    
    // Verify order: out[0] = oldest (4), out[5] = most recent (9)
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ(out[i].id, 4 + i, "Item order should be preserved across wrap");
    }
}

// Test: peek() does not modify tail (thread-safety for independent readers)
TEST(test_peek_does_not_modify_tail) {
    SpscRingBuffer<TestData, 16> buffer;
    
    // Push some items
    for (int i = 0; i < 5; ++i) {
        buffer.push({i, i * 100});
    }
    
    // Get initial tail position (via size)
    size_t size_before = buffer.size();
    
    // Peek multiple times
    TestData out[10];
    for (int j = 0; j < 10; ++j) {
        buffer.peek(3, out);
    }
    
    // Size should be unchanged (tail not modified)
    ASSERT_EQ(buffer.size(), size_before, "Size should be unchanged after multiple peeks");
    
    // Verify we can still pop all items
    TestData popped;
    for (int i = 0; i < 5; ++i) {
        bool ok = buffer.pop(popped);
        ASSERT_TRUE(ok, "Pop should succeed");
        ASSERT_EQ(popped.id, i, "Popped item should match");
    }
}

// Test: peek() with zero count
TEST(test_peek_zero_count) {
    SpscRingBuffer<TestData, 16> buffer;
    
    buffer.push({1, 100});
    buffer.push({2, 200});
    
    TestData out[10];
    size_t count = buffer.peek(0, out);
    ASSERT_EQ(count, 0, "Peek with n=0 should return 0");
    
    // Buffer should still have items
    ASSERT_EQ(buffer.size(), 2, "Buffer should still have 2 items");
}

int main() {
    printf("=== SpscRingBuffer::peek() Unit Tests ===\n\n");
    
    RUN_TEST(test_peek_empty);
    RUN_TEST(test_peek_non_consuming);
    RUN_TEST(test_peek_partial);
    RUN_TEST(test_peek_more_than_available);
    RUN_TEST(test_peek_wrap_around);
    RUN_TEST(test_peek_does_not_modify_tail);
    RUN_TEST(test_peek_zero_count);
    
    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    
    return tests_failed == 0 ? 0 : 1;
}
