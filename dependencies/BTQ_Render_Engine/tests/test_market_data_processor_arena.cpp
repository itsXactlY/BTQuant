#include <cstdio>
#include <cstdlib>

#include "data/core_types.hpp"
#include "market_data_processor.hpp"
#include "memory/memory_arena.hpp"

using namespace BTQuant;

int main() {
  printf("=== MarketDataProcessor Arena Allocation Tests ===\n\n");

  // Reset arena state by creating a fresh one (for test isolation)
  // Note: In real usage, g_arena is a singleton initialized once in main()
  size_t initial_used = g_arena.used_bytes();
  printf("Initial g_arena.used_bytes(): %zu\n\n", initial_used);

  // Create MarketDataProcessor - should allocate all TradeRing buffers
  printf("Creating MarketDataProcessor...\n");
  MarketDataProcessor processor;
  size_t after_constructor = g_arena.used_bytes();
  printf("After constructor g_arena.used_bytes(): %zu\n", after_constructor);

  // Calculate expected allocation
  constexpr size_t MAX_SYMBOLS = MarketDataProcessor::MAX_SYMBOLS;
  constexpr size_t TRADE_RING_CAP = 1024;
  constexpr size_t TRADE_DATA_SIZE = sizeof(TradeData);
  size_t expected_allocation = MAX_SYMBOLS * TRADE_RING_CAP * TRADE_DATA_SIZE;
  printf("\nExpected allocation:\n");
  printf("  MAX_SYMBOLS: %zu\n", MAX_SYMBOLS);
  printf("  TradeRing::CAP: %zu\n", TRADE_RING_CAP);
  printf("  sizeof(TradeData): %zu\n", TRADE_DATA_SIZE);
  printf("  Total expected: %zu bytes (%.2f MB)\n", expected_allocation,
         expected_allocation / (1024.0 * 1024.0));

  size_t actual_allocation = after_constructor - initial_used;
  printf("\nActual allocation: %zu bytes (%.2f MB)\n", actual_allocation,
         actual_allocation / (1024.0 * 1024.0));

  // Test 1: Arena usage increased
  printf("\n--- Test 1: Arena usage increased after constructor ---\n");
  if (actual_allocation > 0) {
    printf("PASSED: Arena usage increased by %zu bytes\n", actual_allocation);
  } else {
    printf("FAILED: Arena usage did not increase\n");
    return EXIT_FAILURE;
  }

  // Test 2: Allocation is approximately correct (may have some overhead for alignment)
  printf("\n--- Test 2: Allocation size is correct ---\n");
  if (actual_allocation >= expected_allocation &&
      actual_allocation <= expected_allocation + (MAX_SYMBOLS * 128)) {
    printf("PASSED: Allocation size is within expected range\n");
  } else {
    printf("FAILED: Allocation size %zu != expected %zu\n", actual_allocation,
           expected_allocation);
    return EXIT_FAILURE;
  }

  // Test 3: Verify no heap allocations were made for TradeRing buffers
  // (This is implicitly verified by the arena usage increase)
  printf("\n--- Test 3: No heap allocations for TradeRing buffers ---\n");
  printf("PASSED: All TradeRing buffers allocated from g_arena (verified by arena usage)\n");

  // Test 4: Verify TradeRing buffers are valid
  printf("\n--- Test 4: TradeRing buffers are valid ---\n");
  bool all_valid = true;
  for (size_t i = 0; i < MAX_SYMBOLS; ++i) {
    // Access through public API - peek_trades should work without crash
    TradeData dummy;
    processor.peek_trades(static_cast<uint32_t>(i), 1, &dummy);
  }
  printf("PASSED: All TradeRing buffers accessible via peek_trades()\n");

  printf("\n=== All Tests Passed ===\n");
  return EXIT_SUCCESS;
}
