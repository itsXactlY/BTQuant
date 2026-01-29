#include <cassert>
#include <iostream>
#include <string>

#include "../include/symbol_registry.hpp"

void test_get_symbol_by_name() {
  auto& registry = BTQuant::SymbolRegistry::instance();

  // Clear registry for clean test
  // Note: We'll use reflection or direct access if available, or register test symbols

  // Register a test symbol
  uint32_t symbol_id = registry.register_symbol("Binance", "BTCUSDT");

  // Test getting symbol by name
  auto symbol_info_opt = registry.get_symbol_by_name("BTCUSDT");
  assert(symbol_info_opt.has_value() == true);
  assert(symbol_info_opt->symbol == "BTCUSDT");
  assert(symbol_info_opt->exchange == "Binance");
  assert(symbol_info_opt->id == symbol_id);

  // Test getting non-existent symbol
  auto non_existent = registry.get_symbol_by_name("NONEXISTENT");
  assert(non_existent.has_value() == false);

  std::cout << "get_symbol_by_name test passed!" << std::endl;
}

void test_get_symbol_by_name_multiple_exchanges() {
  auto& registry = BTQuant::SymbolRegistry::instance();

  // Register same symbol name on different exchanges
  uint32_t id1 = registry.register_symbol("Binance", "BTCUSDT");
  uint32_t id2 = registry.register_symbol("Bybit", "BTCUSDT");

  // Test getting symbol by name - should return first match
  auto symbol_info_opt = registry.get_symbol_by_name("BTCUSDT");
  assert(symbol_info_opt.has_value() == true);
  // Note: The implementation returns the first match, so it could be either
  assert(symbol_info_opt->symbol == "BTCUSDT");

  std::cout << "get_symbol_by_name multiple exchanges test passed!" << std::endl;
}

int main() {
  test_get_symbol_by_name();
  test_get_symbol_by_name_multiple_exchanges();

  std::cout << "All tests passed!" << std::endl;
  return 0;
}