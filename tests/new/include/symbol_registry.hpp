#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace BTQuant {

struct SymbolInfo {
  uint32_t id;
  std::string exchange;
  std::string symbol;

  std::string full_symbol() const { return exchange + "/" + symbol; }
};

class SymbolRegistry {
public:
  static SymbolRegistry &instance();

  // Load symbol mappings from JSON file
  bool load_from_file(const std::string &filepath);

  // Register a symbol manually
  uint32_t register_symbol(const std::string &exchange,
                           const std::string &symbol,
                           uint32_t id = 0); // 0 = auto-assign

  // Lookup functions
  std::optional<SymbolInfo> get_symbol_info(uint32_t id) const;
  std::optional<uint32_t> get_symbol_id(const std::string &exchange,
                                        const std::string &symbol) const;

  // Get all symbols for an exchange
  std::vector<SymbolInfo>
  get_exchange_symbols(const std::string &exchange) const;

  // Get all registered exchanges
  std::vector<std::string> get_exchanges() const;

  // Check if symbol exists
  bool has_symbol(uint32_t id) const;
  bool has_symbol(const std::string &exchange, const std::string &symbol) const;

  // Save all symbol mappings to JSON file
  bool save_to_file(const std::string &filepath) const;

  // Get all registered symbols
  std::vector<SymbolInfo> get_all_symbols() const;

private:
  SymbolRegistry() = default;
  ~SymbolRegistry() = default;
  SymbolRegistry(const SymbolRegistry &) = delete;
  SymbolRegistry &operator=(const SymbolRegistry &) = delete;

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, SymbolInfo> id_to_info_;
  std::unordered_map<std::string, uint32_t>
      key_to_id_; // "exchange:symbol" -> id
  uint32_t next_auto_id_ = 10000;

  std::string make_key(const std::string &exchange,
                       const std::string &symbol) const;
};

} // namespace BTQuant