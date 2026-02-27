#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace BTQuant {

// Symbol information structure
struct SymbolInfo {
  uint32_t id;
  uint32_t symbol_id;  // Alias for id for compatibility
  std::string exchange;
  std::string symbol;
  std::string key;  // exchange:symbol
  std::string name;  // Added for workspace manager compatibility
  std::string base_currency;
  std::string quote_currency;
  double min_notional = 0.0;
  double min_qty = 0.0;
  double max_qty = 0.0;
  double step_size = 0.0;
  double tick_size = 0.0;
  std::string status = "active";

  SymbolInfo() = default;
  SymbolInfo(uint32_t i, uint32_t sid, const std::string& ex, const std::string& sym,
             const std::string& k)
      : id(i), symbol_id(sid), exchange(ex), symbol(sym), key(k) {}
};

// Global symbol registry for mapping symbol IDs to names
class SymbolRegistry {
 public:
  static SymbolRegistry& instance();

  bool load_from_file(const std::string& filepath);
  std::optional<SymbolInfo> get_symbol_info(uint32_t symbol_id) const;
  std::optional<uint32_t> get_symbol_id(const std::string& exchange,
                                        const std::string& symbol) const;
  std::optional<SymbolInfo> get_symbol_by_name(const std::string& symbol) const;
  std::vector<SymbolInfo> get_all_symbols() const;
  std::vector<std::string> get_exchanges() const;
  std::vector<SymbolInfo> get_exchange_symbols(const std::string& exchange) const;
  uint32_t register_symbol(const std::string& exchange, const std::string& symbol);
  bool has_symbol(uint32_t id) const;
  bool has_symbol(const std::string& exchange, const std::string& symbol) const;
  bool save_to_file(const std::string& filepath) const;
  void clear();
  bool add_symbol(const SymbolInfo& symbol_info);

 private:
  std::string make_key(const std::string& exchange, const std::string& symbol) const;
  SymbolRegistry() = default;
  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, SymbolInfo> id_to_info_;
  std::unordered_map<std::string, uint32_t> key_to_id_;
  uint32_t next_auto_id_ = 1;
};

}  // namespace BTQuant