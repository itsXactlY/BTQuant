#include "symbol_registry.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

// Simple JSON parser (no dependencies)
#include <regex>

namespace BTQuant {

SymbolRegistry &SymbolRegistry::instance() {
  static SymbolRegistry instance;
  return instance;
}

std::string SymbolRegistry::make_key(const std::string &exchange,
                                     const std::string &symbol) const {
  return exchange + ":" + symbol;
}

bool SymbolRegistry::load_from_file(const std::string &filepath) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    std::ifstream file(filepath);
    if (!file.is_open()) {
      std::cerr << "Failed to open symbol mapping file: " << filepath
                << std::endl;
      return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Regex-based JSON parsing (good enough for our use case)
    // 1. Try parsing flat format (used by save_to_file / shared memory)
    // Format: {"id": 123, "exchange": "binance", "symbol": "BTCUSDT"}
    std::regex flat_pattern(
        R"json(\{"id"\s*:\s*(\d+),\s*"exchange"\s*:\s*"([^"]+)",\s*"symbol"\s*:\s*"([^"]+)"\})json");

    std::sregex_iterator flat_begin(content.begin(), content.end(),
                                    flat_pattern);
    std::sregex_iterator flat_end;

    int flat_matches = 0;
    for (auto it = flat_begin; it != flat_end; ++it) {
      uint32_t id = std::stoul((*it)[1].str());
      std::string exchange = (*it)[2].str();
      std::string symbol = (*it)[3].str();

      SymbolInfo info{id, exchange, symbol};
      id_to_info_[id] = info;
      key_to_id_[make_key(exchange, symbol)] = id;
      flat_matches++;
    }

    if (flat_matches > 0) {
      std::cout << "Loaded " << flat_matches
                << " symbol mappings (flat format) from " << filepath
                << std::endl;
      return true;
    }

    // 2. Try parsing hierarchical format (legacy config)
    // Format: "name": "binance" ... "symbols": [{"symbol": "BTCUSDT", "id":
    // 123}, ...]
    std::regex exchange_pattern(R"json("name"\s*:\s*"([^"]+)")json");
    std::regex symbol_pattern(
        R"json(\{"symbol"\s*:\s*"([^"]+)"\s*,\s*"id"\s*:\s*(\d+)\})json");

    std::sregex_iterator exchanges_begin(content.begin(), content.end(),
                                         exchange_pattern);
    std::sregex_iterator exchanges_end;

    int hier_matches = 0;
    for (auto it = exchanges_begin; it != exchanges_end; ++it) {
      std::string exchange = (*it)[1].str();

      // Find symbols for this exchange
      size_t exchange_pos = it->position();
      size_t next_exchange_pos =
          content.find(R"json("name")json", exchange_pos + 1);
      if (next_exchange_pos == std::string::npos) {
        next_exchange_pos = content.length();
      }

      std::string exchange_section =
          content.substr(exchange_pos, next_exchange_pos - exchange_pos);

      std::sregex_iterator symbols_begin(
          exchange_section.begin(), exchange_section.end(), symbol_pattern);
      std::sregex_iterator symbols_end;

      for (auto sym_it = symbols_begin; sym_it != symbols_end; ++sym_it) {
        std::string symbol = (*sym_it)[1].str();
        uint32_t id = std::stoul((*sym_it)[2].str());

        SymbolInfo info{id, exchange, symbol};
        id_to_info_[id] = info;
        key_to_id_[make_key(exchange, symbol)] = id;
        hier_matches++;
      }
    }

    if (hier_matches > 0) {
      std::cout << "Loaded " << hier_matches
                << " symbol mappings (hierarchical format) from " << filepath
                << std::endl;
      return true;
    }

    std::cerr << "Warning: No symbols found in " << filepath
              << " (checked flat and hierarchical formats)" << std::endl;
    return false;

  } catch (const std::exception &e) {
    std::cerr << "Error loading symbol mappings: " << e.what() << std::endl;
    return false;
  }
}

uint32_t SymbolRegistry::register_symbol(const std::string &exchange,
                                         const std::string &symbol,
                                         uint32_t id) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string key = make_key(exchange, symbol);

  // Check if already registered
  auto it = key_to_id_.find(key);
  if (it != key_to_id_.end()) {
    return it->second;
  }

  // Use provided ID or auto-assign
  if (id == 0) {
    id = next_auto_id_++;
  }

  SymbolInfo info{id, exchange, symbol};
  id_to_info_[id] = info;
  key_to_id_[key] = id;

  return id;
}

std::optional<SymbolInfo> SymbolRegistry::get_symbol_info(uint32_t id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = id_to_info_.find(id);
  if (it != id_to_info_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<uint32_t>
SymbolRegistry::get_symbol_id(const std::string &exchange,
                              const std::string &symbol) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string key = make_key(exchange, symbol);
  auto it = key_to_id_.find(key);
  if (it != key_to_id_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::vector<SymbolInfo>
SymbolRegistry::get_exchange_symbols(const std::string &exchange) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<SymbolInfo> result;
  for (const auto &[id, info] : id_to_info_) {
    if (info.exchange == exchange) {
      result.push_back(info);
    }
  }
  return result;
}

std::vector<std::string> SymbolRegistry::get_exchanges() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::string> result;
  for (const auto &[id, info] : id_to_info_) {
    if (std::find(result.begin(), result.end(), info.exchange) ==
        result.end()) {
      result.push_back(info.exchange);
    }
  }
  return result;
}

bool SymbolRegistry::has_symbol(uint32_t id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return id_to_info_.find(id) != id_to_info_.end();
}

bool SymbolRegistry::has_symbol(const std::string &exchange,
                                const std::string &symbol) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = make_key(exchange, symbol);
  return key_to_id_.find(key) != key_to_id_.end();
}

bool SymbolRegistry::save_to_file(const std::string &filepath) const {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    std::ofstream file(filepath);
    if (!file.is_open()) {
      std::cerr << "Failed to open file for writing: " << filepath << std::endl;
      return false;
    }

    // Write JSON format compatible with Python reader
    file << "{\n  \"symbols\": [\n";

    bool first = true;
    for (const auto &[id, info] : id_to_info_) {
      if (!first)
        file << ",\n";
      first = false;
      file << "    {\"id\": " << id << ", \"exchange\": \"" << info.exchange
           << "\", \"symbol\": \"" << info.symbol << "\"}";
    }

    file << "\n  ]\n}\n";
    file.close();

    std::cout << "Saved " << id_to_info_.size() << " symbol mappings to "
              << filepath << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "Error saving symbol mappings: " << e.what() << std::endl;
    return false;
  }
}

std::vector<SymbolInfo> SymbolRegistry::get_all_symbols() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<SymbolInfo> result;
  result.reserve(id_to_info_.size());

  for (const auto &[id, info] : id_to_info_) {
    result.push_back(info);
  }

  return result;
}

} // namespace BTQuant
