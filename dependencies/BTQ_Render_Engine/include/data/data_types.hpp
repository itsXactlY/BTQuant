#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <string>

// ============================================================================
// MMT GENESIS - Bare Metal Spine Data Types
// Zero-allocation, lock-free, GPU-compatible POD structures
// ============================================================================
// NOTE: This file provides legacy compatibility types.
// New code should use core_types.hpp which defines types in BTQuant namespace.
// ============================================================================

// Import the new core types - they are the authoritative definitions
#include "core_types.hpp"

// Volume profile level for market profile analysis
struct VolumeProfileLevel {
  double price;
  double total_volume;
  double buy_volume;
  double sell_volume;
};

namespace BTQuant {
namespace Data {

// Legacy TradeData structure for backward compatibility
// Used by existing code that depends on std::string symbol
struct LegacyTradeData {
  std::string symbol;
  uint32_t symbol_id = 0;
  uint64_t timestamp = 0;
  double price = 0.0;
  double size = 0.0;
  bool is_buy = true;
  
  // Convert to new POD TradeData (BTQuant::TradeData from core_types.hpp)
  BTQuant::TradeData toPod() const {
    BTQuant::TradeData pod{};
    pod.timestamp_us = timestamp;
    pod.symbol_id = symbol_id;
    pod.side = is_buy ? BTQuant::TradeSide::BUY : BTQuant::TradeSide::SELL;
    pod.flags = 0;
    pod.price = price;
    pod.volume = static_cast<float>(size);
    pod.exchange_id = 0;
    return pod;
  }
  
  // Convert from new POD TradeData
  static LegacyTradeData fromPod(const BTQuant::TradeData& pod, const std::string& symbol_name = "") {
    LegacyTradeData legacy;
    legacy.symbol = symbol_name;
    legacy.symbol_id = pod.symbol_id;
    legacy.timestamp = pod.timestamp_us;
    legacy.price = pod.price;
    legacy.size = pod.volume;
    legacy.is_buy = pod.is_buy();
    return legacy;
  }
};

} // namespace Data
} // namespace BTQuant