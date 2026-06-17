// Static assertions validating the spec-compliant TradeData layout.
// Include this header from one TU to force evaluation.

#ifndef BTQ_TEST_TRADE_DATA_STATIC_ASSERTS_HPP
#define BTQ_TEST_TRADE_DATA_STATIC_ASSERTS_HPP

#include <type_traits>
#include "TradeData.h"

static_assert(sizeof(BTQuant::Data::TradeData) == 64,
              "TradeData must be exactly 64 bytes (alignas(64))");
static_assert(std::is_trivial_v<BTQuant::Data::TradeData>,
              "TradeData must be trivial for hot-path memcpy/atomic safety");
static_assert(std::is_standard_layout_v<BTQuant::Data::TradeData>,
              "TradeData must be standard-layout for ABI stability");
static_assert(alignof(BTQuant::Data::TradeData) == 64,
              "TradeData must be aligned to 64 bytes");
static_assert(std::is_trivially_copyable_v<BTQuant::Data::TradeData>,
              "TradeData must be trivially copyable for zero-copy SPSC use");

// Backwards-compat: legacy `timestamp` field is a union alias for
// `timestamp_us`. They must share storage.
static_assert(offsetof(BTQuant::Data::TradeData, ts) == 16,
              "TradeData timestamp field must be at offset 16");
static_assert(sizeof(BTQuant::Data::TimestampAlias) == 8,
              "TimestampAlias must be 8 bytes (uint64_t union)");

#endif // BTQ_TEST_TRADE_DATA_STATIC_ASSERTS_HPP
