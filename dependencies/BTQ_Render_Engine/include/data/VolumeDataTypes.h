#pragma once

#include <cstdint>

namespace BTQuant {
namespace Data {

// Volume Analysis Type Enum for Footprint and Volume Profile charts
// Based on Quantower data types
enum class VolumeAnalysisType {
  Trades,             // Total number of trades
  BuyTrades,          // Number of buy trades
  SellTrades,         // Number of sell trades
  Volume,             // Total volume (bid + ask)
  BuyVolume,          // Volume of buy trades
  SellVolume,         // Volume of sell trades
  BuyVolumePercent,   // Percentage of buy volume
  SellVolumePercent,  // Percentage of sell volume
  BuySellVolume,      // Difference between buy and sell volume (BuyVolume - SellVolume)
  Delta,              // Net difference between buy and sell volume (BuyVolume - SellVolume)
  DeltaPercent,       // Delta as percentage of total volume
  CumulativeDelta,    // Running sum of delta values
  AverageSize,        // Average trade size
  AverageBuySize,     // Average size of buy trades
  AverageSellSize,    // Average size of sell trades
  MaxOneTradeVolume,  // Maximum volume of a single trade
  FilteredVolume      // Volume filtered by specific criteria
};

  // Alias for backward compatibility
  using VolumeDataType = VolumeAnalysisType;

// Time Aggregation Type Enum for time-based grouping of data
enum class TimeAggregationType {
  T_1MIN,      // 1 minute aggregation
  T_5MIN,      // 5 minute aggregation
  T_15MIN,     // 15 minute aggregation
  T_30MIN,     // 30 minute aggregation
  T_1HOUR,     // 1 hour aggregation
  T_2HOUR,     // 2 hour aggregation
  T_4HOUR,     // 4 hour aggregation
  VOLUME_BASED, // Volume-based aggregation (every N contracts)
  TICK_BASED    // Tick-based aggregation (every N ticks)
};

  // Price Aggregation Type Enum for price-based grouping of data
  enum class PriceAggregationType {
    P_1TICK,      // 1 tick aggregation
    P_5TICKS,     // 5 ticks aggregation
    P_10TICKS,    // 10 ticks aggregation
    P_POINT1_PCT, // 0.1% aggregation
    P_POINT5_PCT, // 0.5% aggregation
    P_1_PCT,      // 1% aggregation
    P_CUSTOM      // Custom value aggregation
  };

}  // namespace Data
}  // namespace BTQuant
