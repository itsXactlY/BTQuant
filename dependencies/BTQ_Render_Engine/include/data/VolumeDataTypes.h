#pragma once

#include <cstdint>

namespace BTQuant {
namespace Data {

// Volume Data Type Enum for Footprint and Volume Profile charts
// Based on Quantower data types
enum class VolumeDataType {
    Trades,              // Total number of trades
    BuyTrades,           // Number of buy trades
    SellTrades,          // Number of sell trades
    Volume,              // Total volume (bid + ask)
    BuyVolume,           // Volume of buy trades
    SellVolume,          // Volume of sell trades
    BuyVolumePercent,    // Percentage of buy volume
    SellVolumePercent,   // Percentage of sell volume
    BuySellVolume,       // Difference between buy and sell volume (BuyVolume - SellVolume)
    Delta,               // Net difference between buy and sell volume (BuyVolume - SellVolume)
    DeltaPercent,        // Delta as percentage of total volume
    CumulativeDelta,     // Running sum of delta values
    AverageSize,         // Average trade size
    AverageBuySize,      // Average size of buy trades
    AverageSellSize,     // Average size of sell trades
    MaxOneTradeVolume,   // Maximum volume of a single trade
    FilteredVolume       // Volume filtered by specific criteria
};

} // namespace Data
} // namespace BTQuant