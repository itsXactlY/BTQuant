#ifndef VOLUMEDATATYPES_H
#define VOLUMEDATATYPES_H

namespace BTQ {
namespace RenderEngine {
namespace Data {

enum class VolumeDataType {
    Trades,
    BuyTrades,
    SellTrades,
    Volume,
    BuyVolume,
    SellVolume,
    BuyVolumePercent,
    SellVolumePercent,
    BuySellVolume,
    Delta,
    DeltaPercent,
    CumulativeDelta,
    AverageSize,
    AverageBuySize,
    AverageSellSize,
    MaxOneTradeVolume,
    FilteredVolume
};

} // namespace Data
} // namespace RenderEngine
} // namespace BTQ

#endif // VOLUMEDATATYPES_H