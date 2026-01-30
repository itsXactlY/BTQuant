#ifndef FOOTPRINT_CELL_H
#define FOOTPRINT_CELL_H

#include <cstdint>

struct FootprintCell {
    double priceLevel;
    int64_t timeBucket;
    double buyVolume;
    double sellVolume;
    double delta;
    int32_t numBuyTrades;
    int32_t numSellTrades;
    double maxSingleTrade;
};

#endif // FOOTPRINT_CELL_H