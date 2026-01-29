#ifndef PRICE_LEVEL_H
#define PRICE_LEVEL_H

#include <vector>

struct PriceLevel {
    double price;
    double bidVolume;
    double askVolume;
    int bidOrders;
    int askOrders;
};

#endif // PRICE_LEVEL_H