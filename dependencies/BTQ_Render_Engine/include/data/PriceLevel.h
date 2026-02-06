#ifndef PRICE_LEVEL_H
#define PRICE_LEVEL_H

#include <vector>

struct PriceLevel {
  double price;
  double bidVolume;
  double askVolume;
  int bidOrders;
  int askOrders;
  uint64_t timestamp = 0;  // Optional timestamp for individual price levels
};

#endif  // PRICE_LEVEL_H