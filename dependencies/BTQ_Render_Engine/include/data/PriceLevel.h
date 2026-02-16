#ifndef PRICE_LEVEL_H
#define PRICE_LEVEL_H

#include <vector>

// Price level for order book data
// alignas(64) prevents cache-line false sharing across CPU cores
struct alignas(64) PriceLevel {
  double price;
  double bidVolume;
  double askVolume;
  int bidOrders;
  int askOrders;
};

#endif  // PRICE_LEVEL_H