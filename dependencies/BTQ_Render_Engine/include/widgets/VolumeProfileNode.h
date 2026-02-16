#ifndef VOLUME_PROFILE_NODE_H
#define VOLUME_PROFILE_NODE_H

#include <cstdint>

// Volume profile node for volume analysis
// alignas(64) prevents cache-line false sharing across CPU cores
struct alignas(64) VolumeProfileNode {
  double priceLevel;
  double totalVolume;
  double buyVolume;
  double sellVolume;
  double delta;
  uint32_t numTrades;
};

#endif  // VOLUME_PROFILE_NODE_H