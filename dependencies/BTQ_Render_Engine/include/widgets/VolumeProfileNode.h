#ifndef VOLUME_PROFILE_NODE_H
#define VOLUME_PROFILE_NODE_H

#include <cstdint>

struct VolumeProfileNode {
  double priceLevel;
  double totalVolume;
  double buyVolume;
  double sellVolume;
  double delta;
  uint32_t numTrades;
};

#endif  // VOLUME_PROFILE_NODE_H