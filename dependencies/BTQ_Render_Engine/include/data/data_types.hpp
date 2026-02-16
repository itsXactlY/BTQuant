#pragma once

#include <vector>
#include <cstdint>

#include "core_types.hpp"

// Price level for order book data (alias for backward compatibility)
using PriceLevel = BTQuant::PriceLevel;

namespace BTQuant {
namespace RenderEngine {

// Type aliases for backward compatibility
using TimeFrame = BTQuant::TimeFrame;
using OHLCVCandle = BTQuant::OHLCVCandle;
using VolumeProfileLevel = BTQuant::VolumeProfileLevel;

} // namespace RenderEngine
} // namespace BTQuant