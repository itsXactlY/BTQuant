#pragma once

#include <atomic>
#include <cstdint>

namespace BTQuant {

// Atomic crosshair state shared across all panels for synchronized crosshair rendering.
// Writers: ChartPanel mouse hover handler (memory_order_release)
// Readers: DOM, TPO, Footprint panel renderers (memory_order_acquire)
inline std::atomic<double> g_crosshair_price{0.0};
inline std::atomic<int32_t> g_crosshair_symbol_id{-1};

}  // namespace BTQuant
