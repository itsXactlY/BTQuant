#pragma once

#include <atomic>

namespace BTQuant {

/**
 * @brief Global synchronization state for crosshair and shared data
 * 
 * Contains atomic variables for crosshair position and other global
 * state that needs to be shared across all panels.
 */
namespace GlobalState {
    inline std::atomic<double> g_crosshair_price{0.0};
    inline std::atomic<int32_t> g_crosshair_symbol_id{-1}; // -1 = no crosshair active
}

using namespace GlobalState;

}  // namespace BTQuant