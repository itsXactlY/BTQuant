/// @file frame_pacer.cpp
/// @brief Implements the frame pacer for maintaining consistent frame timing.

#include "rendering/frame_pacer.hpp"

#include <thread>

namespace BTQuant {

FramePacer::FramePacer() {
    mark_frame_start();
}

void FramePacer::pace(uint64_t budget_us) {
    using Clock = std::chrono::high_resolution_clock;
    auto now = Clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        now - frame_start_).count();
    if (elapsed < static_cast<int64_t>(budget_us)) {
        // std::this_thread::yield() lets the OS schedule the network/ingestion thread
        std::this_thread::yield();
    }
    frame_start_ = Clock::now();
}

void FramePacer::mark_frame_start() {
    frame_start_ = std::chrono::high_resolution_clock::now();
}

FramePacer g_frame_pacer;

}  // namespace BTQuant