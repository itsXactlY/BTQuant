#pragma once

#include <chrono>
#include <cstdint>

namespace BTQuant {

/**
 * @brief Frame Pacer for maintaining consistent frame timing
 * 
 * Manages frame timing to maintain consistent performance and allow
 * the ingestion thread to run when rendering is ahead of schedule.
 */
class FramePacer {
public:
    FramePacer();

    /**
     * @brief Pace the current frame to maintain consistent timing
     * @param budget_us Budget in microseconds (default 6944μs = ~144fps)
     * 
     * Call after vkQueuePresentKHR returns. If frame completed in < budget_us 
     * microseconds, yields to ingestion thread.
     */
    void pace(uint64_t budget_us = 6944);

    /**
     * @brief Mark the start of a frame timing cycle
     */
    void mark_frame_start();

private:
    std::chrono::high_resolution_clock::time_point frame_start_;
};

// Global frame pacer instance
extern FramePacer g_frame_pacer;

}  // namespace BTQuant