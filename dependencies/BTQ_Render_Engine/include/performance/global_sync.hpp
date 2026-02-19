#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>
#include <array>
#include <algorithm>

// ============================================================================
// GLOBAL SYNCHRONIZATION PRIMITIVES
// ============================================================================

// Atomic crosshair price - synchronized across all Charts, DOMs, and TPOs
// C++26: lock-free atomic for cross-thread communication
extern std::atomic<double> g_crosshair_price;

// Atomic price update timestamp for crosshair
extern std::atomic<uint64_t> g_crosshair_timestamp;

// ============================================================================
// FRAME TELEMETRY SYSTEM - Bare-Metal Performance Monitoring
// ============================================================================

namespace BTQuant {
namespace Telemetry {

// TSC (Time Stamp Counter) intrinsics for microsecond-precision latency measurement
#if defined(__x86_64__) || defined(_M_X64)
    #include <x86intrin.h>
    #define READ_TSC() __rdtsc()
    #define TSC_TICKS_PER_SECOND 3'400'000'000.0  // Typical 3.4 GHz
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define READ_TSC() 0  // ARM doesn't have direct TSC, use chrono instead
    #define TSC_TICKS_PER_SECOND 3'000'000'000.0  // ARM CPUs typically 3.0 GHz
#else
    #define READ_TSC() 0
    #define TSC_TICKS_PER_SECOND 1'000'000'000.0
#endif

// Frame telemetry statistics
struct FrameTelemetry {
    std::atomic<uint64_t> network_ingress_tsc{0};      // TSC when network packet arrived
    std::atomic<uint64_t> processing_start_tsc{0};    // TSC when processing started
    std::atomic<uint64_t> processing_end_tsc{0};       // TSC when processing ended
    std::atomic<uint64_t> render_start_tsc{0};        // TSC when rendering started
    std::atomic<uint64_t> render_complete_tsc{0};     // TSC when render completed
    
    // Calculated latencies (in microseconds)
    std::atomic<double> network_to_processing_us{0};
    std::atomic<double> processing_latency_us{0};
    std::atomic<double> render_latency_us{0};
    std::atomic<double> total_latency_us{0};
    
    // Frame time statistics
    std::atomic<double> frame_time_ms{0};
    std::atomic<double> frame_time_p50_ms{0};
    std::atomic<double> frame_time_p99_ms{0};
    
    // Alert flag for 99th percentile frame exceeding 10ms
    std::atomic<bool> frame_time_alert{false};
};

// Global frame telemetry instance
inline FrameTelemetry& get_frame_telemetry() {
    static FrameTelemetry instance;
    return instance;
}

// Record network ingress timestamp
inline void record_network_ingress() {
    auto& tel = get_frame_telemetry();
    tel.network_ingress_tsc.store(READ_TSC(), std::memory_order_release);
}

// Record processing start
inline void record_processing_start() {
    auto& tel = get_frame_telemetry();
    tel.processing_start_tsc.store(READ_TSC(), std::memory_order_release);
}

// Record processing complete and calculate latencies
inline void record_processing_complete() {
    auto& tel = get_frame_telemetry();
    uint64_t now = READ_TSC();
    tel.processing_end_tsc.store(now, std::memory_order_release);
    
    uint64_t ingress = tel.network_ingress_tsc.load(std::memory_order_acquire);
    uint64_t proc_start = tel.processing_start_tsc.load(std::memory_order_acquire);
    
    if (ingress > 0 && proc_start > 0) {
        double n2p_us = static_cast<double>(proc_start - ingress) / (TSC_TICKS_PER_SECOND / 1'000'000.0);
        double proc_us = static_cast<double>(now - proc_start) / (TSC_TICKS_PER_SECOND / 1'000'000.0);
        
        tel.network_to_processing_us.store(n2p_us, std::memory_order_release);
        tel.processing_latency_us.store(proc_us, std::memory_order_release);
    }
}

// Record render start
inline void record_render_start() {
    auto& tel = get_frame_telemetry();
    tel.render_start_tsc.store(READ_TSC(), std::memory_order_release);
}

// Record render complete and calculate total latency
inline void record_render_complete() {
    auto& tel = get_frame_telemetry();
    uint64_t now = READ_TSC();
    tel.render_complete_tsc.store(now, std::memory_order_release);
    
    uint64_t proc_end = tel.processing_end_tsc.load(std::memory_order_acquire);
    uint64_t rend_start = tel.render_start_tsc.load(std::memory_order_acquire);
    
    if (proc_end > 0 && rend_start > proc_end) {
        double rend_us = static_cast<double>(now - rend_start) / (TSC_TICKS_PER_SECOND / 1'000'000.0);
        tel.render_latency_us.store(rend_us, std::memory_order_release);
        
        double total = tel.network_to_processing_us.load(std::memory_order_acquire) +
                      tel.processing_latency_us.load(std::memory_order_acquire) +
                      rend_us;
        tel.total_latency_us.store(total, std::memory_order_release);
    }
}

// Frame time history for percentile calculations
class FrameTimeHistory {
public:
    static constexpr size_t HISTORY_SIZE = 1000;
    
    FrameTimeHistory() : history_{}, index_{0}, count_{0} {}
    
    void record(double frame_time_ms) {
        history_[index_] = frame_time_ms;
        index_ = (index_ + 1) % HISTORY_SIZE;
        if (count_ < HISTORY_SIZE) count_++;
    }
    
    double percentile(double p) const {
        if (count_ == 0) return 0.0;
        
        std::array<double, HISTORY_SIZE> sorted{};
        for (size_t i = 0; i < count_; ++i) {
            sorted[i] = history_[i];
        }
        std::sort(sorted.begin(), sorted.begin() + count_);
        
        size_t idx = static_cast<size_t>((p / 100.0) * (count_ - 1));
        return sorted[idx];
    }
    
private:
    std::array<double, HISTORY_SIZE> history_;
    size_t index_;
    size_t count_;
};

// Global frame time history
inline FrameTimeHistory& get_frame_time_history() {
    static FrameTimeHistory instance;
    return instance;
}

// Update frame telemetry with current frame time
inline void update_frame_telemetry(double frame_time_ms) {
    auto& tel = get_frame_telemetry();
    auto& history = get_frame_time_history();
    
    tel.frame_time_ms.store(frame_time_ms, std::memory_order_release);
    history.record(frame_time_ms);
    
    // Calculate percentiles
    tel.frame_time_p50_ms.store(history.percentile(50.0), std::memory_order_release);
    tel.frame_time_p99_ms.store(history.percentile(99.0), std::memory_order_release);
    
    // Alert if 99th percentile exceeds 10ms
    bool alert = history.percentile(99.0) > 10.0;
    tel.frame_time_alert.store(alert, std::memory_order_release);
}

// Check if frame time alert is active
inline bool is_frame_alert_active() {
    return get_frame_telemetry().frame_time_alert.load(std::memory_order_acquire);
}

} // namespace Telemetry
} // namespace BTQuant

// ============================================================================
// C++26: LOCK-FREE CROSSHAIR SYNCHRONIZATION
// ============================================================================

namespace BTQuant {

// Set crosshair price from any thread (producer)
inline void set_crosshair_price(double price) {
    g_crosshair_price.store(price, std::memory_order_release);
    g_crosshair_timestamp.store(
        std::chrono::high_resolution_clock::now().time_since_epoch().count(),
        std::memory_order_release
    );
}

// Get crosshair price (consumer)
inline double get_crosshair_price() {
    return g_crosshair_price.load(std::memory_order_acquire);
}

// Try to set crosshair price (lock-free CAS)
inline bool try_set_crosshair_price(double expected, double desired) {
    return g_crosshair_price.compare_exchange_strong(expected, desired,
        std::memory_order_release, std::memory_order_relaxed);
}

} // namespace BTQuant
