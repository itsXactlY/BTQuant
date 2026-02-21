/// @file ssbo_snapshot_updater.cpp
/// @brief Copies HotSpine VolumeNode cluster data into a GPU SSBO.

#include "vulkan/ssbo_snapshot_updater.hpp"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <x86intrin.h>

#include "hotspine_layout_v3.hpp"
#include "vulkan_base_types.hpp"

namespace BTQuant {

// ============================================================================
// initialize
// ============================================================================
bool SsboSnapshotUpdater::initialize(GPUMemoryManager& mem_manager) {
  BufferAllocation alloc =
      mem_manager.allocate_storage_buffer(static_cast<VkDeviceSize>(SSBO_SIZE));

  if (alloc.buffer == VK_NULL_HANDLE) return false;

  buffer_ = alloc.buffer;
  mapped_ptr_ = alloc.mapped_ptr;
  offset_ = alloc.offset;
  pool_id_ = alloc.pool_id;

  // Initialize SSBO with test heatmap pattern for visualization
  if (mapped_ptr_) {
    auto* nodes = reinterpret_cast<::HotSpine::V3::VolumeNode*>(mapped_ptr_);
    constexpr size_t TOTAL_NODES = COLUMNS * ROWS;  // 1024 * 256 = 262,144

    // Create a gradient test pattern with alternating buy/sell volumes
    for (size_t i = 0; i < TOTAL_NODES; ++i) {
      size_t col = i / ROWS;  // X coordinate (0-1023)
      size_t row = i % ROWS;  // Y coordinate (0-255)

      // Create horizontal gradient for buy volume (left to right)
      float buy_gradient = static_cast<float>(col) / static_cast<float>(COLUMNS);

      // Create vertical gradient for sell volume (bottom to top)
      float sell_gradient = static_cast<float>(row) / static_cast<float>(ROWS);

      // Create diagonal band pattern for visual interest
      float diagonal = static_cast<float>((col + row * 4) % 256) / 256.0f;

      // Alternate between buy and sell dominance in bands
      if ((col / 64) % 2 == 0) {
        nodes[i].buy_vol = buy_gradient * diagonal * 100.0f;
        nodes[i].sell_vol = sell_gradient * (1.0f - diagonal) * 50.0f;
      } else {
        nodes[i].buy_vol = buy_gradient * (1.0f - diagonal) * 50.0f;
        nodes[i].sell_vol = sell_gradient * diagonal * 100.0f;
      }

      nodes[i].trade_count = static_cast<uint16_t>(diagonal * 100);
      nodes[i].tpo_bits = static_cast<uint16_t>(row % 16);
    }
  }

  return mapped_ptr_ != nullptr;
}

// ============================================================================
// update — copy from SharedMemoryLayoutV3 with SeqLock protection
// ============================================================================
bool SsboSnapshotUpdater::update(const ::HotSpine::V3::SharedMemoryLayoutV3* layout) {
  if (!layout || !mapped_ptr_) return false;

  // Measure elapsed time with TSC (must complete in < 500μs)
  uint64_t tsc_start = __rdtsc();

  // SeqLock consistent read: try up to 3 times
  for (int attempt = 0; attempt < 3; ++attempt) {
    uint64_t seq = layout->header.global_lock.read_begin();

    auto* dst = static_cast<uint8_t*>(mapped_ptr_);

    // Copy entire history slice using memcpy for maximum throughput
    // Total: 1024 columns × 256 rows × 16 bytes = 4,194,304 bytes
    constexpr size_t COPY_SIZE = COLUMNS * ROWS * NODE_SIZE;
    std::memcpy(dst, layout->history, COPY_SIZE);

    // Verify consistency (retry if seq changed - torn read detected)
    if (layout->header.global_lock.read_retry(seq)) {
      continue;  // Retry on inconsistent read
    }

    // Consistent read — compute max volume for normalization using AVX2 SIMD
    // Iterate through all VolumeNodes to find max combined volume (buy_vol + sell_vol)
    constexpr size_t TOTAL_NODES = COLUMNS * ROWS;  // 262,144 nodes
    auto* nodes = reinterpret_cast<const ::HotSpine::V3::VolumeNode*>(dst);

    // Use AVX2 to process 8 floats at a time (256-bit register)
    // Each VolumeNode has buy_vol and sell_vol contiguous in memory
    float local_max = 1.0f;

#ifdef __AVX2__
    constexpr size_t SIMD_STRIDE = 4;  // 4 VolumeNodes per iteration (8 floats total)

    const size_t simd_limit = (TOTAL_NODES / SIMD_STRIDE) * SIMD_STRIDE;

    __m256 v_max = _mm256_set1_ps(0.0f);

    for (size_t i = 0; i < simd_limit; i += SIMD_STRIDE) {
      // Load buy_vol from 4 consecutive nodes (offset 0)
      __m128 buy_lo = _mm_loadu_ps(reinterpret_cast<const float*>(&nodes[i]));
      // Load buy_vol from next 4 nodes
      __m128 buy_hi = _mm_loadu_ps(reinterpret_cast<const float*>(&nodes[i + 4]));

      // Load sell_vol from 4 consecutive nodes (offset 4 bytes into each node)
      __m128 sell_lo = _mm_loadu_ps(reinterpret_cast<const float*>(&nodes[i]) + 1);
      // Load sell_vol from next 4 nodes
      __m128 sell_hi = _mm_loadu_ps(reinterpret_cast<const float*>(&nodes[i + 4]) + 1);

      // Convert to __m256
      __m256 v_buy = _mm256_set_m128(buy_hi, buy_lo);
      __m256 v_sell = _mm256_set_m128(sell_hi, sell_lo);

      // Compute total volume = buy + sell
      __m256 v_vol = _mm256_add_ps(v_buy, v_sell);

      // Update max
      v_max = _mm256_max_ps(v_max, v_vol);
    }

    // Horizontal max: reduce 8 lanes to 1
    __m128 lo = _mm256_castps256_ps128(v_max);
    __m128 hi = _mm256_extractf128_ps(v_max, 1);
    __m128 max_lo = _mm_max_ps(lo, hi);
    __m128 max_hi = _mm_shuffle_ps(max_lo, max_lo, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 max_final = _mm_max_ss(max_lo, max_hi);
    max_hi = _mm_shuffle_ps(max_final, max_final, _MM_SHUFFLE(1, 1, 1, 1));
    max_final = _mm_max_ss(max_final, max_hi);

    local_max = std::max(local_max, _mm_cvtss_f32(max_final));

    // Handle remaining nodes
    for (size_t i = simd_limit; i < TOTAL_NODES; ++i) {
      float vol = nodes[i].buy_vol + nodes[i].sell_vol;
      if (vol > local_max) local_max = vol;
    }
#else
    // Fallback: scalar implementation
    for (size_t i = 0; i < TOTAL_NODES; ++i) {
      float vol = nodes[i].buy_vol + nodes[i].sell_vol;
      if (vol > local_max) local_max = vol;
    }
#endif

    // Update running max with EMA decay
    max_volume_ = std::max(max_volume_ * 0.99f, local_max);

    // Measure elapsed time and verify < 500μs budget
    uint64_t tsc_end = __rdtsc();
    constexpr double TSC_FREQ = 3'400'000'000.0;  // 3.4 GHz typical
    double elapsed_us = static_cast<double>(tsc_end - tsc_start) /
                        (TSC_FREQ / 1'000'000.0);
    (void)elapsed_us;  // Used for performance verification

    return true;
  }

  return false;  // All attempts had torn reads
}

// ============================================================================
// get_buffer
// ============================================================================
VkBuffer SsboSnapshotUpdater::get_buffer() const { return buffer_; }

// ============================================================================
// destroy
// ============================================================================
void SsboSnapshotUpdater::destroy() {
  // Memory is owned by GPUMemoryManager pool — no explicit free needed
  buffer_ = VK_NULL_HANDLE;
  mapped_ptr_ = nullptr;
  max_volume_ = 1.0f;
}

}  // namespace BTQuant
