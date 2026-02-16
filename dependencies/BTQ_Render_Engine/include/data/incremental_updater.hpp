#pragma once

#include <atomic>

namespace BTQuant {
namespace RenderEngine {

// Forward declarations
struct SymbolAnalytics;
struct TradeData;

/**
 * @brief Atomic dirty flag for heatmap compute shader dispatch
 * Set when incremental updates signal a state change
 */
class HeatmapDirtyFlag {
 public:
  /**
   * @brief Mark the heatmap data as dirty (state changed)
   */
  void set() noexcept { dirty_.store(true, std::memory_order_release); }

  /**
   * @brief Check if the flag is dirty without consuming it
   * @return true if dirty, false otherwise
   */
  [[nodiscard]] bool isSet() const noexcept {
    return dirty_.load(std::memory_order_acquire);
  }

  /**
   * @brief Check and consume the dirty flag (atomic exchange)
   * @return true if was dirty, false if already clean
   */
  [[nodiscard]] bool consume() noexcept {
    return dirty_.exchange(false, std::memory_order_acq_rel);
  }

  /**
   * @brief Reset the dirty flag to clean state
   */
  void reset() noexcept { dirty_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> dirty_{false};
};

/**
 * @brief Process a single trade incrementally
 * Only updates affected analytics without recalculating everything
 * @param symbol_data The symbol analytics to update
 * @param trade The trade data to process
 * @param dirty_flag Optional dirty flag to set on state change
 */
void processTradeIncrementally(SymbolAnalytics& symbol_data, const TradeData& trade,
                               HeatmapDirtyFlag* dirty_flag = nullptr);

}  // namespace RenderEngine
}  // namespace BTQuant