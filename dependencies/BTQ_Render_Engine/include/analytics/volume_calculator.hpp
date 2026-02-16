#pragma once

#include <limits>
#include <vector>

#include "../data/TradeData.h"

namespace BTQuant {
namespace Analytics {

/**
 * @brief Utility class for calculating various volume-based metrics
 */
class VolumeCalculator {
 public:
  /**
   * @brief Calculate the delta (difference between buy and sell volumes)
   * @param trades Vector of trade data
   * @return Delta value (buy volume - sell volume)
   */
  static double calculateDelta(const std::vector<Data::TradeData>& trades);

  /**
   * @brief Calculate the delta as a percentage of total volume
   * @param trades Vector of trade data
   * @return Delta percentage ((buy volume - sell volume) / total volume * 100)
   */
  static double calculateDeltaPercent(const std::vector<Data::TradeData>& trades);

  /**
   * @brief Calculate the percentage of buy volume relative to total volume
   * @param trades Vector of trade data
   * @return Buy volume percentage (buy volume / total volume * 100)
   */
  static double calculateBuyVolumePercent(const std::vector<Data::TradeData>& trades);

  /**
   * @brief Calculate the percentage of sell volume relative to total volume
   * @param trades Vector of trade data
   * @return Sell volume percentage (sell volume / total volume * 100)
   */
  static double calculateSellVolumePercent(const std::vector<Data::TradeData>& trades);

  /**
   * @brief Calculate the average trade size
   * @param trades Vector of trade data
   * @return Average trade size
   */
  static double calculateAverageSize(const std::vector<Data::TradeData>& trades);

  /**
   * @brief Calculate the average size of buy trades
   * @param trades Vector of trade data
   * @return Average buy trade size
   */
  static double calculateAverageBuySize(const std::vector<Data::TradeData>& trades);

  /**
   * @brief Calculate the average size of sell trades
   * @param trades Vector of trade data
   * @return Average sell trade size
   */
  static double calculateAverageSellSize(const std::vector<Data::TradeData>& trades);

  /**
   * @brief Calculate the maximum volume of a single trade
   * @param trades Vector of trade data
   * @return Maximum volume of a single trade
   */
  static float calculateMaxOneTradeVolume(const std::vector<Data::TradeData>& trades);

  /**
   * @brief Calculate volume filtered by specific criteria
   * @param trades Vector of trade data
   * @param min_price Minimum price threshold (inclusive)
   * @param max_price Maximum price threshold (inclusive)
   * @return Filtered volume
   */
  static double calculateFilteredVolume(const std::vector<Data::TradeData>& trades,
                                        double min_price = 0.0,
                                        double max_price = std::numeric_limits<double>::max());

  /**
   * @brief Structure representing a single price level in the volume profile
   */
  struct VolumeProfileNode {
    double price;
    double buys;
    double sells;
    double total;
  };

  /**
   * @brief Calculate the volume profile for a specific visible time range
   * @param trades Vector of trade data
   * @param start_time Start of the time range (timestamp)
   * @param end_time End of the time range (timestamp)
   * @param tick_size Price tick size for aggregation
   * @return Vector of volume profile nodes
   */
  static std::vector<VolumeProfileNode> get_visible_volume_profile(
      const std::vector<Data::TradeData>& trades, uint64_t start_time, uint64_t end_time,
      double tick_size);

 private:
  // Private helper methods
  static double calculateTotalVolume(const std::vector<Data::TradeData>& trades);
  static double calculateBuyVolume(const std::vector<Data::TradeData>& trades);
  static double calculateSellVolume(const std::vector<Data::TradeData>& trades);
};

}  // namespace Analytics
}  // namespace BTQuant