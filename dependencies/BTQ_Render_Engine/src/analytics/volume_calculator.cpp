#include "analytics/volume_calculator.hpp"
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

namespace BTQuant {
namespace Analytics {

double VolumeCalculator::calculateDelta(const std::vector<Data::TradeData>& trades) {
    double buy_volume = calculateBuyVolume(trades);
    double sell_volume = calculateSellVolume(trades);
    return buy_volume - sell_volume;
}

double VolumeCalculator::calculateDeltaPercent(const std::vector<Data::TradeData>& trades) {
    if (trades.empty()) {
        return 0.0;
    }

    double total_volume = calculateTotalVolume(trades);
    if (total_volume == 0.0) {
        return 0.0;
    }

    double delta = calculateDelta(trades);
    return (delta / total_volume) * 100.0;
}

double VolumeCalculator::calculateBuyVolumePercent(const std::vector<Data::TradeData>& trades) {
    if (trades.empty()) {
        return 0.0;
    }

    double total_volume = calculateTotalVolume(trades);
    if (total_volume == 0.0) {
        return 0.0;
    }

    double buy_volume = calculateBuyVolume(trades);
    return (buy_volume / total_volume) * 100.0;
}

double VolumeCalculator::calculateSellVolumePercent(const std::vector<Data::TradeData>& trades) {
    if (trades.empty()) {
        return 0.0;
    }

    double total_volume = calculateTotalVolume(trades);
    if (total_volume == 0.0) {
        return 0.0;
    }

    double sell_volume = calculateSellVolume(trades);
    return (sell_volume / total_volume) * 100.0;
}

double VolumeCalculator::calculateAverageSize(const std::vector<Data::TradeData>& trades) {
    if (trades.empty()) {
        return 0.0;
    }

    double total_volume = calculateTotalVolume(trades);
    return total_volume / static_cast<double>(trades.size());
}

double VolumeCalculator::calculateAverageBuySize(const std::vector<Data::TradeData>& trades) {
    if (trades.empty()) {
        return 0.0;
    }

    double buy_volume = calculateBuyVolume(trades);
    int buy_count = 0;
    
    for (const auto& trade : trades) {
        if (trade.side == Data::TradeSide::BUY) {
            buy_count++;
        }
    }

    if (buy_count == 0) {
        return 0.0;
    }

    return buy_volume / static_cast<double>(buy_count);
}

double VolumeCalculator::calculateAverageSellSize(const std::vector<Data::TradeData>& trades) {
    if (trades.empty()) {
        return 0.0;
    }

    double sell_volume = calculateSellVolume(trades);
    int sell_count = 0;
    
    for (const auto& trade : trades) {
        if (trade.side == Data::TradeSide::SELL) {
            sell_count++;
        }
    }

    if (sell_count == 0) {
        return 0.0;
    }

    return sell_volume / static_cast<double>(sell_count);
}

float VolumeCalculator::calculateMaxOneTradeVolume(const std::vector<Data::TradeData>& trades) {
    if (trades.empty()) {
        return 0.0f;
    }

    float max_volume = 0.0f;
    for (const auto& trade : trades) {
        if (trade.volume > max_volume) {
            max_volume = trade.volume;
        }
    }

    return max_volume;
}

double VolumeCalculator::calculateFilteredVolume(const std::vector<Data::TradeData>& trades, 
                                                double min_price, 
                                                double max_price) {
    double filtered_volume = 0.0;
    
    for (const auto& trade : trades) {
        if (trade.price >= min_price && trade.price <= max_price) {
            filtered_volume += static_cast<double>(trade.volume);
        }
    }

    return filtered_volume;
}

// Private helper methods implementation
double VolumeCalculator::calculateTotalVolume(const std::vector<Data::TradeData>& trades) {
    double total_volume = 0.0;
    for (const auto& trade : trades) {
        total_volume += static_cast<double>(trade.volume);
    }
    return total_volume;
}

double VolumeCalculator::calculateBuyVolume(const std::vector<Data::TradeData>& trades) {
    double buy_volume = 0.0;
    for (const auto& trade : trades) {
        if (trade.side == Data::TradeSide::BUY) {
            buy_volume += static_cast<double>(trade.volume);
        }
    }
    return buy_volume;
}

double VolumeCalculator::calculateSellVolume(const std::vector<Data::TradeData>& trades) {
    double sell_volume = 0.0;
    for (const auto& trade : trades) {
        if (trade.side == Data::TradeSide::SELL) {
            sell_volume += static_cast<double>(trade.volume);
        }
    }
    return sell_volume;
}

} // namespace Analytics
} // namespace BTQuant