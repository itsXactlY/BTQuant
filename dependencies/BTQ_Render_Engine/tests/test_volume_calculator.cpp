#include "../include/analytics/volume_calculator.hpp"
#include "../include/data/TradeData.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

using namespace BTQuant::Analytics;
using namespace BTQuant::Data;

void testCalculateDelta() {
    std::cout << "Testing calculateDelta..." << std::endl;
    
    // Test case 1: Mixed trades
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 100.0, 10.0f, TradeSide::BUY, 1, 0));
    trades1.push_back(TradeData(2, 101.0, 5.0f, TradeSide::SELL, 1, 0));
    trades1.push_back(TradeData(3, 102.0, 15.0f, TradeSide::BUY, 1, 0));
    
    double delta = VolumeCalculator::calculateDelta(trades1);
    assert(delta == 20.0); // (10 + 15) - 5 = 20
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    double delta_empty = VolumeCalculator::calculateDelta(empty_trades);
    assert(delta_empty == 0.0);
    
    // Test case 3: Only buy trades
    std::vector<TradeData> buy_only;
    buy_only.push_back(TradeData(1, 100.0, 10.0f, TradeSide::BUY, 1, 0));
    buy_only.push_back(TradeData(2, 101.0, 20.0f, TradeSide::BUY, 1, 0));
    double delta_buy_only = VolumeCalculator::calculateDelta(buy_only);
    assert(delta_buy_only == 30.0); // 10 + 20 = 30
    
    // Test case 4: Only sell trades
    std::vector<TradeData> sell_only;
    sell_only.push_back(TradeData(1, 100.0, 10.0f, TradeSide::SELL, 1, 0));
    sell_only.push_back(TradeData(2, 101.0, 5.0f, TradeSide::SELL, 1, 0));
    double delta_sell_only = VolumeCalculator::calculateDelta(sell_only);
    assert(delta_sell_only == -15.0); // 0 - (10 + 5) = -15
    
    std::cout << "calculateDelta tests PASSED!" << std::endl;
}

void testCalculateDeltaPercent() {
    std::cout << "Testing calculateDeltaPercent..." << std::endl;
    
    // Test case 1: Mixed trades
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 100.0, 10.0f, TradeSide::BUY, 1, 0));
    trades1.push_back(TradeData(2, 101.0, 5.0f, TradeSide::SELL, 1, 0));
    
    double delta_percent = VolumeCalculator::calculateDeltaPercent(trades1);
    // Delta = 5, Total = 15, Percent = (5/15)*100 = 33.33%
    assert(std::abs(delta_percent - 33.33) < 0.01);
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    double delta_percent_empty = VolumeCalculator::calculateDeltaPercent(empty_trades);
    assert(delta_percent_empty == 0.0);
    
    // Test case 3: Zero total volume
    std::vector<TradeData> zero_vol;
    // All volumes are 0
    zero_vol.push_back(TradeData(1, 100.0, 0.0f, TradeSide::BUY, 1, 0));
    zero_vol.push_back(TradeData(2, 101.0, 0.0f, TradeSide::SELL, 1, 0));
    double delta_percent_zero = VolumeCalculator::calculateDeltaPercent(zero_vol);
    assert(delta_percent_zero == 0.0);
    
    std::cout << "calculateDeltaPercent tests PASSED!" << std::endl;
}

void testCalculateBuyVolumePercent() {
    std::cout << "Testing calculateBuyVolumePercent..." << std::endl;
    
    // Test case 1: Mixed trades
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 100.0, 20.0f, TradeSide::BUY, 1, 0));
    trades1.push_back(TradeData(2, 101.0, 30.0f, TradeSide::SELL, 1, 0));
    
    double buy_percent = VolumeCalculator::calculateBuyVolumePercent(trades1);
    // Buy vol = 20, Total vol = 50, Percent = (20/50)*100 = 40%
    assert(std::abs(buy_percent - 40.0) < 0.01);
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    double buy_percent_empty = VolumeCalculator::calculateBuyVolumePercent(empty_trades);
    assert(buy_percent_empty == 0.0);
    
    // Test case 3: Zero total volume
    std::vector<TradeData> zero_vol;
    zero_vol.push_back(TradeData(1, 100.0, 0.0f, TradeSide::BUY, 1, 0));
    zero_vol.push_back(TradeData(2, 101.0, 0.0f, TradeSide::SELL, 1, 0));
    double buy_percent_zero = VolumeCalculator::calculateBuyVolumePercent(zero_vol);
    assert(buy_percent_zero == 0.0);
    
    std::cout << "calculateBuyVolumePercent tests PASSED!" << std::endl;
}

void testCalculateSellVolumePercent() {
    std::cout << "Testing calculateSellVolumePercent..." << std::endl;
    
    // Test case 1: Mixed trades
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 100.0, 20.0f, TradeSide::BUY, 1, 0));
    trades1.push_back(TradeData(2, 101.0, 30.0f, TradeSide::SELL, 1, 0));
    
    double sell_percent = VolumeCalculator::calculateSellVolumePercent(trades1);
    // Sell vol = 30, Total vol = 50, Percent = (30/50)*100 = 60%
    assert(std::abs(sell_percent - 60.0) < 0.01);
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    double sell_percent_empty = VolumeCalculator::calculateSellVolumePercent(empty_trades);
    assert(sell_percent_empty == 0.0);
    
    // Test case 3: Zero total volume
    std::vector<TradeData> zero_vol;
    zero_vol.push_back(TradeData(1, 100.0, 0.0f, TradeSide::BUY, 1, 0));
    zero_vol.push_back(TradeData(2, 101.0, 0.0f, TradeSide::SELL, 1, 0));
    double sell_percent_zero = VolumeCalculator::calculateSellVolumePercent(zero_vol);
    assert(sell_percent_zero == 0.0);
    
    std::cout << "calculateSellVolumePercent tests PASSED!" << std::endl;
}

void testCalculateAverageSize() {
    std::cout << "Testing calculateAverageSize..." << std::endl;
    
    // Test case 1: Multiple trades
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 100.0, 10.0f, TradeSide::BUY, 1, 0));
    trades1.push_back(TradeData(2, 101.0, 20.0f, TradeSide::SELL, 1, 0));
    trades1.push_back(TradeData(3, 102.0, 30.0f, TradeSide::BUY, 1, 0));
    
    double avg_size = VolumeCalculator::calculateAverageSize(trades1);
    // Total vol = 60, Count = 3, Avg = 60/3 = 20
    assert(std::abs(avg_size - 20.0) < 0.01);
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    double avg_size_empty = VolumeCalculator::calculateAverageSize(empty_trades);
    assert(avg_size_empty == 0.0);
    
    // Test case 3: Single trade
    std::vector<TradeData> single_trade;
    single_trade.push_back(TradeData(1, 100.0, 42.0f, TradeSide::BUY, 1, 0));
    double avg_size_single = VolumeCalculator::calculateAverageSize(single_trade);
    assert(std::abs(avg_size_single - 42.0) < 0.01);
    
    std::cout << "calculateAverageSize tests PASSED!" << std::endl;
}

void testCalculateAverageBuySize() {
    std::cout << "Testing calculateAverageBuySize..." << std::endl;
    
    // Test case 1: Mixed trades
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 100.0, 10.0f, TradeSide::BUY, 1, 0));
    trades1.push_back(TradeData(2, 101.0, 20.0f, TradeSide::SELL, 1, 0));
    trades1.push_back(TradeData(3, 102.0, 30.0f, TradeSide::BUY, 1, 0));
    
    double avg_buy_size = VolumeCalculator::calculateAverageBuySize(trades1);
    // Buy vol = 40, Buy count = 2, Avg = 40/2 = 20
    assert(std::abs(avg_buy_size - 20.0) < 0.01);
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    double avg_buy_size_empty = VolumeCalculator::calculateAverageBuySize(empty_trades);
    assert(avg_buy_size_empty == 0.0);
    
    // Test case 3: No buy trades
    std::vector<TradeData> no_buy;
    no_buy.push_back(TradeData(1, 100.0, 20.0f, TradeSide::SELL, 1, 0));
    no_buy.push_back(TradeData(2, 101.0, 30.0f, TradeSide::SELL, 1, 0));
    double avg_buy_size_no_buy = VolumeCalculator::calculateAverageBuySize(no_buy);
    assert(avg_buy_size_no_buy == 0.0);
    
    std::cout << "calculateAverageBuySize tests PASSED!" << std::endl;
}

void testCalculateAverageSellSize() {
    std::cout << "Testing calculateAverageSellSize..." << std::endl;
    
    // Test case 1: Mixed trades
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 100.0, 10.0f, TradeSide::BUY, 1, 0));
    trades1.push_back(TradeData(2, 101.0, 20.0f, TradeSide::SELL, 1, 0));
    trades1.push_back(TradeData(3, 102.0, 30.0f, TradeSide::SELL, 1, 0));
    
    double avg_sell_size = VolumeCalculator::calculateAverageSellSize(trades1);
    // Sell vol = 50, Sell count = 2, Avg = 50/2 = 25
    assert(std::abs(avg_sell_size - 25.0) < 0.01);
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    double avg_sell_size_empty = VolumeCalculator::calculateAverageSellSize(empty_trades);
    assert(avg_sell_size_empty == 0.0);
    
    // Test case 3: No sell trades
    std::vector<TradeData> no_sell;
    no_sell.push_back(TradeData(1, 100.0, 20.0f, TradeSide::BUY, 1, 0));
    no_sell.push_back(TradeData(2, 101.0, 30.0f, TradeSide::BUY, 1, 0));
    double avg_sell_size_no_sell = VolumeCalculator::calculateAverageSellSize(no_sell);
    assert(avg_sell_size_no_sell == 0.0);
    
    std::cout << "calculateAverageSellSize tests PASSED!" << std::endl;
}

void testCalculateMaxOneTradeVolume() {
    std::cout << "Testing calculateMaxOneTradeVolume..." << std::endl;
    
    // Test case 1: Multiple trades with different volumes
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 100.0, 10.0f, TradeSide::BUY, 1, 0));
    trades1.push_back(TradeData(2, 101.0, 25.0f, TradeSide::SELL, 1, 0));
    trades1.push_back(TradeData(3, 102.0, 15.0f, TradeSide::BUY, 1, 0));
    
    float max_vol = VolumeCalculator::calculateMaxOneTradeVolume(trades1);
    assert(max_vol == 25.0f);
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    float max_vol_empty = VolumeCalculator::calculateMaxOneTradeVolume(empty_trades);
    assert(max_vol_empty == 0.0f);
    
    // Test case 3: Single trade
    std::vector<TradeData> single_trade;
    single_trade.push_back(TradeData(1, 100.0, 42.0f, TradeSide::BUY, 1, 0));
    float max_vol_single = VolumeCalculator::calculateMaxOneTradeVolume(single_trade);
    assert(max_vol_single == 42.0f);
    
    std::cout << "calculateMaxOneTradeVolume tests PASSED!" << std::endl;
}

void testCalculateFilteredVolume() {
    std::cout << "Testing calculateFilteredVolume..." << std::endl;
    
    // Test case 1: Mixed prices
    std::vector<TradeData> trades1;
    trades1.push_back(TradeData(1, 95.0, 10.0f, TradeSide::BUY, 1, 0));  // Below range
    trades1.push_back(TradeData(2, 100.0, 20.0f, TradeSide::SELL, 1, 0)); // In range
    trades1.push_back(TradeData(3, 105.0, 30.0f, TradeSide::BUY, 1, 0)); // In range
    trades1.push_back(TradeData(4, 110.0, 40.0f, TradeSide::SELL, 1, 0)); // Above range
    
    double filtered_vol = VolumeCalculator::calculateFilteredVolume(trades1, 100.0, 105.0);
    // Only trades with prices 100.0 and 105.0 are in range, volumes 20 + 30 = 50
    assert(std::abs(filtered_vol - 50.0) < 0.01);
    
    // Test case 2: Empty vector
    std::vector<TradeData> empty_trades;
    double filtered_vol_empty = VolumeCalculator::calculateFilteredVolume(empty_trades, 100.0, 110.0);
    assert(filtered_vol_empty == 0.0);
    
    // Test case 3: No trades in range
    std::vector<TradeData> no_in_range;
    no_in_range.push_back(TradeData(1, 90.0, 10.0f, TradeSide::BUY, 1, 0));
    no_in_range.push_back(TradeData(2, 120.0, 20.0f, TradeSide::SELL, 1, 0));
    double filtered_vol_none = VolumeCalculator::calculateFilteredVolume(no_in_range, 100.0, 110.0);
    assert(filtered_vol_none == 0.0);
    
    // Test case 4: Default parameters (should include all trades)
    std::vector<TradeData> all_trades;
    all_trades.push_back(TradeData(1, 95.0, 10.0f, TradeSide::BUY, 1, 0));
    all_trades.push_back(TradeData(2, 100.0, 20.0f, TradeSide::SELL, 1, 0));
    all_trades.push_back(TradeData(3, 105.0, 30.0f, TradeSide::BUY, 1, 0));
    double filtered_vol_default = VolumeCalculator::calculateFilteredVolume(all_trades); // Using defaults
    // Total volume should be 10 + 20 + 30 = 60
    assert(std::abs(filtered_vol_default - 60.0) < 0.01);
    
    std::cout << "calculateFilteredVolume tests PASSED!" << std::endl;
}

int main() {
    std::cout << "Running VolumeCalculator tests..." << std::endl;
    
    testCalculateDelta();
    testCalculateDeltaPercent();
    testCalculateBuyVolumePercent();
    testCalculateSellVolumePercent();
    testCalculateAverageSize();
    testCalculateAverageBuySize();
    testCalculateAverageSellSize();
    testCalculateMaxOneTradeVolume();
    testCalculateFilteredVolume();
    
    std::cout << "All VolumeCalculator tests PASSED!" << std::endl;
    return 0;
}