#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "../include/data/PriceLevel.h"
#include "../include/data/TradeData.h"
#include "../include/data/VolumeDataTypes.h"
#include "../include/widgets/FootprintCell.h"
#include "../include/widgets/VolumeProfileNode.h"

using namespace BTQuant::Data;

void testTradeDataStructure() {
  // Test default constructor
  TradeData defaultTrade;
  assert(defaultTrade.timestamp == 0);
  assert(defaultTrade.price == 0.0);
  assert(defaultTrade.volume == 0.0);
  assert(defaultTrade.side == TradeSide::BUY);

  // Test parameterized constructor
  TradeData trade(1634567890000, 45000.50, 2.5, TradeSide::SELL, "Binance");
  assert(trade.timestamp == 1634567890000);
  assert(std::abs(trade.price - 45000.50) < 1e-6);
  assert(std::abs(trade.volume - 2.5) < 1e-6);
  assert(trade.side == TradeSide::SELL);
  assert(trade.exchange == "Binance");

  std::cout << "✓ TradeData structure tests passed" << std::endl;
}

void testPriceLevelStructure() {
  PriceLevel level;
  level.price = 45000.0;
  level.bidVolume = 100.5;
  level.askVolume = 80.3;
  level.bidOrders = 5;
  level.askOrders = 3;

  assert(std::abs(level.price - 45000.0) < 1e-6);
  assert(std::abs(level.bidVolume - 100.5) < 1e-6);
  assert(std::abs(level.askVolume - 80.3) < 1e-6);
  assert(level.bidOrders == 5);
  assert(level.askOrders == 3);

  std::cout << "✓ PriceLevel structure tests passed" << std::endl;
}

void testFootprintCellStructure() {
  FootprintCell cell;
  cell.priceLevel = 45000.0;
  cell.timeBucket = 1634567890;
  cell.buyVolume = 150.5;
  cell.sellVolume = 120.3;
  cell.delta = 30.2;
  cell.numBuyTrades = 8;
  cell.numSellTrades = 6;
  cell.maxSingleTrade = 25.0;

  assert(std::abs(cell.priceLevel - 45000.0) < 1e-6);
  assert(cell.timeBucket == 1634567890);
  assert(std::abs(cell.buyVolume - 150.5) < 1e-6);
  assert(std::abs(cell.sellVolume - 120.3) < 1e-6);
  assert(std::abs(cell.delta - 30.2) < 1e-6);
  assert(cell.numBuyTrades == 8);
  assert(cell.numSellTrades == 6);
  assert(std::abs(cell.maxSingleTrade - 25.0) < 1e-6);

  std::cout << "✓ FootprintCell structure tests passed" << std::endl;
}

void testVolumeProfileNodeStructure() {
  VolumeProfileNode node;
  node.priceLevel = 45000.0;
  node.totalVolume = 270.8;
  node.buyVolume = 150.5;
  node.sellVolume = 120.3;
  node.delta = 30.2;
  node.numTrades = 14;

  assert(std::abs(node.priceLevel - 45000.0) < 1e-6);
  assert(std::abs(node.totalVolume - 270.8) < 1e-6);
  assert(std::abs(node.buyVolume - 150.5) < 1e-6);
  assert(std::abs(node.sellVolume - 120.3) < 1e-6);
  assert(std::abs(node.delta - 30.2) < 1e-6);
  assert(node.numTrades == 14);

  std::cout << "✓ VolumeProfileNode structure tests passed" << std::endl;
}

void testVolumeAnalysisTypeConsistency() {
  // Test that all values are sequential starting from 0
  assert(static_cast<int>(VolumeAnalysisType::Trades) == 0);
  assert(static_cast<int>(VolumeAnalysisType::BuyTrades) == 1);
  assert(static_cast<int>(VolumeAnalysisType::SellTrades) == 2);
  assert(static_cast<int>(VolumeAnalysisType::Volume) == 3);
  assert(static_cast<int>(VolumeAnalysisType::BuyVolume) == 4);
  assert(static_cast<int>(VolumeAnalysisType::SellVolume) == 5);
  assert(static_cast<int>(VolumeAnalysisType::BuyVolumePercent) == 6);
  assert(static_cast<int>(VolumeAnalysisType::SellVolumePercent) == 7);
  assert(static_cast<int>(VolumeAnalysisType::BuySellVolume) == 8);
  assert(static_cast<int>(VolumeAnalysisType::Delta) == 9);
  assert(static_cast<int>(VolumeAnalysisType::DeltaPercent) == 10);
  assert(static_cast<int>(VolumeAnalysisType::CumulativeDelta) == 11);
  assert(static_cast<int>(VolumeAnalysisType::AverageSize) == 12);
  assert(static_cast<int>(VolumeAnalysisType::AverageBuySize) == 13);
  assert(static_cast<int>(VolumeAnalysisType::AverageSellSize) == 14);
  assert(static_cast<int>(VolumeAnalysisType::MaxOneTradeVolume) == 15);
  assert(static_cast<int>(VolumeAnalysisType::FilteredVolume) == 16);

  // Verify the highest value
  assert(static_cast<int>(VolumeAnalysisType::FilteredVolume) == 16);

  std::cout << "✓ VolumeAnalysisType consistency tests passed" << std::endl;
}

void testTradeSideEnum() {
  // Test that TradeSide enum has correct values
  assert(TradeSide::BUY != TradeSide::SELL);

  TradeData buyTrade(0, 0, 0, TradeSide::BUY, "");
  TradeData sellTrade(0, 0, 0, TradeSide::SELL, "");

  assert(buyTrade.side == TradeSide::BUY);
  assert(sellTrade.side == TradeSide::SELL);

  std::cout << "✓ TradeSide enum tests passed" << std::endl;
}

void testVolumeCalculations() {
  // Test volume calculations using the structures
  FootprintCell cell;
  cell.buyVolume = 150.5;
  cell.sellVolume = 120.3;
  cell.numBuyTrades = 8;
  cell.numSellTrades = 6;

  // Calculate derived values
  double totalVolume = cell.buyVolume + cell.sellVolume;
  double delta = cell.buyVolume - cell.sellVolume;
  double buyPercent = (cell.buyVolume / totalVolume) * 100.0;
  double sellPercent = (cell.sellVolume / totalVolume) * 100.0;

  assert(std::abs(totalVolume - 270.8) < 1e-6);
  assert(std::abs(delta - 30.2) < 1e-6);
  assert(std::abs(buyPercent - 55.58) < 0.01);   // Approximate
  assert(std::abs(sellPercent - 44.42) < 0.01);  // Approximate

  std::cout << "✓ Volume calculation tests passed" << std::endl;
}

void testVolumeProfileNodeCalculations() {
  VolumeProfileNode node;
  node.buyVolume = 150.5;
  node.sellVolume = 120.3;
  node.numTrades = 14;

  // Calculate derived values
  double totalVolume = node.buyVolume + node.sellVolume;
  double delta = node.buyVolume - node.sellVolume;

  assert(std::abs(totalVolume - 270.8) < 1e-6);
  assert(std::abs(delta - 30.2) < 1e-6);

  std::cout << "✓ VolumeProfileNode calculation tests passed" << std::endl;
}

int main() {
  std::cout << "Testing merged volume analysis features..." << std::endl;

  testTradeDataStructure();
  testPriceLevelStructure();
  testFootprintCellStructure();
  testVolumeProfileNodeStructure();
  testVolumeAnalysisTypeConsistency();
  testTradeSideEnum();
  testVolumeCalculations();
  testVolumeProfileNodeCalculations();

  std::cout << "All merged feature tests passed!" << std::endl;
  return 0;
}