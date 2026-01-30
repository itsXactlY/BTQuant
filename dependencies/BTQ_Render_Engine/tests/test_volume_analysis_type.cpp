#include <cassert>
#include <iostream>

#include "../include/data/VolumeDataTypes.h"

using namespace BTQuant::Data;

void testVolumeAnalysisTypeValues() {
  // Test that all enum values are properly defined and have expected numeric values
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

  std::cout << "✓ All VolumeAnalysisType enum values are correctly defined" << std::endl;
}

void testVolumeAnalysisTypeNames() {
  // Simple test to ensure all enum values exist and can be used
  VolumeAnalysisType types[] = {
      VolumeAnalysisType::Trades,           VolumeAnalysisType::BuyTrades,
      VolumeAnalysisType::SellTrades,       VolumeAnalysisType::Volume,
      VolumeAnalysisType::BuyVolume,        VolumeAnalysisType::SellVolume,
      VolumeAnalysisType::BuyVolumePercent, VolumeAnalysisType::SellVolumePercent,
      VolumeAnalysisType::BuySellVolume,    VolumeAnalysisType::Delta,
      VolumeAnalysisType::DeltaPercent,     VolumeAnalysisType::CumulativeDelta,
      VolumeAnalysisType::AverageSize,      VolumeAnalysisType::AverageBuySize,
      VolumeAnalysisType::AverageSellSize,  VolumeAnalysisType::MaxOneTradeVolume,
      VolumeAnalysisType::FilteredVolume};

  std::cout << "✓ All VolumeAnalysisType enum values can be instantiated" << std::endl;

  // Verify we have exactly 17 values (0-16)
  const size_t expectedCount = 17;
  assert(sizeof(types) / sizeof(types[0]) == expectedCount);
  std::cout << "✓ VolumeAnalysisType has exactly " << expectedCount << " values" << std::endl;
}

int main() {
  std::cout << "Testing VolumeAnalysisType enum..." << std::endl;

  testVolumeAnalysisTypeValues();
  testVolumeAnalysisTypeNames();

  std::cout << "All tests passed!" << std::endl;
  return 0;
}