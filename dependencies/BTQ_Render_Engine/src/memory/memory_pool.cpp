#include "../include/memory/memory_pool.hpp"

#include <mutex>

namespace BTQuant {

// TradeDataPool implementation
TradeDataPool& TradeDataPool::getInstance() {
    static TradeDataPool instance;
    return instance;
}

Data::TradeData* TradeDataPool::allocate() {
    return pool_.allocate();
}

void TradeDataPool::deallocate(Data::TradeData* trade) {
    pool_.deallocate(trade);
}

void TradeDataPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// ClusterCellPool implementation
ClusterCellPool& ClusterCellPool::getInstance() {
    static ClusterCellPool instance;
    return instance;
}

Analytics::ClusterCell* ClusterCellPool::allocate() {
    return pool_.allocate();
}

void ClusterCellPool::deallocate(Analytics::ClusterCell* cell) {
    pool_.deallocate(cell);
}

void ClusterCellPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// EMAIndicatorPool implementation
EMAIndicatorPool& EMAIndicatorPool::getInstance() {
    static EMAIndicatorPool instance;
    return instance;
}

EMAIndicator* EMAIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void EMAIndicatorPool::deallocate(EMAIndicator* indicator) {
    pool_.deallocate(indicator);
}

void EMAIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// SMAIndicatorPool implementation
SMAIndicatorPool& SMAIndicatorPool::getInstance() {
    static SMAIndicatorPool instance;
    return instance;
}

SMAIndicator* SMAIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void SMAIndicatorPool::deallocate(SMAIndicator* indicator) {
    pool_.deallocate(indicator);
}

void SMAIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// RSIIndicatorPool implementation
RSIIndicatorPool& RSIIndicatorPool::getInstance() {
    static RSIIndicatorPool instance;
    return instance;
}

RSIIndicator* RSIIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void RSIIndicatorPool::deallocate(RSIIndicator* indicator) {
    pool_.deallocate(indicator);
}

void RSIIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// MACDIndicatorPool implementation
MACDIndicatorPool& MACDIndicatorPool::getInstance() {
    static MACDIndicatorPool instance;
    return instance;
}

MACDIndicator* MACDIndicatorPool::allocate(int fast_period, int slow_period, int signal_period) {
    return pool_.allocate(fast_period, slow_period, signal_period);
}

void MACDIndicatorPool::deallocate(MACDIndicator* indicator) {
    pool_.deallocate(indicator);
}

void MACDIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// Additional specialized memory pools for other frequently allocated objects
// BollingerBandIndicatorPool implementation
BollingerBandIndicatorPool& BollingerBandIndicatorPool::getInstance() {
    static BollingerBandIndicatorPool instance;
    return instance;
}

BollingerBandIndicator* BollingerBandIndicatorPool::allocate(int period, double std_dev) {
    return pool_.allocate(period, std_dev);
}

void BollingerBandIndicatorPool::deallocate(BollingerBandIndicator* indicator) {
    pool_.deallocate(indicator);
}

void BollingerBandIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// StochasticIndicatorPool implementation
StochasticIndicatorPool& StochasticIndicatorPool::getInstance() {
    static StochasticIndicatorPool instance;
    return instance;
}

StochasticIndicator* StochasticIndicatorPool::allocate(int k_period, int d_period, int slowing_period) {
    return pool_.allocate(k_period, d_period, slowing_period);
}

void StochasticIndicatorPool::deallocate(StochasticIndicator* indicator) {
    pool_.deallocate(indicator);
}

void StochasticIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// ATRIndicatorPool implementation
ATRIndicatorPool& ATRIndicatorPool::getInstance() {
    static ATRIndicatorPool instance;
    return instance;
}

ATRIndicator* ATRIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void ATRIndicatorPool::deallocate(ATRIndicator* indicator) {
    pool_.deallocate(indicator);
}

void ATRIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// OrderPool implementation
OrderPool& OrderPool::getInstance() {
    static OrderPool instance;
    return instance;
}

OrderManager::Order* OrderPool::allocate() {
    return pool_.allocate();
}

void OrderPool::deallocate(OrderManager::Order* order) {
    pool_.deallocate(order);
}

void OrderPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// ProcessedTradePool implementation
ProcessedTradePool& ProcessedTradePool::getInstance() {
    static ProcessedTradePool instance;
    return instance;
}

ProcessedTrade* ProcessedTradePool::allocate() {
    return pool_.allocate();
}

void ProcessedTradePool::deallocate(ProcessedTrade* trade) {
    pool_.deallocate(trade);
}

void ProcessedTradePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// OHLCVCandlePool implementation
OHLCVCandlePool& OHLCVCandlePool::getInstance() {
    static OHLCVCandlePool instance;
    return instance;
}

RenderEngine::OHLCVCandle* OHLCVCandlePool::allocate() {
    return pool_.allocate();
}

void OHLCVCandlePool::deallocate(RenderEngine::OHLCVCandle* candle) {
    pool_.deallocate(candle);
}

void OHLCVCandlePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// VolumeProfileLevelPool implementation
VolumeProfileLevelPool& VolumeProfileLevelPool::getInstance() {
    static VolumeProfileLevelPool instance;
    return instance;
}

VolumeProfileLevel* VolumeProfileLevelPool::allocate() {
    return pool_.allocate();
}

void VolumeProfileLevelPool::deallocate(VolumeProfileLevel* level) {
    pool_.deallocate(level);
}

void VolumeProfileLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// TradeRecordPool implementation
TradeRecordPool& TradeRecordPool::getInstance() {
    static TradeRecordPool instance;
    return instance;
}

PositionManager::TradeRecord* TradeRecordPool::allocate() {
    return pool_.allocate();
}

void TradeRecordPool::deallocate(PositionManager::TradeRecord* record) {
    pool_.deallocate(record);
}

void TradeRecordPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// HotspineTradeTickPool implementation
HotspineTradeTickPool& HotspineTradeTickPool::getInstance() {
    static HotspineTradeTickPool instance;
    return instance;
}

RenderEngine::HotspineTradeTick* HotspineTradeTickPool::allocate() {
    return pool_.allocate();
}

void HotspineTradeTickPool::deallocate(RenderEngine::HotspineTradeTick* tick) {
    pool_.deallocate(tick);
}

void HotspineTradeTickPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// OrderBookLevelPool implementation
OrderBookLevelPool& OrderBookLevelPool::getInstance() {
    static OrderBookLevelPool instance;
    return instance;
}

RenderEngine::OrderBookLevel* OrderBookLevelPool::allocate() {
    return pool_.allocate();
}

void OrderBookLevelPool::deallocate(RenderEngine::OrderBookLevel* level) {
    pool_.deallocate(level);
}

void OrderBookLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// CompressedCandlePool implementation
CompressedCandlePool& CompressedCandlePool::getInstance() {
    static CompressedCandlePool instance;
    return instance;
}

Data::CompressedCandle* CompressedCandlePool::allocate() {
    return pool_.allocate();
}

void CompressedCandlePool::deallocate(Data::CompressedCandle* candle) {
    pool_.deallocate(candle);
}

void CompressedCandlePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// CompressedTradePool implementation
CompressedTradePool& CompressedTradePool::getInstance() {
    static CompressedTradePool instance;
    return instance;
}

Data::CompressedTrade* CompressedTradePool::allocate() {
    return pool_.allocate();
}

void CompressedTradePool::deallocate(Data::CompressedTrade* trade) {
    pool_.deallocate(trade);
}

void CompressedTradePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FibonacciLevelPool implementation
FibonacciLevelPool& FibonacciLevelPool::getInstance() {
    static FibonacciLevelPool instance;
    return instance;
}

FibonacciLevel* FibonacciLevelPool::allocate() {
    return pool_.allocate();
}

void FibonacciLevelPool::deallocate(FibonacciLevel* level) {
    pool_.deallocate(level);
}

void FibonacciLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// GpuOrderBookLevelPool implementation
GpuOrderBookLevelPool& GpuOrderBookLevelPool::getInstance() {
    static GpuOrderBookLevelPool instance;
    return instance;
}

RenderEngine::GpuOrderBookLevel* GpuOrderBookLevelPool::allocate() {
    return pool_.allocate();
}

void GpuOrderBookLevelPool::deallocate(RenderEngine::GpuOrderBookLevel* level) {
    pool_.deallocate(level);
}

void GpuOrderBookLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// DepthLevelPool implementation
DepthLevelPool& DepthLevelPool::getInstance() {
    static DepthLevelPool instance;
    return instance;
}

BTQuant::MarketDepthAnalyzer::DepthLevel* DepthLevelPool::allocate() {
    return pool_.allocate();
}

void DepthLevelPool::deallocate(BTQuant::MarketDepthAnalyzer::DepthLevel* level) {
    pool_.deallocate(level);
}

void DepthLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}


// HotOrderbookLevelPool implementation
HotOrderbookLevelPool& HotOrderbookLevelPool::getInstance() {
    static HotOrderbookLevelPool instance;
    return instance;
}

HotOrderbookLevel* HotOrderbookLevelPool::allocate() {
    return pool_.allocate();
}

void HotOrderbookLevelPool::deallocate(HotOrderbookLevel* level) {
    pool_.deallocate(level);
}

void HotOrderbookLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

} // namespace BTQuant