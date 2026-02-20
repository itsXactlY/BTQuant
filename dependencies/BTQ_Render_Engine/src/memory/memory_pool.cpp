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

// CandleClusterPool implementation
CandleClusterPool& CandleClusterPool::getInstance() {
    static CandleClusterPool instance;
    return instance;
}

RenderEngine::CandleCluster* CandleClusterPool::allocate(float x, float y, float w, float h,
                                                        uint32_t bidVol, uint32_t askVol, uint32_t tradeCnt,
                                                        float vw, bool hasTrades) {
    return pool_.allocate(x, y, w, h, bidVol, askVol, tradeCnt, vw, hasTrades);
}

void CandleClusterPool::deallocate(RenderEngine::CandleCluster* cluster) {
    pool_.deallocate(cluster);
}

void CandleClusterPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// VolumeProfileNodePool implementation
VolumeProfileNodePool& VolumeProfileNodePool::getInstance() {
    static VolumeProfileNodePool instance;
    return instance;
}

VolumeProfileNode* VolumeProfileNodePool::allocate() {
    return pool_.allocate();
}

void VolumeProfileNodePool::deallocate(VolumeProfileNode* node) {
    pool_.deallocate(node);
}

void VolumeProfileNodePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FootprintCellPool implementation
FootprintCellPool& FootprintCellPool::getInstance() {
    static FootprintCellPool instance;
    return instance;
}

FootprintCell* FootprintCellPool::allocate() {
    return pool_.allocate();
}

void FootprintCellPool::deallocate(FootprintCell* cell) {
    pool_.deallocate(cell);
}

void FootprintCellPool::preallocate(size_t count) {
    pool_.preallocate(count);
}


// FastTradeDataPool implementation
FastTradeDataPool& FastTradeDataPool::getInstance() {
    static FastTradeDataPool instance;
    return instance;
}

Data::TradeData* FastTradeDataPool::allocate() {
    return pool_.allocate();
}

void FastTradeDataPool::deallocate(Data::TradeData* trade) {
    pool_.deallocate(trade);
}

void FastTradeDataPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastClusterCellPool implementation
FastClusterCellPool& FastClusterCellPool::getInstance() {
    static FastClusterCellPool instance;
    return instance;
}

Analytics::ClusterCell* FastClusterCellPool::allocate() {
    return pool_.allocate();
}

void FastClusterCellPool::deallocate(Analytics::ClusterCell* cell) {
    pool_.deallocate(cell);
}

void FastClusterCellPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastEMAIndicatorPool implementation
FastEMAIndicatorPool& FastEMAIndicatorPool::getInstance() {
    static FastEMAIndicatorPool instance;
    return instance;
}

EMAIndicator* FastEMAIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void FastEMAIndicatorPool::deallocate(EMAIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastEMAIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastSMAIndicatorPool implementation
FastSMAIndicatorPool& FastSMAIndicatorPool::getInstance() {
    static FastSMAIndicatorPool instance;
    return instance;
}

SMAIndicator* FastSMAIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void FastSMAIndicatorPool::deallocate(SMAIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastSMAIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastRSIIndicatorPool implementation
FastRSIIndicatorPool& FastRSIIndicatorPool::getInstance() {
    static FastRSIIndicatorPool instance;
    return instance;
}

RSIIndicator* FastRSIIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void FastRSIIndicatorPool::deallocate(RSIIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastRSIIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastMACDIndicatorPool implementation
FastMACDIndicatorPool& FastMACDIndicatorPool::getInstance() {
    static FastMACDIndicatorPool instance;
    return instance;
}

MACDIndicator* FastMACDIndicatorPool::allocate(int fast_period, int slow_period, int signal_period) {
    return pool_.allocate(fast_period, slow_period, signal_period);
}

void FastMACDIndicatorPool::deallocate(MACDIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastMACDIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastBollingerBandIndicatorPool implementation
FastBollingerBandIndicatorPool& FastBollingerBandIndicatorPool::getInstance() {
    static FastBollingerBandIndicatorPool instance;
    return instance;
}

BollingerBandIndicator* FastBollingerBandIndicatorPool::allocate(int period, double std_dev) {
    return pool_.allocate(period, std_dev);
}

void FastBollingerBandIndicatorPool::deallocate(BollingerBandIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastBollingerBandIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastStochasticIndicatorPool implementation
FastStochasticIndicatorPool& FastStochasticIndicatorPool::getInstance() {
    static FastStochasticIndicatorPool instance;
    return instance;
}

StochasticIndicator* FastStochasticIndicatorPool::allocate(int k_period, int d_period, int slowing_period) {
    return pool_.allocate(k_period, d_period, slowing_period);
}

void FastStochasticIndicatorPool::deallocate(StochasticIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastStochasticIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastATRIndicatorPool implementation
FastATRIndicatorPool& FastATRIndicatorPool::getInstance() {
    static FastATRIndicatorPool instance;
    return instance;
}

ATRIndicator* FastATRIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void FastATRIndicatorPool::deallocate(ATRIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastATRIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastOrderPool implementation
FastOrderPool& FastOrderPool::getInstance() {
    static FastOrderPool instance;
    return instance;
}

OrderManager::Order* FastOrderPool::allocate() {
    return pool_.allocate();
}

void FastOrderPool::deallocate(OrderManager::Order* order) {
    pool_.deallocate(order);
}

void FastOrderPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastProcessedTradePool implementation
FastProcessedTradePool& FastProcessedTradePool::getInstance() {
    static FastProcessedTradePool instance;
    return instance;
}

ProcessedTrade* FastProcessedTradePool::allocate() {
    return pool_.allocate();
}

void FastProcessedTradePool::deallocate(ProcessedTrade* trade) {
    pool_.deallocate(trade);
}

void FastProcessedTradePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastOHLCVCandlePool implementation
FastOHLCVCandlePool& FastOHLCVCandlePool::getInstance() {
    static FastOHLCVCandlePool instance;
    return instance;
}

RenderEngine::OHLCVCandle* FastOHLCVCandlePool::allocate() {
    return pool_.allocate();
}

void FastOHLCVCandlePool::deallocate(RenderEngine::OHLCVCandle* candle) {
    pool_.deallocate(candle);
}

void FastOHLCVCandlePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastVolumeProfileLevelPool implementation
FastVolumeProfileLevelPool& FastVolumeProfileLevelPool::getInstance() {
    static FastVolumeProfileLevelPool instance;
    return instance;
}

VolumeProfileLevel* FastVolumeProfileLevelPool::allocate() {
    return pool_.allocate();
}

void FastVolumeProfileLevelPool::deallocate(VolumeProfileLevel* level) {
    pool_.deallocate(level);
}

void FastVolumeProfileLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastTradeRecordPool implementation
FastTradeRecordPool& FastTradeRecordPool::getInstance() {
    static FastTradeRecordPool instance;
    return instance;
}

PositionManager::TradeRecord* FastTradeRecordPool::allocate() {
    return pool_.allocate();
}

void FastTradeRecordPool::deallocate(PositionManager::TradeRecord* record) {
    pool_.deallocate(record);
}

void FastTradeRecordPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastHotspineTradeTickPool implementation
FastHotspineTradeTickPool& FastHotspineTradeTickPool::getInstance() {
    static FastHotspineTradeTickPool instance;
    return instance;
}

RenderEngine::HotspineTradeTick* FastHotspineTradeTickPool::allocate() {
    return pool_.allocate();
}

void FastHotspineTradeTickPool::deallocate(RenderEngine::HotspineTradeTick* tick) {
    pool_.deallocate(tick);
}

void FastHotspineTradeTickPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastOrderBookLevelPool implementation
FastOrderBookLevelPool& FastOrderBookLevelPool::getInstance() {
    static FastOrderBookLevelPool instance;
    return instance;
}

RenderEngine::OrderBookLevel* FastOrderBookLevelPool::allocate() {
    return pool_.allocate();
}

void FastOrderBookLevelPool::deallocate(RenderEngine::OrderBookLevel* level) {
    pool_.deallocate(level);
}

void FastOrderBookLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastCompressedCandlePool implementation
FastCompressedCandlePool& FastCompressedCandlePool::getInstance() {
    static FastCompressedCandlePool instance;
    return instance;
}

Data::CompressedCandle* FastCompressedCandlePool::allocate() {
    return pool_.allocate();
}

void FastCompressedCandlePool::deallocate(Data::CompressedCandle* candle) {
    pool_.deallocate(candle);
}

void FastCompressedCandlePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastCompressedTradePool implementation
FastCompressedTradePool& FastCompressedTradePool::getInstance() {
    static FastCompressedTradePool instance;
    return instance;
}

Data::CompressedTrade* FastCompressedTradePool::allocate() {
    return pool_.allocate();
}

void FastCompressedTradePool::deallocate(Data::CompressedTrade* trade) {
    pool_.deallocate(trade);
}

void FastCompressedTradePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastFibonacciLevelPool implementation
FastFibonacciLevelPool& FastFibonacciLevelPool::getInstance() {
    static FastFibonacciLevelPool instance;
    return instance;
}

FibonacciLevel* FastFibonacciLevelPool::allocate() {
    return pool_.allocate();
}

void FastFibonacciLevelPool::deallocate(FibonacciLevel* level) {
    pool_.deallocate(level);
}

void FastFibonacciLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastGpuOrderBookLevelPool implementation
FastGpuOrderBookLevelPool& FastGpuOrderBookLevelPool::getInstance() {
    static FastGpuOrderBookLevelPool instance;
    return instance;
}

RenderEngine::GpuOrderBookLevel* FastGpuOrderBookLevelPool::allocate() {
    return pool_.allocate();
}

void FastGpuOrderBookLevelPool::deallocate(RenderEngine::GpuOrderBookLevel* level) {
    pool_.deallocate(level);
}

void FastGpuOrderBookLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastDepthLevelPool implementation
FastDepthLevelPool& FastDepthLevelPool::getInstance() {
    static FastDepthLevelPool instance;
    return instance;
}

BTQuant::MarketDepthAnalyzer::DepthLevel* FastDepthLevelPool::allocate() {
    return pool_.allocate();
}

void FastDepthLevelPool::deallocate(BTQuant::MarketDepthAnalyzer::DepthLevel* level) {
    pool_.deallocate(level);
}

void FastDepthLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastHotOrderbookLevelPool implementation
FastHotOrderbookLevelPool& FastHotOrderbookLevelPool::getInstance() {
    static FastHotOrderbookLevelPool instance;
    return instance;
}

HotOrderbookLevel* FastHotOrderbookLevelPool::allocate() {
    return pool_.allocate();
}

void FastHotOrderbookLevelPool::deallocate(HotOrderbookLevel* level) {
    pool_.deallocate(level);
}

void FastHotOrderbookLevelPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastCandleClusterPool implementation
FastCandleClusterPool& FastCandleClusterPool::getInstance() {
    static FastCandleClusterPool instance;
    return instance;
}

RenderEngine::CandleCluster* FastCandleClusterPool::allocate(float x, float y, float w, float h,
                                                            uint32_t bidVol, uint32_t askVol, uint32_t tradeCnt,
                                                            float vw, bool hasTrades) {
    return pool_.allocate(x, y, w, h, bidVol, askVol, tradeCnt, vw, hasTrades);
}

void FastCandleClusterPool::deallocate(RenderEngine::CandleCluster* cluster) {
    pool_.deallocate(cluster);
}

void FastCandleClusterPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastVolumeProfileNodePool implementation
FastVolumeProfileNodePool& FastVolumeProfileNodePool::getInstance() {
    static FastVolumeProfileNodePool instance;
    return instance;
}

VolumeProfileNode* FastVolumeProfileNodePool::allocate() {
    return pool_.allocate();
}

void FastVolumeProfileNodePool::deallocate(VolumeProfileNode* node) {
    pool_.deallocate(node);
}

void FastVolumeProfileNodePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastFootprintCellPool implementation
FastFootprintCellPool& FastFootprintCellPool::getInstance() {
    static FastFootprintCellPool instance;
    return instance;
}

FootprintCell* FastFootprintCellPool::allocate() {
    return pool_.allocate();
}

void FastFootprintCellPool::deallocate(FootprintCell* cell) {
    pool_.deallocate(cell);
}

void FastFootprintCellPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// Additional memory pool implementations for other frequently allocated objects

// TradePool implementation
TradePool& TradePool::getInstance() {
    static TradePool instance;
    return instance;
}

BTQuant::Trade* TradePool::allocate() {
    return pool_.allocate();
}

void TradePool::deallocate(BTQuant::Trade* trade) {
    pool_.deallocate(trade);
}

void TradePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastTradePool implementation
FastTradePool& FastTradePool::getInstance() {
    static FastTradePool instance;
    return instance;
}

BTQuant::Trade* FastTradePool::allocate() {
    return pool_.allocate();
}

void FastTradePool::deallocate(BTQuant::Trade* trade) {
    pool_.deallocate(trade);
}

void FastTradePool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// OrderBookSnapshotPool implementation
OrderBookSnapshotPool& OrderBookSnapshotPool::getInstance() {
    static OrderBookSnapshotPool instance;
    return instance;
}

BTQuant::OrderBookSnapshot* OrderBookSnapshotPool::allocate() {
    return pool_.allocate();
}

void OrderBookSnapshotPool::deallocate(BTQuant::OrderBookSnapshot* snapshot) {
    pool_.deallocate(snapshot);
}

void OrderBookSnapshotPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastOrderBookSnapshotPool implementation
FastOrderBookSnapshotPool& FastOrderBookSnapshotPool::getInstance() {
    static FastOrderBookSnapshotPool instance;
    return instance;
}

BTQuant::OrderBookSnapshot* FastOrderBookSnapshotPool::allocate() {
    return pool_.allocate();
}

void FastOrderBookSnapshotPool::deallocate(BTQuant::OrderBookSnapshot* snapshot) {
    pool_.deallocate(snapshot);
}

void FastOrderBookSnapshotPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// IndicatorResultPool implementation
IndicatorResultPool& IndicatorResultPool::getInstance() {
    static IndicatorResultPool instance;
    return instance;
}

BTQuant::TechnicalIndicators::IndicatorResult* IndicatorResultPool::allocate() {
    return pool_.allocate();
}

void IndicatorResultPool::deallocate(BTQuant::TechnicalIndicators::IndicatorResult* result) {
    pool_.deallocate(result);
}

void IndicatorResultPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastIndicatorResultPool implementation
FastIndicatorResultPool& FastIndicatorResultPool::getInstance() {
    static FastIndicatorResultPool instance;
    return instance;
}

BTQuant::TechnicalIndicators::IndicatorResult* FastIndicatorResultPool::allocate() {
    return pool_.allocate();
}

void FastIndicatorResultPool::deallocate(BTQuant::TechnicalIndicators::IndicatorResult* result) {
    pool_.deallocate(result);
}

void FastIndicatorResultPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// OrderExecutionPool implementation
OrderExecutionPool& OrderExecutionPool::getInstance() {
    static OrderExecutionPool instance;
    return instance;
}

OrderManager::OrderExecution* OrderExecutionPool::allocate() {
    return pool_.allocate();
}

void OrderExecutionPool::deallocate(OrderManager::OrderExecution* execution) {
    pool_.deallocate(execution);
}

void OrderExecutionPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastOrderExecutionPool implementation
FastOrderExecutionPool& FastOrderExecutionPool::getInstance() {
    static FastOrderExecutionPool instance;
    return instance;
}

OrderManager::OrderExecution* FastOrderExecutionPool::allocate() {
    return pool_.allocate();
}

void FastOrderExecutionPool::deallocate(OrderManager::OrderExecution* execution) {
    pool_.deallocate(execution);
}

void FastOrderExecutionPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// TradePaceDataPool implementation
TradePaceDataPool& TradePaceDataPool::getInstance() {
    static TradePaceDataPool instance;
    return instance;
}

BTQuant::TapePanel::TradePaceData* TradePaceDataPool::allocate() {
    return pool_.allocate();
}

void TradePaceDataPool::deallocate(BTQuant::TapePanel::TradePaceData* data) {
    pool_.deallocate(data);
}

void TradePaceDataPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastTradePaceDataPool implementation
FastTradePaceDataPool& FastTradePaceDataPool::getInstance() {
    static FastTradePaceDataPool instance;
    return instance;
}

BTQuant::TapePanel::TradePaceData* FastTradePaceDataPool::allocate() {
    return pool_.allocate();
}

void FastTradePaceDataPool::deallocate(BTQuant::TapePanel::TradePaceData* data) {
    pool_.deallocate(data);
}

void FastTradePaceDataPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// TradePairPool implementation
TradePairPool& TradePairPool::getInstance() {
    static TradePairPool instance;
    return instance;
}

std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>* TradePairPool::allocate() {
    return pool_.allocate();
}

void TradePairPool::deallocate(std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>* pair) {
    pool_.deallocate(pair);
}

void TradePairPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastTradePairPool implementation
FastTradePairPool& FastTradePairPool::getInstance() {
    static FastTradePairPool instance;
    return instance;
}

std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>* FastTradePairPool::allocate() {
    return pool_.allocate();
}

void FastTradePairPool::deallocate(std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>* pair) {
    pool_.deallocate(pair);
}

void FastTradePairPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// IndicatorValuePairPool implementation
IndicatorValuePairPool& IndicatorValuePairPool::getInstance() {
    static IndicatorValuePairPool instance;
    return instance;
}

std::pair<float, float>* IndicatorValuePairPool::allocate() {
    return pool_.allocate();
}

void IndicatorValuePairPool::deallocate(std::pair<float, float>* pair) {
    pool_.deallocate(pair);
}

void IndicatorValuePairPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastIndicatorValuePairPool implementation
FastIndicatorValuePairPool& FastIndicatorValuePairPool::getInstance() {
    static FastIndicatorValuePairPool instance;
    return instance;
}

std::pair<float, float>* FastIndicatorValuePairPool::allocate() {
    return pool_.allocate();
}

void FastIndicatorValuePairPool::deallocate(std::pair<float, float>* pair) {
    pool_.deallocate(pair);
}

void FastIndicatorValuePairPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// PSARIndicatorPool implementation
PSARIndicatorPool& PSARIndicatorPool::getInstance() {
    static PSARIndicatorPool instance;
    return instance;
}

PSARIndicator* PSARIndicatorPool::allocate(float acceleration_step, float acceleration_max) {
    return pool_.allocate(acceleration_step, acceleration_max);
}

void PSARIndicatorPool::deallocate(PSARIndicator* indicator) {
    pool_.deallocate(indicator);
}

void PSARIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastPSARIndicatorPool implementation
FastPSARIndicatorPool& FastPSARIndicatorPool::getInstance() {
    static FastPSARIndicatorPool instance;
    return instance;
}

PSARIndicator* FastPSARIndicatorPool::allocate(float acceleration_step, float acceleration_max) {
    return pool_.allocate(acceleration_step, acceleration_max);
}

void FastPSARIndicatorPool::deallocate(PSARIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastPSARIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// CCIIndicatorPool implementation
CCIIndicatorPool& CCIIndicatorPool::getInstance() {
    static CCIIndicatorPool instance;
    return instance;
}

CCIIndicator* CCIIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void CCIIndicatorPool::deallocate(CCIIndicator* indicator) {
    pool_.deallocate(indicator);
}

void CCIIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastCCIIndicatorPool implementation
FastCCIIndicatorPool& FastCCIIndicatorPool::getInstance() {
    static FastCCIIndicatorPool instance;
    return instance;
}

CCIIndicator* FastCCIIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void FastCCIIndicatorPool::deallocate(CCIIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastCCIIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// WilliamsRIndicatorPool implementation
WilliamsRIndicatorPool& WilliamsRIndicatorPool::getInstance() {
    static WilliamsRIndicatorPool instance;
    return instance;
}

WilliamsRIndicator* WilliamsRIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void WilliamsRIndicatorPool::deallocate(WilliamsRIndicator* indicator) {
    pool_.deallocate(indicator);
}

void WilliamsRIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

// FastWilliamsRIndicatorPool implementation
FastWilliamsRIndicatorPool& FastWilliamsRIndicatorPool::getInstance() {
    static FastWilliamsRIndicatorPool instance;
    return instance;
}

WilliamsRIndicator* FastWilliamsRIndicatorPool::allocate(int period) {
    return pool_.allocate(period);
}

void FastWilliamsRIndicatorPool::deallocate(WilliamsRIndicator* indicator) {
    pool_.deallocate(indicator);
}

void FastWilliamsRIndicatorPool::preallocate(size_t count) {
    pool_.preallocate(count);
}

} // namespace BTQuant