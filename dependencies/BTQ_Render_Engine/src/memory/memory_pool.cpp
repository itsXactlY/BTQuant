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

} // namespace BTQuant