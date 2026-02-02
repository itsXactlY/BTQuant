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

} // namespace BTQuant