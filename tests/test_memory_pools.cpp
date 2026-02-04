#include <iostream>
#include "dependencies/BTQ_Render_Engine/include/memory/memory_pool.hpp"

int main() {
    std::cout << "Testing new memory pools..." << std::endl;

    // Test TradePool
    auto& trade_pool = BTQuant::TradePool::getInstance();
    auto* trade = trade_pool.allocate();
    if (trade) {
        std::cout << "Successfully allocated trade from TradePool" << std::endl;
        trade_pool.deallocate(trade);
        std::cout << "Successfully deallocated trade" << std::endl;
    } else {
        std::cout << "Failed to allocate trade from TradePool" << std::endl;
    }

    // Test FastTradePool
    auto& fast_trade_pool = BTQuant::FastTradePool::getInstance();
    auto* fast_trade = fast_trade_pool.allocate();
    if (fast_trade) {
        std::cout << "Successfully allocated trade from FastTradePool" << std::endl;
        fast_trade_pool.deallocate(fast_trade);
        std::cout << "Successfully deallocated fast trade" << std::endl;
    } else {
        std::cout << "Failed to allocate trade from FastTradePool" << std::endl;
    }

    // Test IndicatorResultPool
    auto& indicator_result_pool = BTQuant::IndicatorResultPool::getInstance();
    auto* indicator_result = indicator_result_pool.allocate();
    if (indicator_result) {
        std::cout << "Successfully allocated indicator result from IndicatorResultPool" << std::endl;
        indicator_result_pool.deallocate(indicator_result);
        std::cout << "Successfully deallocated indicator result" << std::endl;
    } else {
        std::cout << "Failed to allocate indicator result from IndicatorResultPool" << std::endl;
    }

    // Test FastIndicatorResultPool
    auto& fast_indicator_result_pool = BTQuant::FastIndicatorResultPool::getInstance();
    auto* fast_indicator_result = fast_indicator_result_pool.allocate();
    if (fast_indicator_result) {
        std::cout << "Successfully allocated indicator result from FastIndicatorResultPool" << std::endl;
        fast_indicator_result_pool.deallocate(fast_indicator_result);
        std::cout << "Successfully deallocated fast indicator result" << std::endl;
    } else {
        std::cout << "Failed to allocate indicator result from FastIndicatorResultPool" << std::endl;
    }

    // Test HotOrderbookSnapshotPool
    auto& snapshot_pool = BTQuant::HotOrderbookSnapshotPool::getInstance();
    auto* snapshot = snapshot_pool.allocate();
    if (snapshot) {
        std::cout << "Successfully allocated snapshot from HotOrderbookSnapshotPool" << std::endl;
        snapshot_pool.deallocate(snapshot);
        std::cout << "Successfully deallocated snapshot" << std::endl;
    } else {
        std::cout << "Failed to allocate snapshot from HotOrderbookSnapshotPool" << std::endl;
    }

    // Test FastHotOrderbookSnapshotPool
    auto& fast_snapshot_pool = BTQuant::FastHotOrderbookSnapshotPool::getInstance();
    auto* fast_snapshot = fast_snapshot_pool.allocate();
    if (fast_snapshot) {
        std::cout << "Successfully allocated snapshot from FastHotOrderbookSnapshotPool" << std::endl;
        fast_snapshot_pool.deallocate(fast_snapshot);
        std::cout << "Successfully deallocated fast snapshot" << std::endl;
    } else {
        std::cout << "Failed to allocate snapshot from FastHotOrderbookSnapshotPool" << std::endl;
    }

    // Test OrderExecutionPool
    auto& execution_pool = BTQuant::OrderExecutionPool::getInstance();
    auto* execution = execution_pool.allocate();
    if (execution) {
        std::cout << "Successfully allocated execution from OrderExecutionPool" << std::endl;
        execution_pool.deallocate(execution);
        std::cout << "Successfully deallocated execution" << std::endl;
    } else {
        std::cout << "Failed to allocate execution from OrderExecutionPool" << std::endl;
    }

    // Test FastOrderExecutionPool
    auto& fast_execution_pool = BTQuant::FastOrderExecutionPool::getInstance();
    auto* fast_execution = fast_execution_pool.allocate();
    if (fast_execution) {
        std::cout << "Successfully allocated execution from FastOrderExecutionPool" << std::endl;
        fast_execution_pool.deallocate(fast_execution);
        std::cout << "Successfully deallocated fast execution" << std::endl;
    } else {
        std::cout << "Failed to allocate execution from FastOrderExecutionPool" << std::endl;
    }

    std::cout << "All memory pool tests completed!" << std::endl;
    return 0;
}