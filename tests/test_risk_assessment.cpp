#include "trading/risk_assessment.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <vector>

void test_daily_loss_limit_check() {
    std::cout << "Testing Daily Loss Limit Check...\n";

    RiskAssessment risk(10000.0, 1000.0);  // $10k daily loss limit

    // Order that should pass
    Order small_order(10, 100.0, true, "TEST");
    assert(risk.check_daily_loss_limit(small_order) == RiskAssessmentResult::APPROVED);
    std::cout << "  Small order approved: PASS\n";

    // Simulate losses (leave room for small order)
    risk.update_daily_pnl(-8500.0);

    // Small order should still pass (8500 + 1000 = 9500 < 10000)
    assert(risk.check_daily_loss_limit(small_order) == RiskAssessmentResult::APPROVED);
    std::cout << "  Order with remaining limit: PASS\n";

    // Large order that would exceed limit
    Order large_order(200, 100.0, false, "TEST");  // $20k potential impact -> 8500 + 20000 > 10000
    assert(risk.check_daily_loss_limit(large_order) == RiskAssessmentResult::DAILY_LOSS_LIMIT_EXCEEDED);
    std::cout << "  Large order exceeding limit rejected: PASS\n";

    std::cout << "Daily Loss Limit Check: ALL TESTS PASSED\n\n";
}

void test_max_position_size_check() {
    std::cout << "Testing Max Position Size Check...\n";

    RiskAssessment risk(10000.0, 100.0);  // 100 units max position

    // Order that should pass
    Order small_order(10, 100.0, true, "TEST");
    assert(risk.check_max_position_size(small_order) == RiskAssessmentResult::APPROVED);
    std::cout << "  Small order approved: PASS\n";

    // Build up position
    risk.update_position_size(85.0);

    // Order that would exceed max
    Order large_order(20, 100.0, true, "TEST");  // Would make position 105
    assert(risk.check_max_position_size(large_order) == RiskAssessmentResult::MAX_POSITION_SIZE_EXCEEDED);
    std::cout << "  Order exceeding max position rejected: PASS\n";

    // Sell order that reduces position should pass
    Order sell_order(10, 100.0, false, "TEST");
    assert(risk.check_max_position_size(sell_order) == RiskAssessmentResult::APPROVED);
    std::cout << "  Sell order reducing position approved: PASS\n";

    std::cout << "Max Position Size Check: ALL TESTS PASSED\n\n";
}

void test_combined_assessment() {
    std::cout << "Testing Combined Order Assessment...\n";

    RiskAssessment risk(10000.0, 100.0);

    // Order that passes both checks
    Order good_order(10, 100.0, true, "TEST");
    assert(risk.assess_order(good_order) == RiskAssessmentResult::APPROVED);
    std::cout << "  Good order approved: PASS\n";

    // Set up daily loss limit violation
    risk.update_daily_pnl(-9500.0);
    Order loss_violating_order(100, 100.0, false, "TEST");  // Would exceed limit
    assert(risk.assess_order(loss_violating_order) == RiskAssessmentResult::DAILY_LOSS_LIMIT_EXCEEDED);
    std::cout << "  Daily loss limit violation detected: PASS\n";

    // Reset and test position size violation
    risk.reset_daily_pnl();
    risk.update_position_size(95.0);
    Order position_violating_order(10, 100.0, true, "TEST");
    assert(risk.assess_order(position_violating_order) == RiskAssessmentResult::MAX_POSITION_SIZE_EXCEEDED);
    std::cout << "  Max position size violation detected: PASS\n";

    std::cout << "Combined Order Assessment: ALL TESTS PASSED\n\n";
}

void test_thread_safety() {
    std::cout << "Testing Thread Safety (Lock-free Operations)...\n";

    RiskAssessment risk(100000.0, 10000.0);

    const int num_threads = 10;
    const int updates_per_thread = 1000;

    std::vector<std::thread> threads;

    // Spawn multiple threads updating P&L concurrently
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&risk, updates_per_thread]() {
            for (int j = 0; j < updates_per_thread; ++j) {
                risk.update_daily_pnl(1.0);
                risk.update_position_size(1.0);  // Use integer values to avoid FP issues
            }
        });
    }

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }

    // Verify final values
    double expected_pnl = static_cast<double>(num_threads * updates_per_thread);
    double expected_position = static_cast<double>(num_threads * updates_per_thread);

    assert(risk.get_current_daily_pnl() == expected_pnl);
    assert(risk.get_current_position_size() == expected_position);

    std::cout << "  Concurrent P&L updates: " << risk.get_current_daily_pnl() << " (expected: " << expected_pnl << ")\n";
    std::cout << "  Concurrent position updates: " << risk.get_current_position_size() << " (expected: " << expected_position << ")\n";
    std::cout << "Thread Safety: ALL TESTS PASSED\n\n";
}

void test_daily_reset() {
    std::cout << "Testing Daily Reset Mechanism...\n";

    RiskAssessment risk(10000.0, 100.0);

    // Simulate some P&L
    risk.update_daily_pnl(-5000.0);
    assert(risk.get_current_daily_pnl() == -5000.0);
    std::cout << "  P&L set to -5000: PASS\n";

    // Manual reset
    risk.reset_daily_pnl();
    assert(risk.get_current_daily_pnl() == 0.0);
    std::cout << "  Manual reset works: PASS\n";

    std::cout << "Daily Reset Mechanism: ALL TESTS PASSED\n\n";
}

void test_config_updates() {
    std::cout << "Testing Configuration Updates...\n";

    RiskAssessment risk(10000.0, 100.0);

    assert(risk.get_daily_loss_limit() == 10000.0);
    assert(risk.get_max_position_size() == 100.0);
    std::cout << "  Initial config correct: PASS\n";

    risk.set_daily_loss_limit(50000.0);
    risk.set_max_position_size(500.0);

    assert(risk.get_daily_loss_limit() == 50000.0);
    assert(risk.get_max_position_size() == 500.0);
    std::cout << "  Config updates work: PASS\n";

    std::cout << "Configuration Updates: ALL TESTS PASSED\n\n";
}

int main() {
    std::cout << "=== Risk Assessment Module Tests ===\n\n";

    test_daily_loss_limit_check();
    test_max_position_size_check();
    test_combined_assessment();
    test_thread_safety();
    test_daily_reset();
    test_config_updates();

    std::cout << "=== ALL TESTS PASSED ===\n";
    return 0;
}
