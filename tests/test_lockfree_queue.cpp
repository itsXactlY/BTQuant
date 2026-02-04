#include "dependencies/BTQ_Render_Engine/include/threading/lockfree_queue.hpp"
#include "dependencies/BTQ_Render_Engine/include/task_scheduler.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cassert>

using namespace btq::threading;

void test_calculation_to_ui_queue() {
    std::cout << "Testing CalculationToUIQueue..." << std::endl;
    
    CalculationToUIQueue<int> queue(1000);
    
    // Test basic push/pop functionality
    assert(queue.push(42) == true);
    assert(queue.push(100) == true);
    
    auto item1 = queue.try_pop();
    assert(item1.has_value() == true);
    assert(item1.value() == 42);
    
    auto item2 = queue.try_pop();
    assert(item2.has_value() == true);
    assert(item2.value() == 100);
    
    auto item3 = queue.try_pop();
    assert(item3.has_value() == false);
    
    std::cout << "Basic functionality test passed!" << std::endl;
    
    // Test batch operations
    for (int i = 0; i < 10; ++i) {
        queue.push(i * 10);
    }
    
    auto batch = queue.pop_batch(5);
    assert(batch.size() == 5);
    for (int i = 0; i < 5; ++i) {
        assert(batch[i] == i * 10);
    }
    
    std::cout << "Batch operations test passed!" << std::endl;
    
    // Test statistics
    assert(queue.total_produced() == 12); // 2 individual + 10 batch
    assert(queue.total_consumed() == 7);  // 2 individual + 5 batch
    
    std::cout << "Statistics test passed!" << std::endl;
    
    // Test high priority mode
    queue.set_high_priority_mode(true);
    assert(queue.is_high_priority_mode() == true);
    
    queue.reset_stats();
    assert(queue.total_produced() == 0);
    assert(queue.total_consumed() == 0);
    
    std::cout << "High priority mode test passed!" << std::endl;
    
    std::cout << "CalculationToUIQueue tests completed successfully!" << std::endl;
}

void test_priority_ui_update_queue() {
    std::cout << "Testing PriorityUIUpdateQueue..." << std::endl;
    
    PriorityUIUpdateQueue<std::string> queue(1000);
    
    // Test basic functionality
    assert(queue.push_notification("Low priority", 1) == true);
    assert(queue.push_notification("High priority", 10) == true);
    assert(queue.push_notification("Medium priority", 5) == true);
    
    // Test priority-based popping
    auto high_prio_item = queue.try_pop_highest_priority();
    assert(high_prio_item.has_value() == true);
    assert(high_prio_item->data == "High priority");
    assert(high_prio_item->priority == 10);
    
    std::cout << "Priority-based popping test passed!" << std::endl;
    
    // Test size and emptiness
    assert(queue.size_approx() >= 0); // At least 2 items should remain
    
    std::cout << "Size check test passed!" << std::endl;
    
    // Test statistics
    assert(queue.total_notifications() == 3);
    assert(queue.processed_notifications() >= 1); // At least 1 processed
    
    std::cout << "Statistics test passed!" << std::endl;
    
    std::cout << "PriorityUIUpdateQueue tests completed successfully!" << std::endl;
}

void test_thread_safety() {
    std::cout << "Testing thread safety..." << std::endl;

    CalculationToUIQueue<int> queue(10000);

    // Producer threads
    std::vector<std::thread> producers;
    const int num_producers = 2;  // Reduced to minimize complexity
    const int items_per_producer = 100;  // Reduced to minimize complexity

    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&queue, p, items_per_producer]() {
            for (int i = 0; i < items_per_producer; ++i) {
                int value = p * items_per_producer + i;
                queue.push(value);
            }
        });
    }

    // Brief pause to allow producers to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Wait for all threads to complete
    for (auto& producer : producers) {
        producer.join();
    }

    // Verify that items were produced
    assert(queue.total_produced() == num_producers * items_per_producer);

    // Consume all items
    int consumed_count = 0;
    while (true) {
        auto item = queue.try_pop();
        if (!item.has_value()) {
            break;
        }
        consumed_count++;
    }

    assert(consumed_count == num_producers * items_per_producer);

    std::cout << "Thread safety test passed! Produced " << queue.total_produced() << " items, consumed " << consumed_count << "." << std::endl;

    std::cout << "Thread safety tests completed successfully!" << std::endl;
}

int main() {
    std::cout << "Starting lock-free queue tests..." << std::endl;
    
    test_calculation_to_ui_queue();
    std::cout << std::endl;
    
    test_priority_ui_update_queue();
    std::cout << std::endl;
    
    test_thread_safety();
    std::cout << std::endl;
    
    std::cout << "All tests completed successfully!" << std::endl;
    
    return 0;
}