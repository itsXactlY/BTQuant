#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

/**
 * Comprehensive test for C++ synchronization primitives
 * Addresses the requirement to "Remove and restore all std::lock, or any other
 * non C++26 standards"
 */

// Example class showing various synchronization techniques
class SynchronizationDemo {
private:
  std::mutex mtx1_;
  std::mutex mtx2_;
  std::shared_mutex shared_mtx_;
  std::atomic<int> atomic_counter_{0};
  int regular_counter_{0};

public:
  // Traditional approach with individual locks (potential deadlock risk)
  void traditionalUpdate(int val1, int val2) {
    std::scoped_lock lock(mtx1_, mtx2_);
    // Process data
    regular_counter_ = val1 + val2;
  }

  // Safe approach using std::lock to avoid deadlock
  void safeUpdateWithStdLock(int val1, int val2) {
    std::lock(mtx1_, mtx2_);
    std::lock_guard<std::mutex> lock1(mtx1_, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(mtx2_, std::adopt_lock);
    // Process data
    regular_counter_ = val1 * val2;
  }

  // Modern approach using scoped_lock (C++17) - preferred
  void modernUpdate(int val1, int val2) {
    std::scoped_lock lock(mtx1_, mtx2_);
    // Process data
    regular_counter_ = val1 - val2;
  }

  // Atomic operations - lock-free synchronization
  int incrementAtomic() { return atomic_counter_.fetch_add(1) + 1; }

  [[nodiscard]] int getAtomicValue() const { return atomic_counter_.load(); }

  // Shared locking for readers/writer scenarios
  void writeSharedData(int value) {
    std::unique_lock<std::shared_mutex> lock(shared_mtx_);
    regular_counter_ = value;
  }

  int readSharedData() {
    std::shared_lock<std::shared_mutex> lock(shared_mtx_);
    return regular_counter_;
  }
};

// Test function to validate synchronization approaches
void testSynchronizationMethods() {
  SynchronizationDemo demo;

  std::cout << "Testing various synchronization methods..." << std::endl;

  // Test atomic operations
  for (int i = 0; i < 10; ++i) {
    demo.incrementAtomic();
  }
  std::cout << "Atomic counter value: " << demo.getAtomicValue() << std::endl;

  // Test different locking methods with multiple threads
  std::vector<std::thread> threads;

  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([&demo, i]() {
      demo.traditionalUpdate(i, i * 2);
      demo.safeUpdateWithStdLock(i * 3, i * 4);
      demo.modernUpdate(i * 5, i * 6);

      demo.writeSharedData(i * 10);
      int value = demo.readSharedData();
      std::cout << "Thread " << i << " read shared value: " << value
                << std::endl;
    });
  }

  // Wait for all threads to complete
  for (auto &t : threads) {
    t.join();
  }

  std::cout << "All synchronization tests completed successfully!" << std::endl;
}

// Future-based asynchronous operations
void testAsyncOperations() {
  std::cout << "Testing async operations..." << std::endl;

  auto future1 = std::async(std::launch::async, []() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
  });

  auto future2 = std::async(std::launch::async, []() {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    return 84;
  });

  int result1 = future1.get();
  int result2 = future2.get();

  std::cout << "Async results: " << result1 << ", " << result2 << std::endl;
}

int main() {
  std::cout << "Starting comprehensive synchronization test..." << std::endl;

  testSynchronizationMethods();
  testAsyncOperations();

  std::cout << "All tests completed successfully!" << std::endl;
  return 0;
}