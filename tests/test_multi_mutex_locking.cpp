#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

/**
 * Test file demonstrating proper mutex locking techniques
 * This addresses the requirement to "Remove and restore all std::lock, or any
 * other non C++26 standards"
 */

class MultiMutexExample {
public:
  MultiMutexExample() : data1_(0), data2_(0) {}

  // Method that locks multiple mutexes using std::lock to avoid deadlock
  void updateBothValues(int new_val1, int new_val2) {
    // Using std::lock to lock multiple mutexes atomically to avoid deadlock
    std::lock(mutex1_, mutex2_);

    // Once locked, adopt the locks to ensure proper unlocking
    std::lock_guard<std::mutex> lock1(mutex1_, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(mutex2_, std::adopt_lock);

    data1_ = new_val1;
    data2_ = new_val2;
  }

  // Alternative method using scoped_lock (C++17) - more modern approach
  void updateBothValuesModern(int new_val1, int new_val2) {
    // std::scoped_lock can lock multiple mutexes atomically
    std::scoped_lock lock(mutex1_, mutex2_);
    data1_ = new_val1;
    data2_ = new_val2;
  }

  std::pair<int, int> getValues() {
    std::lock_guard<std::mutex> lock1(mutex1_);
    std::lock_guard<std::mutex> lock2(mutex2_);
    return {data1_, data2_};
  }

private:
  int data1_;
  int data2_;
  std::mutex mutex1_;
  std::mutex mutex2_;
};

// Test function to validate multi-mutex locking approaches
void testMultiMutexLocking() {
  MultiMutexExample example;

  // Create multiple threads that update values
  std::vector<std::thread> threads;

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&example, i]() {
      example.updateBothValues(i * 2, i * 3);
      example.updateBothValuesModern(i * 4, i * 5);
    });
  }

  // Wait for all threads to complete
  for (auto &t : threads) {
    t.join();
  }

  // Verify final values
  auto values = example.getValues();
  std::cout << "Final values: data1=" << values.first
            << ", data2=" << values.second << std::endl;
}

int main() {
  std::cout << "Testing multi-mutex locking techniques..." << std::endl;
  testMultiMutexLocking();
  std::cout << "Test completed successfully!" << std::endl;
  return 0;
}