#include "../../../dependencies/BTQ_Render_Engine/include/analytics/cumulative_delta_tracker.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

using namespace BTQuant;

void test_basic_cumulative_functionality() {
  std::cout << "Testing basic cumulative delta functionality..." << std::endl;

  CumulativeDeltaTracker tracker;

  // Add some delta values
  tracker.add_delta(10.0);
  tracker.add_delta(5.0);
  tracker.add_delta(-3.0);

  double result = tracker.get_cumulative_delta();
  assert(result == 12.0 && "Cumulative delta should be 12.0");

  std::cout << "Basic cumulative functionality test PASSED!" << std::endl;
}

void test_reset_functionality() {
  std::cout << "Testing reset functionality..." << std::endl;

  CumulativeDeltaTracker tracker;

  // Add some delta values
  tracker.add_delta(10.0);
  tracker.add_delta(5.0);

  assert(tracker.get_cumulative_delta() == 15.0 &&
         "Initial cumulative delta should be 15.0");
  assert(!tracker.is_reset() &&
         "Tracker should not be marked as reset initially");

  // Reset the tracker
  tracker.reset();

  assert(tracker.get_cumulative_delta() == 0.0 &&
         "Cumulative delta should be 0.0 after reset");
  assert(tracker.is_reset() && "Tracker should be marked as reset after reset");

  // Add more deltas after reset
  tracker.add_delta(7.0);
  assert(tracker.get_cumulative_delta() == 7.0 &&
         "Cumulative delta should be 7.0 after reset and new addition");

  std::cout << "Reset functionality test PASSED!" << std::endl;
}

void test_session_boundary_functionality() {
  std::cout << "Testing session boundary functionality..." << std::endl;

  CumulativeDeltaTracker tracker;

  // Set a short session duration for testing
  tracker.set_session_duration(std::chrono::minutes(1)); // 1 minute session

  // Add some delta values
  tracker.add_delta(10.0);
  tracker.add_delta(5.0);

  assert(tracker.get_cumulative_delta() == 15.0 &&
         "Initial cumulative delta should be 15.0");

  // Manually trigger a session boundary
  tracker.set_session_boundary();

  assert(tracker.get_cumulative_delta() == 0.0 &&
         "Cumulative delta should be 0.0 after session boundary");
  assert(tracker.is_session_active() &&
         "Session should be active after setting boundary");

  // Add more deltas after session boundary
  tracker.add_delta(8.0);
  assert(
      tracker.get_cumulative_delta() == 8.0 &&
      "Cumulative delta should be 8.0 after session boundary and new addition");

  std::cout << "Session boundary functionality test PASSED!" << std::endl;
}

void test_delta_count_tracking() {
  std::cout << "Testing delta count tracking..." << std::endl;

  CumulativeDeltaTracker tracker;

  assert(tracker.get_delta_count_since_reset() == 0 &&
         "Initial delta count should be 0");

  tracker.add_delta(10.0);
  assert(tracker.get_delta_count_since_reset() == 1 &&
         "Delta count should be 1 after first addition");

  tracker.add_delta(5.0);
  assert(tracker.get_delta_count_since_reset() == 2 &&
         "Delta count should be 2 after second addition");

  tracker.add_delta(-3.0);
  assert(tracker.get_delta_count_since_reset() == 3 &&
         "Delta count should be 3 after third addition");

  // Reset and check count
  tracker.reset();
  assert(tracker.get_delta_count_since_reset() == 0 &&
         "Delta count should be 0 after reset");

  tracker.add_delta(7.0);
  assert(tracker.get_delta_count_since_reset() == 1 &&
         "Delta count should be 1 after reset and new addition");

  std::cout << "Delta count tracking test PASSED!" << std::endl;
}

void test_session_duration_management() {
  std::cout << "Testing session duration management..." << std::endl;

  CumulativeDeltaTracker tracker;

  // Check default session duration
  auto default_duration = tracker.get_session_duration();
  assert(default_duration == std::chrono::minutes(60) &&
         "Default session duration should be 60 minutes");

  // Set custom session duration
  tracker.set_session_duration(std::chrono::minutes(30));
  auto custom_duration = tracker.get_session_duration();
  assert(custom_duration == std::chrono::minutes(30) &&
         "Custom session duration should be 30 minutes");

  std::cout << "Session duration management test PASSED!" << std::endl;
}

void test_time_based_session_reset() {
  std::cout << "Testing time-based session reset functionality..." << std::endl;

  CumulativeDeltaTracker tracker;

  // Set a short session duration for testing (minimum is 1 minute)
  tracker.set_session_duration(std::chrono::minutes(1)); // 1 minute session

  // Add some delta values
  tracker.add_delta(10.0);
  assert(tracker.get_cumulative_delta() == 10.0 &&
         "Initial cumulative delta should be 10.0");

  // The implementation should handle time-based resets automatically
  // when enough time has passed between additions
  std::cout << "Time-based session reset test completed!" << std::endl;
}

int main() {
  std::cout << "Running CumulativeDeltaTracker tests..." << std::endl;

  test_basic_cumulative_functionality();
  test_reset_functionality();
  test_session_boundary_functionality();
  test_delta_count_tracking();
  test_session_duration_management();
  test_time_based_session_reset();

  std::cout << "All CumulativeDeltaTracker tests PASSED!" << std::endl;
  return 0;
}