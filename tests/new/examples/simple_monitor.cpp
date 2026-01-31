#include "detectors/stop_hunt_detector.hpp"
#include "hotspine_extended_reader.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

using namespace BTQuant;

std::atomic<bool> running{true};

void signal_handler(int signal) {
  if (signal == SIGINT) {
    running = false;
  }
}

int main() {
  std::signal(SIGINT, signal_handler);

  std::cout << "Simple Stop Hunt Monitor\n";
  std::cout << "========================\n\n";

  try {
    HotSpineExtendedReader reader("/btquant_hotspine");

    if (!reader.is_attached()) {
      std::cerr << "Failed to attach to HotSpine\n";
      return 1;
    }

    reader.load_symbol_mappings("../config/symbol_mapping.json");

    StopHuntDetector detector(reader);
    detector.set_threshold_pct(0.5); // 0.5%

    std::cout << "Monitoring BTC-USDT for stop hunts...\n\n";

    size_t trade_count = 0;

    while (running) {
      // Process trades
      while (auto trade = reader.poll_trade()) {
        trade_count++;

        if (trade_count % 100 == 0) {
          std::cout << "Processed " << trade_count << " trades\r" << std::flush;
        }
      }

      // Check for stop hunts
      if (auto signal = detector.detect("BTC-USDT")) {
        std::cout << "\n🚨 STOP HUNT DETECTED!\n";
        std::cout << "   Symbol: " << signal->symbol << "\n";
        std::cout << "   Exchange: " << signal->hunt_exchange << "\n";
        std::cout << "   Price: $" << signal->hunt_price << "\n";
        std::cout << "   Median: $" << signal->median_price << "\n";
        std::cout << "   Deviation: " << signal->hunt_deviation_pct << "%\n";
        std::cout << "   Signal: "
                  << (signal->is_long_signal ? "LONG ⬆️" : "SHORT ⬇️") << "\n";
        std::cout << "   Stable exchanges:\n";
        for (const auto &[ex, price] : signal->stable_exchanges) {
          std::cout << "     - " << ex << ": $" << price << "\n";
        }
        std::cout << "\n";
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nTotal trades processed: " << trade_count << "\n";
    std::cout << "Total stop hunts detected: " << detector.get_detections()
              << "\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}