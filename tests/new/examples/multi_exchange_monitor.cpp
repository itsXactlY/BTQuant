#include "detectors/spread_arbitrage_detector.hpp"
#include "detectors/stop_hunt_detector.hpp"
#include "hotspine_extended_reader.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <thread>

using namespace BTQuant;

std::atomic<bool> running{true};

void signal_handler(int signal) {
  if (signal == SIGINT) {
    running = false;
  }
}

void print_price_comparison(const std::map<std::string, double> &prices,
                            const std::string &symbol) {
  if (prices.empty())
    return;

  // Find min and max
  double min_price = std::numeric_limits<double>::max();
  double max_price = 0.0;
  std::string min_exchange, max_exchange;

  for (const auto &[exchange, price] : prices) {
    if (price < min_price) {
      min_price = price;
      min_exchange = exchange;
    }
    if (price > max_price) {
      max_price = price;
      max_exchange = exchange;
    }
  }

  double spread = ((max_price - min_price) / min_price) * 10000.0;

  std::cout << "\n┌─────────────────────────────────────────┐\n";
  std::cout << "│ " << std::left << std::setw(39) << symbol << " │\n";
  std::cout << "├─────────────────────────────────────────┤\n";

  for (const auto &[exchange, price] : prices) {
    std::cout << "│ " << std::left << std::setw(15) << exchange << " $"
              << std::right << std::setw(20) << std::fixed
              << std::setprecision(2) << price << " │\n";
  }

  std::cout << "├─────────────────────────────────────────┤\n";
  std::cout << "│ Spread: " << std::setw(28) << std::fixed
            << std::setprecision(2) << spread << " bps │\n";
  std::cout << "│ Min: " << std::setw(15) << min_exchange << " $"
            << std::setw(15) << min_price << " │\n";
  std::cout << "│ Max: " << std::setw(15) << max_exchange << " $"
            << std::setw(15) << max_price << " │\n";
  std::cout << "└─────────────────────────────────────────┘\n";
}

int main() {
  std::signal(SIGINT, signal_handler);

  std::cout << "Multi-Exchange Price Monitor\n";
  std::cout << "============================\n\n";

  try {
    HotSpineExtendedReader reader("/btquant_hotspine");

    if (!reader.is_attached()) {
      std::cerr << "Failed to attach to HotSpine\n";
      return 1;
    }

    reader.load_symbol_mappings("../config/symbol_mapping.json");

    StopHuntDetector stop_hunt(reader);
    SpreadArbitrageDetector arbitrage(reader);
    arbitrage.set_min_profit_bps(30); // 0.3% minimum

    std::vector<std::string> symbols = {"BTC-USDT", "ETH-USDT", "SOL-USDT"};

    std::cout << "Monitoring cross-exchange prices for:\n";
    for (const auto &symbol : symbols) {
      std::cout << "  - " << symbol << "\n";
    }
    std::cout << "\n";

    auto last_print = std::chrono::steady_clock::now();
    const auto print_interval = std::chrono::seconds(5);

    while (running) {
      // Process trades
      while (auto trade = reader.poll_trade()) {
        // Processed automatically
      }

      // Check for opportunities
      for (const auto &symbol : symbols) {
        // Stop hunts
        if (auto signal = stop_hunt.detect(symbol)) {
          std::cout << "🚨 " << signal->to_string() << "\n";
        }

        // Arbitrage
        auto arb_opps = arbitrage.detect_all(symbol);
        for (const auto &opp : arb_opps) {
          std::cout << "💰 " << opp.to_string() << "\n";
        }
      }

      // Print price comparison periodically
      auto now = std::chrono::steady_clock::now();
      if (now - last_print >= print_interval) {
        for (const auto &symbol : symbols) {
          auto prices = reader.get_all_exchange_prices(symbol);
          print_price_comparison(prices, symbol);
        }

        std::cout << "\nDetections: Stop Hunts=" << stop_hunt.get_detections()
                  << ", Arbitrage=" << arbitrage.get_detections() << "\n";

        last_print = now;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n\nFinal Statistics:\n";
    std::cout << "  Stop Hunts Detected: " << stop_hunt.get_detections()
              << "\n";
    std::cout << "  Arbitrage Opportunities: " << arbitrage.get_detections()
              << "\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}