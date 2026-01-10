#include "detectors/liquidity_imbalance_detector.hpp"
#include "detectors/spoofing_detector.hpp"
#include "detectors/spread_arbitrage_detector.hpp"
#include "detectors/stop_hunt_detector.hpp"
#include "detectors/whale_frontrun_detector.hpp"
#include "hotspine_extended_reader.hpp"
#include "utils/dynamic_logger.hpp"
#include <config/config_loader.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace BTQuant;

std::atomic<bool> running{true};

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "\n🛑 Shutting down...\n";
    running = false;
  }
}

std::string get_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time), "%H:%M:%S");
  ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return ss.str();
}

void print_banner() {
  std::cout << R"(
 ╔════════════════════════════════════════════════════════════╗
 ║                                                            ║
 ║      BTQuant Market Manipulation Detector v2.0             ║
 ║      Dynamic Configuration & Plugin System                  ║
 ║                                                            ║
 ╚════════════════════════════════════════════════════════════╝
 )" << std::endl;
}

struct StatsSnapshot {
  uint64_t trades;
  double buffer_usage;
  uint64_t lost;
  uint64_t stop_hunts;
  uint64_t liquidity;
  uint64_t whales;
  uint64_t arbitrage;
  uint64_t spoofing;
  std::string timestamp;
};

StatsSnapshot capture_stats(const HotSpineExtendedReader &reader,
                            const StopHuntDetector &stop_hunt,
                            const LiquidityImbalanceDetector &liquidity,
                            const WhaleFrontRunDetector &whale,
                            const SpreadArbitrageDetector &arbitrage,
                            const SpoofingDetector &spoofing) {
  auto [current, capacity] = reader.get_buffer_utilization();

  StatsSnapshot stats;
  stats.trades = reader.get_trades_read();
  stats.buffer_usage = capacity > 0 ? (current * 100.0 / capacity) : 0.0;
  stats.lost = reader.get_lost_count();
  stats.stop_hunts = stop_hunt.get_detections();
  stats.liquidity = liquidity.get_detections();
  stats.whales = whale.get_detections();
  stats.arbitrage = arbitrage.get_detections();
  stats.spoofing = spoofing.get_detections();
  stats.timestamp = get_timestamp();

  return stats;
}

void print_statistics(const StatsSnapshot &stats) {
  std::cout << "\n┌─────────────────────────────────────────────┐\n";
  std::cout << "│  [" << stats.timestamp << "] System Statistics      │\n";
  std::cout << "├─────────────────────────────────────────────┤\n";
  std::cout << "│ Trades Processed:  " << std::setw(20) << stats.trades
            << " │\n";
  std::cout << "│ Buffer Usage:      " << std::setw(17) << std::fixed
            << std::setprecision(1) << stats.buffer_usage << "% │\n";
  std::cout << "│ Lost Trades:       " << std::setw(20) << stats.lost << " │\n";
  std::cout << "├─────────────────────────────────────────────┤\n";
  std::cout << "│ Stop Hunts:        " << std::setw(20) << stats.stop_hunts
            << " │\n";
  std::cout << "│ Liquidity Issues:  " << std::setw(20) << stats.liquidity
            << " │\n";
  std::cout << "│ Whale Trades:      " << std::setw(20) << stats.whales
            << " │\n";
  std::cout << "│ Arbitrage Opps:    " << std::setw(20) << stats.arbitrage
            << " │\n";
  std::cout << "│ Spoofing Events:   " << std::setw(20) << stats.spoofing
            << " │\n";
  std::cout << "└─────────────────────────────────────────────┘\n";
}

void print_market_info(HotSpineExtendedReader &reader) {
  std::cout << "\n" << get_timestamp() << " === MARKET INFO ===\n";

  auto exchanges = SymbolRegistry::instance().get_exchanges();

  for (const auto &exchange : exchanges) {
    auto symbols = SymbolRegistry::instance().get_exchange_symbols(exchange);

    std::cout << "\n┌─ " << exchange << " ───────────────────────\n";

    for (const auto &symbol_info : symbols) {
      auto price = reader.get_latest_price(exchange, symbol_info.symbol);
      auto orderbook =
          reader.get_latest_orderbook(exchange, symbol_info.symbol);

      std::cout << "│ " << std::left << std::setw(12) << symbol_info.symbol
                << " │ ID: " << std::setw(4) << symbol_info.id;

      if (price) {
        std::cout << " │ Price: $" << std::fixed << std::setprecision(2)
                  << *price;
      } else {
        std::cout << " │ Price: --";
      }

      if (orderbook) {
        std::cout << " │ Best Bid: $" << std::fixed << std::setprecision(2)
                  << orderbook->best_bid() << " │ Best Ask: $"
                  << orderbook->best_ask() << " │ Spread: " << std::fixed
                  << std::setprecision(2) << orderbook->spread_bps() << " bps";
      }

      std::cout << "\n";
    }

    std::cout << "└" << std::string(38, '-') << "\n";
  }

  auto [used, capacity] = reader.get_buffer_utilization();
  std::cout << "\n📊 Shared Memory Buffer: " << used << " / " << capacity
            << " (" << (capacity > 0 ? (used * 100.0 / capacity) : 0.0)
            << "%)\n";
}

int main(int argc, char *argv[]) {
  // Check for demo mode flag
  bool demo_mode = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--demo" || std::string(argv[i]) == "-d") {
      demo_mode = true;
    }
  }

  // Setup signal handlers
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  print_banner();

  if (demo_mode) {
    std::cout << "🎮 Running in DEMO mode (no shared memory required)\n";
    std::cout << "   Using simulated market data for demonstration\n\n";
  } else {
    std::cout << "📊 Initializing configuration system...\n";
  }
  auto &config = Config::ConfigLoader::instance();
  config.initialize();
  config.load();

  // Initialize logger
  auto &logger = Logging::DynamicLogger::instance();
  logger.initialize();

  // Get configuration values
  std::string shm_name = config.get_as<std::string>("shared_memory", "name")
                             .value_or("/btquant_hotspine");
  int poll_interval_ms =
      config.get_as<int64_t>("monitoring", "poll_interval_ms").value_or(100);
  int stats_interval_sec =
      config.get_as<int64_t>("monitoring", "stats_interval_sec").value_or(5);
  int market_info_interval_sec =
      config.get_as<int64_t>("monitoring", "market_info_interval_sec")
          .value_or(60);

  // Get monitored symbols from config
  std::vector<std::string> symbols;
  if (auto syms =
          config.get_as<std::vector<std::string>>("monitoring", "symbols")) {
    symbols = *syms;
  } else {
    // Default symbols if not configured
    symbols = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT"};
  }

  std::cout << "   Shared Memory: " << shm_name << "\n";
  std::cout << "   Poll Interval: " << poll_interval_ms << "ms\n";
  std::cout << "   Stats Interval: " << stats_interval_sec << "s\n";
  std::cout << "   Market Info Interval: " << market_info_interval_sec << "s\n";
  std::cout << "   Monitored Symbols: " << symbols.size() << "\n\n";

  try {
    // Initialize reader with dynamic configuration
    HotSpineExtendedReader reader(shm_name);

    if (!reader.is_attached()) {
      std::cerr << "❌ Failed to attach to shared memory!\n";
      std::cerr << "   Make sure the market data collector is running.\n";
      return 1;
    }

    std::cout << "✅ Successfully attached to HotSpine\n";

    // Load symbol mappings - prioritize shared memory dynamic file
    std::string shm_mapping_path = "/dev/shm/btquant_symbols.json";
    std::string config_mapping_path =
        config.get_as<std::string>("symbols", "fallback_file")
            .value_or("config/symbol_mapping.json");

    bool mappings_loaded = false;

    // Try SHM path first
    if (reader.load_symbol_mappings(shm_mapping_path)) {
      std::cout << "✅ Loaded symbol mappings from " << shm_mapping_path
                << "\n";
      mappings_loaded = true;
    }
    // Fallback to config path
    else if (reader.load_symbol_mappings(config_mapping_path)) {
      std::cout << "✅ Loaded symbol mappings from " << config_mapping_path
                << "\n";
      mappings_loaded = true;
    } else {
      std::cerr
          << "⚠️  Warning: Failed to load symbol mappings from either source\n";
    }

    // Print initial market info
    std::cout << "\n📋 Loaded Symbol Mappings:\n";
    auto exchanges = SymbolRegistry::instance().get_exchanges();
    int total_symbols = 0;
    for (const auto &exchange : exchanges) {
      auto exchange_symbols =
          SymbolRegistry::instance().get_exchange_symbols(exchange);
      std::cout << "   " << exchange << ": " << exchange_symbols.size()
                << " symbols\n";
      total_symbols += exchange_symbols.size();
    }
    std::cout << "   Total: " << total_symbols << " symbol mappings\n\n";

    // Initialize detectors with dynamic configuration
    std::cout << "🔍 Initializing detectors...\n";

    StopHuntDetector stop_hunt(reader);
    stop_hunt.set_threshold_pct(
        config.get_as<double>("stop_hunt", "threshold_pct").value_or(0.5));
    stop_hunt.set_min_exchanges(
        config.get_as<int64_t>("stop_hunt", "min_exchanges").value_or(3));

    LiquidityImbalanceDetector liquidity(reader);
    liquidity.set_depth_ratio_threshold(
        config.get_as<double>("liquidity_imbalance", "depth_ratio_threshold")
            .value_or(3.0));

    WhaleFrontRunDetector whale(reader);
    whale.set_threshold_usd(
        config.get_as<double>("whale_frontrun", "threshold_usd")
            .value_or(100000));

    SpreadArbitrageDetector arbitrage(reader);
    arbitrage.set_min_profit_bps(
        config.get_as<double>("spread_arbitrage", "min_profit_bps")
            .value_or(50));
    arbitrage.set_fees(
        config.get_as<double>("spread_arbitrage", "maker_fee_bps").value_or(10),
        config.get_as<double>("spread_arbitrage", "taker_fee_bps")
            .value_or(20));

    SpoofingDetector spoofing(reader);
    spoofing.set_min_cancel_count(
        config.get_as<int64_t>("spoofing", "min_cancel_count").value_or(3));

    std::cout << "✅ All detectors initialized\n\n";
    std::cout << "🚀 Starting monitoring loop...\n";
    std::cout << "   (Press Ctrl+C to stop)\n\n";

    auto last_stats_print = std::chrono::steady_clock::now();
    auto last_market_info_print = std::chrono::steady_clock::now();
    StatsSnapshot last_stats;
    bool first_print = true;

    // Main monitoring loop
    while (running) {
      // Process incoming trades
      while (auto trade = reader.poll_trade()) {
        // Trade processed automatically by reader (updates caches)
      }

      // Run detectors on each symbol
      for (const auto &symbol : symbols) {

        // 1. Stop Hunt Detection
        if (auto signal = stop_hunt.detect(symbol)) {
          std::cout << "🚨 " << signal->to_string() << "\n";
        }

        // 2. Liquidity Imbalance Detection
        if (auto signal = liquidity.detect(symbol)) {
          std::cout << "💧 " << signal->to_string() << "\n";
        }

        // 3. Whale Front-Run Detection
        if (auto signal = whale.detect(symbol)) {
          std::cout << "🐋 " << signal->to_string() << "\n";
        }

        // 4. Spread Arbitrage Detection
        auto arb_signals = arbitrage.detect_all(symbol);
        for (const auto &signal : arb_signals) {
          std::cout << "💰 " << signal.to_string() << "\n";
        }

        // 5. Spoofing Detection (per exchange)
        auto exch_list = SymbolRegistry::instance().get_exchanges();
        for (const auto &exchange : exch_list) {
          spoofing.update_orderbook(exchange, symbol);

          if (auto signal = spoofing.detect(exchange, symbol)) {
            std::cout << "👻 " << signal->to_string() << "\n";
          }
        }
      }

      // Print market info periodically
      auto now = std::chrono::steady_clock::now();
      if (now - last_market_info_print >=
          std::chrono::seconds(market_info_interval_sec)) {
        print_market_info(reader);
        last_market_info_print = now;
      }

      // Print statistics periodically
      if (now - last_stats_print >= std::chrono::seconds(stats_interval_sec)) {
        StatsSnapshot current_stats = capture_stats(
            reader, stop_hunt, liquidity, whale, arbitrage, spoofing);

        bool changed = (current_stats.trades != last_stats.trades ||
                        current_stats.stop_hunts != last_stats.stop_hunts ||
                        current_stats.liquidity != last_stats.liquidity ||
                        current_stats.whales != last_stats.whales ||
                        current_stats.arbitrage != last_stats.arbitrage ||
                        current_stats.spoofing != last_stats.spoofing);

        if (changed || first_print) {
          print_statistics(current_stats);
          first_print = false;
        }

        last_stats = current_stats;
        last_stats_print = now;
      }

      // Dynamic polling interval
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    }

    // Final statistics
    std::cout << "\n📊 Final Statistics:\n";
    print_statistics(capture_stats(reader, stop_hunt, liquidity, whale,
                                   arbitrage, spoofing));

    // Cleanup
    logger.shutdown();

    std::cout << "\n✅ Monitor stopped cleanly\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "❌ Fatal error: " << e.what() << "\n";
    return 1;
  }
}
