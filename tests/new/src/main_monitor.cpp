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
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace BTQuant;

std::atomic<bool> running{true};

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    running = false;
  }
}

// Get current timestamp as string (HH:MM:SS.mmm)
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

// --- Dashboard Component ---

struct AlertItem {
  std::string timestamp;
  std::string type;
  std::string message;
};

class Dashboard {
public:
  Dashboard(HotSpineExtendedReader &reader,
            const std::vector<std::string> &symbols)
      : reader_(reader), symbols_(symbols) {
    start_time_ = std::chrono::steady_clock::now();
    // Clear screen once at startup
    std::cout << "\033[2J" << std::flush;
  }

  void add_alert(const std::string &type, const std::string &message) {
    // Spam prevention: Check if we have seen this identical message recently
    // Extract core message without timestamp for deduplication
    std::string core_message = message;
    // Remove timestamp pattern if present (e.g., "[12:34:56.789]")
    size_t timestamp_start = core_message.find('[');
    if (timestamp_start != std::string::npos) {
        size_t timestamp_end = core_message.find(']', timestamp_start);
        if (timestamp_end != std::string::npos) {
            core_message = core_message.substr(timestamp_end + 1);
            // Remove leading/trailing whitespace
            core_message.erase(0, core_message.find_first_not_of(" \t"));
            core_message.erase(core_message.find_last_not_of(" \t") + 1);
        }
    }
    std::string key = type + ":" + core_message;
    auto now = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(mutex_);

    // Cleanup old dedupe entries
    for (auto it = dedupe_cache_.begin(); it != dedupe_cache_.end();) {
      if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second)
              .count() > 30) {  // Increased from 5 to 30 seconds
        it = dedupe_cache_.erase(it);
      } else {
        ++it;
      }
    }

    if (dedupe_cache_.find(key) != dedupe_cache_.end()) {
      return; // Suppress duplicate alert
    }

    dedupe_cache_[key] = now;
    alerts_.push_front({get_timestamp(), type, message});
    if (alerts_.size() > 15) {
      alerts_.pop_back();
    }
  }

  void render() {
    std::stringstream ss;

    // ANSI escape codes: Move cursor to home (don't clear entire screen to
    // reduce flicker)
    ss << "\033[H";

    render_header(ss);
    render_stats(ss);
    render_market_table(ss);
    render_alerts(ss);

    // Clear from cursor to end of screen to clean up any debris
    ss << "\033[J";

    std::cout << ss.str() << std::flush;
  }

private:
  HotSpineExtendedReader &reader_;
  std::vector<std::string> symbols_;
  std::deque<AlertItem> alerts_;
  std::map<std::string, std::chrono::steady_clock::time_point> dedupe_cache_;
  std::mutex mutex_;
  std::chrono::steady_clock::time_point start_time_;

  // Formatting helpers
  std::string color_green(const std::string &s) {
    return "\033[32m" + s + "\033[0m";
  }
  std::string color_red(const std::string &s) {
    return "\033[31m" + s + "\033[0m";
  }
  std::string color_yellow(const std::string &s) {
    return "\033[33m" + s + "\033[0m";
  }
  std::string color_cyan(const std::string &s) {
    return "\033[36m" + s + "\033[0m";
  }
  std::string bold(const std::string &s) { return "\033[1m" + s + "\033[0m"; }

  void render_header(std::stringstream &ss) {
    auto now = std::chrono::steady_clock::now();
    auto uptime =
        std::chrono::duration_cast<std::chrono::seconds>(now - start_time_)
            .count();
    int h = uptime / 3600;
    int m = (uptime % 3600) / 60;
    int s = uptime % 60;

    ss << bold("╔════════════════════════════════════════════════════════════╗")
       << "\033[K\n";
    ss << bold("║      BTQuant Market Manipulation Detector v2.0             ║")
       << "\033[K\n";
    ss << bold("║      ") << std::left << std::setw(54)
       << ("Status: " + color_green("RUNNING") +
           " | Uptime: " + std::to_string(h) + "h " + std::to_string(m) + "m " +
           std::to_string(s) + "s")
       << bold("║") << "\033[K\n";
    ss << bold("╚════════════════════════════════════════════════════════════╝")
       << "\033[K\n\n";
  }

  void render_stats(std::stringstream &ss) {
    auto [used, capacity] = reader_.get_buffer_utilization();
    double usage = capacity > 0 ? (used * 100.0 / capacity) : 0.0;

    ss << bold("System Statistics:") << "\033[K\n";
    ss << "Trades Processed: " << std::setw(10) << reader_.get_trades_read()
       << " | Buffer Usage: " << std::fixed << std::setprecision(2) << usage
       << "%"
       << " | Lost: " << reader_.get_lost_count() << "\033[K\n\n";
  }

  void render_market_table(std::stringstream &ss) {
    ss << bold("Market Data (Real-time):") << "\033[K\n";
    ss << "┌──────────────────┬──────────────┬──────────────┬──────────┬───────"
          "──────┐\033[K\n";
    ss << "│ " << std::left << std::setw(16) << "Symbol"
       << " │ " << std::setw(12) << "Price"
       << " │ " << std::setw(12) << "Size"
       << " │ " << std::setw(8) << "Side"
       << " │ " << std::setw(11) << "Latency(ms)"
       << " │\033[K\n";
    ss << "├──────────────────┼──────────────┼──────────────┼──────────┼───────"
          "──────┤\033[K\n";

    auto exchanges = SymbolRegistry::instance().get_exchanges();

    for (const auto &exchange : exchanges) {
      auto syms = SymbolRegistry::instance().get_exchange_symbols(exchange);
      for (const auto &s : syms) {
        // Filter logic using normalized comparison
        bool is_monitored = false;
        if (symbols_.size() == 1 && symbols_[0] == "ALL") {
          is_monitored = true;
        } else {
          for (const auto &mon_sym : symbols_) {
            std::string s1 = s.symbol;
            std::string s2 = mon_sym;
            // Normalize
            s1.erase(std::remove(s1.begin(), s1.end(), '-'), s1.end());
            s2.erase(std::remove(s2.begin(), s2.end(), '-'), s2.end());

            if (s2.find(':') != std::string::npos) {
              s2 = s2.substr(s2.find(':') + 1);
            }

            if (s1 == s2) {
              is_monitored = true;
              break;
            }
          }
        }

        if (!is_monitored)
          continue;

        auto trades = reader_.get_recent_trades(exchange, s.symbol, 1);

        std::string price_str = "--";
        std::string size_str = "--";
        std::string side_str = "--";
        std::string latency_str = "--";

        if (!trades.empty()) {
          const auto &t = trades.back();
          price_str = std::to_string(t.price);
          size_str = std::to_string(t.size);
          side_str = t.side == 0 ? color_green("BUY") : color_red("SELL");

          // Latency calc
          uint64_t now = reader_.get_current_time_us();
          double lat_ms =
              (now > t.ts_exchange) ? (now - t.ts_exchange) / 1000.0 : 0.0;
          latency_str = std::to_string(lat_ms);
          // Truncate
          latency_str = latency_str.substr(0, latency_str.find('.') + 2);
        }

        std::string full_sym = exchange + ":" + s.symbol;
        if (full_sym.length() > 16)
          full_sym = full_sym.substr(0, 16);

        ss << "│ " << std::left << std::setw(16) << full_sym << " │ "
           << std::setw(12)
           << (price_str.length() > 12 ? price_str.substr(0, 12) : price_str)
           << " │ " << std::setw(12)
           << (size_str.length() > 12 ? size_str.substr(0, 12) : size_str)
           << " │ " << std::setw(17) << side_str // side_str has ansi codes
           << " │ " << std::setw(11) << latency_str << " │\033[K\n";
      }
    }
    ss << "└──────────────────┴──────────────┴──────────────┴──────────┴───────"
          "──────┘\033[K\n\n";
  }

  void render_alerts(std::stringstream &ss) {
    ss << bold("Recent Alerts (Last 15):") << "\033[K\n";
    std::lock_guard<std::mutex> lock(mutex_);
    if (alerts_.empty()) {
      ss << "  (No alerts detected)\033[K\n";
    } else {
      for (const auto &alert : alerts_) {
        ss << "  [" << alert.timestamp << "] " << bold(alert.type) << ": "
           << alert.message << "\033[K\n";
      }
    }
  }
};

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

  // Initialize config
  auto &config = Config::ConfigLoader::instance();
  config.initialize();
  config.load();

  // Initialize logger (file only to avoid messing up dashboard?)
  // Ideally we redirect logger to file and keep stdout for dashboard
  auto &logger = Logging::DynamicLogger::instance();
  logger.initialize();
  // TODO: Configure logger to NOT print to stdout if possible, or we accept
  // some glitching. For now, assuming logger prints to stdout/stderr. We can't
  // easily suppress it without code change in logger. Hopefully silencing the
  // "Trade #..." log is enough.

  std::string shm_name = config.get_as<std::string>("shared_memory", "name")
                             .value_or("/btquant_hotspine");

  // Get monitored symbols from config
  std::vector<std::string> symbols;
  if (auto syms =
          config.get_as<std::vector<std::string>>("monitoring", "symbols")) {
    symbols = *syms;
  } else {
    // Default to ALL if not configured
    symbols = {"ALL"};
  }

  // Remove matching hyphen logic, we will handle it in the render loop
  // dynamically or just normalize here. Let's normalize here for the 'symbols'
  // vector but 'ALL' should be preserved. Actually, best to just keep raw
  // strings and normalize during comparison.

  // Get poll interval from config
  int poll_interval_ms =
      config.get_as<int>("monitoring", "poll_interval_ms").value_or(10);
  int stats_interval_sec =
      config.get_as<int>("monitoring", "stats_interval_sec").value_or(5);
  int market_info_interval_sec =
      config.get_as<int>("monitoring", "market_info_interval_sec").value_or(1);

  std::cout << "   Shared Memory: " << shm_name << "\n";
  std::cout << "   Poll Interval: " << poll_interval_ms << "ms\n";
  std::cout << "   Stats Interval: " << stats_interval_sec << "s\n";
  std::cout << "   Market Info Interval: " << market_info_interval_sec << "s\n";
  std::cout << "   Monitored Symbols: "
            << (symbols.size() == 1 && symbols[0] == "ALL"
                    ? "ALL (Auto-detect)"
                    : std::to_string(symbols.size()))
            << "\n\n";

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

    // Dashboard
    Dashboard dashboard(reader, symbols);

    std::cout << "🚀 Starting monitoring dashboard...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto last_render = std::chrono::steady_clock::now();

    // Main monitoring loop
    while (running) {
      // 1. Poll Trades
      while (auto trade = reader.poll_trade()) {
        // Process trade (updates internal caches)
      }

      // 2. Run Detectors
      // We need to iterate all KNOWN symbols in the registry to be thorough
      auto exchanges = SymbolRegistry::instance().get_exchanges();
      for (const auto &exchange : exchanges) {
        auto syms = SymbolRegistry::instance().get_exchange_symbols(exchange);
        for (const auto &s : syms) {
          std::string symbol = s.symbol;

          // Filter logic for detectors
          bool is_monitored = false;
          if (symbols.size() == 1 && symbols[0] == "ALL") {
            is_monitored = true;
          } else {
            for (const auto &mon_sym : symbols) {
              // Fuzzy match: ignore hyphens
              std::string s1 = symbol;
              std::string s2 = mon_sym;
              s1.erase(std::remove(s1.begin(), s1.end(), '-'), s1.end());
              s2.erase(std::remove(s2.begin(), s2.end(), '-'), s2.end());

              // Remove exchange prefix from mon_sym if present for comparison
              if (s2.find(':') != std::string::npos) {
                s2 = s2.substr(s2.find(':') + 1);
              }

              if (s1 == s2) {
                is_monitored = true;
                break;
              }
            }
          }

          if (!is_monitored)
            continue;

          // 1. Stop Hunt Detection
          if (auto signal = stop_hunt.detect(symbol)) {
            dashboard.add_alert("STOP HUNT", signal->to_string());
          }

          // 2. Liquidity Imbalance Detection
          if (auto signal = liquidity.detect(symbol)) {
            dashboard.add_alert("LIQUIDITY", signal->to_string());
          }

          // 3. Whale Front-Run Detection
          if (auto signal = whale.detect(symbol)) {
            dashboard.add_alert("WHALE", signal->to_string());
          }

          // 4. Spread Arbitrage Detection
          auto arb_signals = arbitrage.detect_all(symbol);
          for (const auto &signal : arb_signals) {
            dashboard.add_alert("ARBITRAGE", signal.to_string());
          }

          // 5. Spoofing Detection (per exchange)
          spoofing.update_orderbook(exchange, symbol);
          if (auto signal = spoofing.detect(exchange, symbol)) {
            dashboard.add_alert("SPOOFING", signal->to_string());
          }
        }
      }

      // 3. Render Dashboard (10fps)
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                last_render)
              .count() >= 100) {
        dashboard.render();
        last_render = now;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "\n✅ Monitor stopped.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "❌ Fatal error: " << e.what() << "\n";
    return 1;
  }
}
