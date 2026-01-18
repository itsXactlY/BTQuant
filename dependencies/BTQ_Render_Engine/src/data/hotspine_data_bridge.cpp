#include "hotspine_data_bridge.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

namespace BTQuant {

HotSpineDataBridge::HotSpineDataBridge(const std::string &shm_path)
    : m_running(false) {
  (void)shm_path; // Placeholder for actual SHM attachment logic
}

HotSpineDataBridge::~HotSpineDataBridge() { stop(); }

bool HotSpineDataBridge::start() {
  m_running = true;
  return true;
}

void HotSpineDataBridge::stop() { m_running = false; }

void HotSpineDataBridge::poll() {
  if (!m_running)
    return;

  std::lock_guard<std::mutex> lock(m_mutex);

  // --- Performance Optimization: High-Frequency Data Aggregation ---
  // In a real scenario, we would poll the SHM ringbuffer here.
  // Fallback: Generate high-fidelity synthetic data for "BINANCE:BTC-USDT"

  auto it = m_instruments.find("BINANCE:BTC-USDT");
  if (it == m_instruments.end()) {
    auto inst = std::make_shared<MarketInstrument>();
    inst->symbol = "BTC-USDT";
    inst->exchange = "BINANCE";

    // Seed initial history (1000 points)
    double now = (double)std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    double base_price = 45000.0;

    for (int i = 0; i < 1000; ++i) {
      double t = now - (1000 - i);
      double noise = (std::sin(i * 0.05) * 50.0) + (std::cos(i * 0.02) * 20.0);
      double o = base_price + noise;
      double c = o + (std::sin(i * 1.5) * 10.0);
      double h = std::max(o, c) + 5.0;
      double l = std::min(o, c) - 5.0;
      inst->push(t, o, h, l, c, 10.5);
    }

    // Mock Orderbook
    for (int i = 0; i < 15; ++i) {
      inst->ask_prices.push_back(base_price + 5.0 + i * 0.5);
      inst->ask_sizes.push_back(1.5 - i * 0.1);
      inst->bid_prices.push_back(base_price - 5.0 - i * 0.5);
      inst->bid_sizes.push_back(1.5 - i * 0.1);
    }

    m_instruments["BINANCE:BTC-USDT"] = inst;
  } else {
    // Update latest candle (Simulation)
    auto inst = it->second;
    double t_last = inst->timestamps.back();
    double now = (double)std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());

    if (now > t_last + 1.0) {
      double last_close = inst->closes.back();
      double noise = (std::sin(now * 0.1) * 2.0);
      inst->push(now, last_close, last_close + 5.0, last_close - 5.0,
                 last_close + noise, 1.2);

      // Randomly update orderbook
      std::lock_guard<std::mutex> inst_lock(inst->mutex);
      for (size_t i = 0; i < inst->ask_prices.size(); ++i) {
        inst->ask_prices[i] = inst->closes.back() + 2.0 + i * 0.5;
        inst->bid_prices[i] = inst->closes.back() - 2.0 - i * 0.5;
      }
    }
  }
}

} // namespace BTQuant