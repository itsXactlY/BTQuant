#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace BTQuant {

// HFT COMPLIANT: Structure of Arrays (SoA) for ImPlot zero-copy mapping.
struct MarketInstrument {
  std::string symbol;
  std::string exchange;

  // Contiguous SoA buffers for ImPlot (Pre-allocated)
  std::vector<double> timestamps, opens, highs, lows, closes, volumes;

  // Depth buffers
  std::vector<double> bid_prices, bid_sizes;
  std::vector<double> ask_prices, ask_sizes;

  int count = 0;
  mutable std::mutex mutex;

  void push(double t, double o, double h, double l, double c, double v) {
    std::lock_guard<std::mutex> lock(mutex);
    timestamps.push_back(t);
    opens.push_back(o);
    highs.push_back(h);
    lows.push_back(l);
    closes.push_back(c);
    volumes.push_back(v);
    count++;

    // Ringbuffer: Keep working set optimal for cache
    if (timestamps.size() > 1000) {
      timestamps.erase(timestamps.begin());
      opens.erase(opens.begin());
      highs.erase(highs.begin());
      lows.erase(lows.begin());
      closes.erase(closes.begin());
      volumes.erase(volumes.begin());
      count--;
    }
  }
};

class HotSpineDataBridge {
public:
  HotSpineDataBridge(const std::string &shm_path = "/btquant");
  ~HotSpineDataBridge();

  bool start();
  void stop();
  void poll();

  std::map<std::string, std::shared_ptr<MarketInstrument>> GetInstruments() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_instruments;
  }

private:
  std::map<std::string, std::shared_ptr<MarketInstrument>> m_instruments;
  std::mutex m_mutex;
  std::atomic<bool> m_running;
};

} // namespace BTQuant