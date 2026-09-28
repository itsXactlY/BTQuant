#ifndef BTQUANT_MOCK_PRODUCER_HPP
#define BTQUANT_MOCK_PRODUCER_HPP

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>

namespace btquant::data {

// In-process port of scripts/mock_producer.py. Writes the same binary format
// to /dev/shm/btquant_hotspine so MarketDataProcessor (which opens that path
// via DataSpine) sees live data even when no external Python producer is
// running. Lets the binary run standalone for demos / CI.
//
// Format (matches mock_producer.py + data_spine.hpp):
//   Header @ 0x0000: "UQTB" (4B) + version u16 LE + symbol_count u16 LE +
//                    padding to 0x1000
//   Symbol records @ 0x1000: 128-byte records each with
//     bid_price (double), ask_price (double), bid_size (double), ask_size (double),
//     timestamp (uint64), seq (uint32), flags (uint32), padding to 128B
class MockProducer {
public:
    // Default: writes BTC/USDT @ $67500 every 100 ms, never exits.
    explicit MockProducer(std::string hotspinePath = "/dev/shm/btquant_hotspine",
                         std::string symbol = "BTC/USDT",
                         std::string exchange = "binance",
                         double priceStart = 67500.0,
                         uint32_t intervalMs = 100);

    ~MockProducer();

    // Spawn the background writer thread. Returns nullptr on success.
    std::optional<std::string> start();

    // Stop the writer thread. Joins cleanly. Safe to call multiple times.
    void stop();

    bool running() const { return m_running.load(std::memory_order_acquire); }
    uint64_t sequence() const { return m_seq.load(std::memory_order_acquire); }
    double lastPrice() const { return m_lastPrice.load(std::memory_order_acquire); }

private:
    void runLoop();

    // Writes the fixed-size header. Truncates the file first.
    bool writeHeader();

    std::string m_path;
    std::string m_symbol;
    std::string m_exchange;
    double      m_priceStart;
    uint32_t    m_intervalMs;

    int m_fd = -1;  // shm file descriptor
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};
    std::thread m_thread;
    std::atomic<uint64_t> m_seq{0};
    std::atomic<double>   m_lastPrice{0.0};
};

} // namespace btquant::data

#endif
