#include "mock_producer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <sys/stat.h>
#include <unistd.h>

namespace btquant::data {

namespace {

constexpr uint32_t kHeaderMagic   = 0x51555442;  // 'UQTB' little-endian
constexpr uint16_t kHeaderVersion = 1;
constexpr uint32_t kHeaderSize    = 0x1000;
constexpr uint32_t kSymbolRecordSize = 128;
constexpr uint32_t kHeaderSymbolCount = 1;  // single-symbol producer

// GBM parameters — match scripts/mock_producer.py.
constexpr double kSigma      = 0.0002;  // per-second volatility
constexpr double kBaseSpread = 0.5;     // half-spread at rest

} // namespace

MockProducer::MockProducer(std::string hotspinePath,
                           std::string symbol,
                           std::string exchange,
                           double priceStart,
                           uint32_t intervalMs)
    : m_path(std::move(hotspinePath)),
      m_symbol(std::move(symbol)),
      m_exchange(std::move(exchange)),
      m_priceStart(priceStart),
      m_intervalMs(intervalMs == 0 ? 100 : intervalMs) {}

MockProducer::~MockProducer() { stop(); }

std::optional<std::string> MockProducer::start() {
    if (m_running.load(std::memory_order_acquire)) {
        return std::string("MockProducer already running");
    }

    // Truncate + open the SHM file. We replace any existing data so a stale
    // binary sequence isn't read as fresh ticks.
    ::unlink(m_path.c_str());
    m_fd = ::open(m_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (m_fd < 0) {
        return std::string("MockProducer: open(") + m_path + ") failed: "
               + std::strerror(errno);
    }
    // Pre-allocate header + 1 symbol record.
    if (::ftruncate(m_fd, kHeaderSize + kSymbolRecordSize) != 0) {
        ::close(m_fd);
        m_fd = -1;
        return std::string("MockProducer: ftruncate failed: ") + std::strerror(errno);
    }

    if (!writeHeader()) {
        ::close(m_fd);
        m_fd = -1;
        return std::string("MockProducer: writeHeader failed");
    }

    std::fprintf(stderr, "[MockProducer] started: %s symbol=%s price_start=%.2f interval=%ums\n",
                 m_path.c_str(), m_symbol.c_str(), m_priceStart, m_intervalMs);

    m_shouldStop.store(false, std::memory_order_release);
    m_running.store(true, std::memory_order_release);
    m_thread = std::thread([this]() { runLoop(); });
    return std::nullopt;
}

void MockProducer::stop() {
    if (!m_running.load(std::memory_order_acquire)) return;
    m_shouldStop.store(true, std::memory_order_release);
    if (m_thread.joinable()) m_thread.join();
    m_running.store(false, std::memory_order_release);
    if (m_fd >= 0) {
        ::close(m_fd);
        m_fd = -1;
    }
    std::fprintf(stderr, "[MockProducer] stopped at seq=%lu\n",
                 static_cast<unsigned long>(m_seq.load()));
}

bool MockProducer::writeHeader() {
    char header[kHeaderSize];
    std::memset(header, 0, sizeof(header));

    // First 4 bytes = 'UQTB' magic (literal, not u32 LE).
    std::memcpy(header, "UQTB", 4);
    // Next 4 bytes = version u16 LE + symbol_count u16 LE.
    uint16_t version = kHeaderVersion;
    uint16_t count   = kHeaderSymbolCount;
    std::memcpy(header + 4, &version, sizeof(uint16_t));
    std::memcpy(header + 6, &count,   sizeof(uint16_t));

    ssize_t n = ::pwrite(m_fd, header, kHeaderSize, 0);
    if (n != static_cast<ssize_t>(kHeaderSize)) {
        std::fprintf(stderr, "[MockProducer] pwrite(header) failed: %s\n",
                     std::strerror(errno));
        return false;
    }
    return true;
}

void MockProducer::runLoop() {
    using clock = std::chrono::steady_clock;

    std::mt19937_64 rng(static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::normal_distribution<double> noise(0.0, 1.0);
    std::uniform_real_distribution<double> sizeJitter(0.05, 5.0);

    double price = m_priceStart;
    const double dtSec = m_intervalMs / 1000.0;

    auto lastLog = clock::now();
    auto nextTick = clock::now() + std::chrono::milliseconds(m_intervalMs);

    while (!m_shouldStop.load(std::memory_order_acquire)) {
        // Sleep until next tick (instead of sleeping AFTER — keeps timing tight).
        std::this_thread::sleep_until(nextTick);

        // GBM step: dS = sigma * S * dW
        double dW = noise(rng) * std::sqrt(dtSec);
        price = std::max(price * std::exp(kSigma * dW), 1.0);
        m_lastPrice.store(price, std::memory_order_release);

        // Spread widens with volatility.
        double spread = kBaseSpread * (1.0 + std::abs(dW) * 50.0);
        double bid = price - spread * 0.5;
        double ask = price + spread * 0.5;

        double bidSize = sizeJitter(rng);
        double askSize = sizeJitter(rng);

        uint64_t seq = m_seq.fetch_add(1, std::memory_order_acq_rel);
        uint64_t tsUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        uint32_t flags = 0;

        // Pack into the 128-byte symbol record.
        char buf[kSymbolRecordSize];
        std::memset(buf, 0, sizeof(buf));
        std::memcpy(buf + 0,  &bid,     sizeof(double));
        std::memcpy(buf + 8,  &ask,     sizeof(double));
        std::memcpy(buf + 16, &bidSize, sizeof(double));
        std::memcpy(buf + 24, &askSize, sizeof(double));
        std::memcpy(buf + 32, &tsUs,    sizeof(uint64_t));
        std::memcpy(buf + 40, &seq,     sizeof(uint32_t));
        std::memcpy(buf + 44, &flags,   sizeof(uint32_t));
        // Bytes 48..127 stay zeroed (padding to 128).

        ssize_t n = ::pwrite(m_fd, buf, kSymbolRecordSize,
                             kHeaderSize + 0 * kSymbolRecordSize);
        if (n != static_cast<ssize_t>(kSymbolRecordSize)) {
            std::fprintf(stderr, "[MockProducer] pwrite(record) failed: %s\n",
                         std::strerror(errno));
            // Continue — the next tick will retry. Don't kill the producer on a
            // single transient write failure.
        }

        // Periodic heartbeat to stderr (every ~5 s).
        auto now = clock::now();
        if (now - lastLog > std::chrono::seconds(5)) {
            std::fprintf(stderr, "[MockProducer] seq=%lu price=%.2f bid=%.2f ask=%.2f\n",
                         static_cast<unsigned long>(seq), price, bid, ask);
            lastLog = now;
        }

        nextTick += std::chrono::milliseconds(m_intervalMs);
    }
}

} // namespace btquant::data
