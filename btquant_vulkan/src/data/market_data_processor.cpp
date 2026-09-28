#include "market_data_processor.hpp"

#include "data_spine.hpp"
#include "market_data.hpp"

#include <cstdio>
#include <chrono>

namespace btquant {

MarketDataProcessor::MarketDataProcessor() = default;

MarketDataProcessor::~MarketDataProcessor() { stop(); }

std::optional<std::string>
MarketDataProcessor::start(const std::string& hotspine_path, uint32_t poll_interval_ms) {
    if (m_running.load()) return std::string("MarketDataProcessor already running");

    m_sourcePath = hotspine_path;
    m_ticksSeen.store(0, std::memory_order_relaxed);
    m_parseErrors.store(0, std::memory_order_relaxed);
    m_spine = std::make_unique<data::DataSpine>();
    if (!m_spine->open(hotspine_path)) {
        // Spine open failed — fall back to a synthetic generator so the UI
        // still shows live data when no producer is running.
        std::fprintf(stderr,
            "[MarketDataProcessor] could not open %s — falling back to "
            "synthetic generator (start the mock producer to see real data)\n",
            hotspine_path.c_str());
    }
    m_pollIntervalMs = poll_interval_ms;
    m_shouldStop.store(false);
    m_running.store(true);
    // Resolve the active symbol against the (now-open) spine; fall back to
    // nullopt if the spine didn't open or doesn't carry the requested symbol
    // — the runLoop then uses the synthetic generator.
    if (m_spine && m_spine->isOpen()) {
        m_activeSymbolIndex.store(m_spine->findSymbolIndex(m_symbol),
                                  std::memory_order_release);
    } else {
        m_activeSymbolIndex.store(std::nullopt, std::memory_order_release);
    }
    m_thread = std::thread([this]() { runLoop(); });
    return std::nullopt;
}

void MarketDataProcessor::stop() {
    if (!m_running.load()) return;
    m_shouldStop.store(true);
    if (m_thread.joinable()) m_thread.join();
    m_running.store(false);
    if (m_spine) { m_spine->close(); m_spine.reset(); }
}

// Switch active symbol — atomically updates the target, resets the
// aggregator and counters so stale state doesn't leak across switches.
void MarketDataProcessor::setSymbol(const std::string& sym) {
    m_symbol = sym;
    std::optional<uint32_t> idx;
    if (m_spine && m_spine->isOpen()) {
        idx = m_spine->findSymbolIndex(sym);
    }
    m_activeSymbolIndex.store(idx, std::memory_order_release);
    // Reset aggregator under snapshot mutex so the next snapshot reflects
    // the cleared state immediately.
    {
        std::lock_guard<std::mutex> lock(m_snapshotMutex);
        m_aggregator.reset();
        m_latestSnapshot = Snapshot{};
        m_latestSnapshot.snapshot_seq = ++m_snapshotSeq;
    }
    m_ticksSeen.store(0, std::memory_order_relaxed);
    m_parseErrors.store(0, std::memory_order_relaxed);
}

void MarketDataProcessor::runLoop() {
    using clock = std::chrono::steady_clock;
    auto last = clock::now();

    // Synthetic state — used when the spine can't be opened.
    double synthPrice = 100.0;
    uint64_t synthSeq = 0;
    double synthBid = 100.0;
    double synthAsk = 100.0;

    while (!m_shouldStop.load(std::memory_order_acquire)) {
        bool gotTick = false;

        if (m_spine && m_spine->isOpen()) {
            // Read all symbols but only feed the active one to the
            // aggregator — switching symbols must not blend two price feeds.
            auto idxOpt = m_activeSymbolIndex.load(std::memory_order_acquire);
            auto entries = m_spine->readAllEntries();
            // Entries are index-correlated: entries[i] == spine symbol i.
            // If the active index is unset we fall back to the synthetic
            // generator branch below; otherwise pick only that slot.
            if (idxOpt.has_value() && *idxOpt < entries.size()) {
                const auto& e = entries[*idxOpt];
                data::MarketTick tick{};
                tick.price = (e.bid_price + e.ask_price) * 0.5;
                tick.size = e.bid_size + e.ask_size;
                tick.timestamp = e.timestamp;
                tick.isBuy = (e.ask_size > e.bid_size);
                m_aggregator.update(tick);
                m_ticksSeen.fetch_add(1, std::memory_order_relaxed);
                synthBid = e.bid_price;
                synthAsk = e.ask_price;
                synthPrice = tick.price;
                gotTick = true;
            }
        } else {
            // Synthetic tick — small random walk around synthPrice.
            const double drift = ((double)rand() / RAND_MAX - 0.5) * 0.05;
            synthPrice += drift;
            synthBid = synthPrice - 0.01 - (rand() % 100) / 5000.0;
            synthAsk = synthPrice + 0.01 + (rand() % 100) / 5000.0;
            data::MarketTick tick{};
            tick.price = synthPrice;
            tick.size = 10.0 + (rand() % 100) / 10.0;
            tick.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            tick.isBuy = (rand() % 2 == 0);
            m_aggregator.update(tick);
            ++synthSeq;
            m_ticksSeen.fetch_add(1, std::memory_order_relaxed);
            gotTick = true;
        }

        if (gotTick) {
            // Publish snapshot under lock.
            std::lock_guard<std::mutex> lock(m_snapshotMutex);
            m_latestSnapshot.snapshot_seq = ++m_snapshotSeq;
            m_latestSnapshot.order_book = m_aggregator.orderBook();
            m_latestSnapshot.metrics = m_aggregator.metrics();
            m_latestSnapshot.recent_trades = m_aggregator.trades();
            m_latestSnapshot.recent_candles = m_aggregator.candles();
            m_latestSnapshot.current_candle = m_aggregator.currentCandle();
        }

        // Sleep until next poll.
        const auto target = last + std::chrono::milliseconds(m_pollIntervalMs);
        std::this_thread::sleep_until(target);
        last = clock::now();
    }
}

MarketDataProcessor::Snapshot MarketDataProcessor::snapshot(
    size_t last_n_trades, size_t last_n_candles) const {
    std::lock_guard<std::mutex> lock(m_snapshotMutex);
    Snapshot s = m_latestSnapshot;
    if (s.recent_trades.size() > last_n_trades) {
        s.recent_trades.resize(last_n_trades);
    }
    if (s.recent_candles.size() > last_n_candles) {
        // Keep the most recent N candles — drop from the front.
        s.recent_candles.erase(
            s.recent_candles.begin(),
            s.recent_candles.begin()
                + static_cast<std::ptrdiff_t>(s.recent_candles.size() - last_n_candles));
    }
    return s;
}

}  // namespace btquant
