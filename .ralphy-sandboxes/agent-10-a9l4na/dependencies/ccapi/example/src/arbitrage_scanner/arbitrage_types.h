#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <atomic>
#include <array>
#include <string_view>

namespace Arbitrage {

inline std::string formatTimestampMicros(int64_t timestamp_us) {
    using namespace std::chrono;
    system_clock::time_point tp(
        duration_cast<system_clock::duration>(microseconds(timestamp_us)));

    std::time_t tt = system_clock::to_time_t(tp);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &tt);
#else
    gmtime_r(&tt, &tm_utc);
#endif

    char buf[64];
    int fractional = static_cast<int>(timestamp_us % 1'000'000);
    std::snprintf(
        buf, sizeof(buf),
        "%04d-%02d-%02d %02d:%02d:%02d.%06d",
        tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
        tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec,
        fractional < 0 ? fractional + 1'000'000 : fractional);

    return std::string(buf);
}

// exchange fee configuration (basis points)
struct ExchangeFees {
    const char* exchange_name;
    double maker_fee_bps;
    double taker_fee_bps;
    
    constexpr double effective_fee_bps() const noexcept {
        return taker_fee_bps; // conservative
    }
};

// lck-free order book level
struct alignas(64) OrderBookLevel {
    std::atomic<double> price{0.0};
    std::atomic<double> volume{0.0};
    std::atomic<int64_t> timestamp_us{0};
    
    void update(double p, double v, int64_t ts) noexcept {
        price.store(p, std::memory_order_relaxed);
        volume.store(v, std::memory_order_relaxed);
        timestamp_us.store(ts, std::memory_order_release);
    }
    
    struct Snapshot {
        double price;
        double volume;
        int64_t timestamp_us;
    };
    
    Snapshot load() const noexcept {
        return Snapshot{
            price.load(std::memory_order_relaxed),
            volume.load(std::memory_order_relaxed),
            timestamp_us.load(std::memory_order_acquire)
        };
    }
};

// arbitrage opportunity
struct Opportunity {
    int64_t timestamp_us{};
    std::string buy_exchange;
    std::string sell_exchange;
    std::string symbol;
    std::string market_type;
    
    double buy_price{};
    double sell_price{};
    double profit_bps{};
    double max_volume{};
    int64_t latency_us{};
    
    std::string toTimestamp() const {
        return formatTimestampMicros(timestamp_us);
    }
    
    std::vector<std::string> toSQLValues() const {
        return {
            toTimestamp(),
            buy_exchange,
            sell_exchange,
            symbol,
            market_type,
            std::to_string(buy_price),
            std::to_string(sell_price),
            std::to_string(profit_bps),
            std::to_string(max_volume),
            std::to_string(latency_us)
        };
    }
    
    bool is_valid(int64_t now_us, int64_t max_age_us = 100'000) const noexcept {
        return (now_us - timestamp_us) < max_age_us && profit_bps > 0.0;
    }
};

// pre-configured exchange fees (NOW constexpr-compatible, ha!)
inline constexpr std::array<ExchangeFees, 5> EXCHANGE_FEES{{
    {"binance",   10.0, 10.0},
    {"coinbase",  50.0, 50.0},
    {"kraken",    26.0, 26.0},
    {"okx",       10.0, 10.0},
    {"bybit",     10.0, 10.0}
}};

inline double get_exchange_fee_bps(std::string_view exchange) {
    for (const auto& fee : EXCHANGE_FEES) {
        if (exchange == fee.exchange_name) {
            return fee.effective_fee_bps();
        }
    }
    return 50.0; // conservative default
}

} // namespace Arbitrage
