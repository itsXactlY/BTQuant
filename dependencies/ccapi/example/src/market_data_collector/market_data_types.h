#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstdio>

namespace MarketData {

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

struct Trade {
    int64_t timestamp_us{};   // epoch microseconds (UTC)
    std::string exchange;
    std::string symbol;
    std::string market_type;  // "spot", "perpetual", ...
    std::string trade_id;
    double price{};
    double quantity{};
    std::string side;         // "buy" / "sell"
    bool is_buyer_maker{};

    std::string toTimestamp() const {
        return formatTimestampMicros(timestamp_us);
    }

    std::vector<std::string> toSQLValues() const {
        return {
            toTimestamp(),
            exchange,
            symbol,
            market_type,
            trade_id,
            std::to_string(price),
            std::to_string(quantity),
            side,
            is_buyer_maker ? "1" : "0"
        };
    }
};

struct OHLCV {
    int64_t timestamp_us{};   // candle open time (epoch microseconds)
    std::string exchange;
    std::string symbol;
    std::string market_type;
    std::string timeframe;    // "1m", "5m", ...

    double open{};
    double high{};
    double low{};
    double close{};
    double volume{};

    std::string toTimestamp() const {
        return formatTimestampMicros(timestamp_us);
    }

    std::vector<std::string> toSQLValues() const {
        return {
            toTimestamp(),
            exchange,
            symbol,
            market_type,
            timeframe,
            std::to_string(open),
            std::to_string(high),
            std::to_string(low),
            std::to_string(close),
            std::to_string(volume)
        };
    }

    std::string getTableName() const {
        return symbol + "_klines";
    }
};

struct OrderbookSnapshot {
    int64_t timestamp_us{};   // epoch microseconds
    std::string exchange;
    std::string symbol;
    std::string market_type;
    std::string bids_json;
    std::string asks_json;
    std::string checksum;

    std::vector<std::string> toSQLValues() const {
        return {
            formatTimestampMicros(timestamp_us),
            exchange,
            symbol,
            market_type,
            bids_json,
            asks_json,
            checksum
        };
    }
};

} // namespace MarketData
