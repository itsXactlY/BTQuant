#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstdio>

namespace MarketData {

inline std::string formatTimestampMs(int64_t timestamp_ms) {
    using namespace std::chrono;
    system_clock::time_point tp{milliseconds(timestamp_ms)};
    std::time_t tt = system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    char out[40];
    std::snprintf(out, sizeof(out), "%s.%03lld", buf,
                  static_cast<long long>(timestamp_ms % 1000));
    return std::string(out);
}

struct Trade {
    int64_t timestamp_ms{};
    std::string exchange;
    std::string symbol;
    std::string market_type;  // "spot", "perpetual", ...
    std::string trade_id;
    double price{};
    double quantity{};
    std::string side;         // "buy" / "sell"
    bool is_buyer_maker{};

    std::string toTimestamp() const {
        return formatTimestampMs(timestamp_ms);
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
    int64_t timestamp_ms{};   // candle open time
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
        return formatTimestampMs(timestamp_ms);
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
    int64_t timestamp_ms{};
    std::string exchange;
    std::string symbol;
    std::string market_type;
    std::string bids_json;
    std::string asks_json;
    std::string checksum;

    std::vector<std::string> toSQLValues() const {
        return {
            formatTimestampMs(timestamp_ms),
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
