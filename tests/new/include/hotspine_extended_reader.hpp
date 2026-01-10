#pragma once

#include "hotspine_layout.hpp"
#include "hotspine_reader.hpp"
#include "symbol_registry.hpp"
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace BTQuant {

using namespace HotSpine;

// Extended trade data with resolved symbol info
struct TradeData {
  uint64_t ts_exchange;
  uint64_t ts_local;
  double price;
  double size;
  uint32_t symbol_id;
  uint8_t side; // 0=buy, 1=sell

  std::string exchange;
  std::string symbol;

  bool is_buy() const { return side == 0; }
  bool is_sell() const { return side == 1; }

  std::string side_str() const { return is_buy() ? "BUY" : "SELL"; }
};

// Extended orderbook data with resolved symbol info
struct OrderbookData {
  uint64_t ts_exchange;
  uint64_t ts_local;
  uint32_t symbol_id;

  std::string exchange;
  std::string symbol;

  std::vector<std::pair<double, double>> bids; // price, size
  std::vector<std::pair<double, double>> asks;

  double best_bid() const { return bids.empty() ? 0.0 : bids[0].first; }

  double best_ask() const { return asks.empty() ? 0.0 : asks[0].first; }

  double mid_price() const {
    if (bids.empty() || asks.empty())
      return 0.0;
    return (best_bid() + best_ask()) / 2.0;
  }

  double spread() const {
    if (bids.empty() || asks.empty())
      return 0.0;
    return best_ask() - best_bid();
  }

  double spread_bps() const {
    double mid = mid_price();
    if (mid == 0.0)
      return 0.0;
    return (spread() / mid) * 10000.0;
  }
};

// Price cache for efficient lookups
struct PriceCache {
  double price;
  uint64_t timestamp_us;

  bool is_stale(uint64_t now_us, uint64_t max_age_us = 5'000'000) const {
    return (now_us - timestamp_us) > max_age_us;
  }
};

// Orderbook cache
struct OrderbookCache {
  OrderbookData data;
  uint64_t timestamp_us;

  bool is_stale(uint64_t now_us, uint64_t max_age_us = 5'000'000) const {
    return (now_us - timestamp_us) > max_age_us;
  }
};

class HotSpineExtendedReader {
public:
  explicit HotSpineExtendedReader(
      const std::string &shm_name = "/btquant_hotspine");
  ~HotSpineExtendedReader();

  // Initialization
  bool is_attached() const { return reader_ && reader_->isAttached(); }

  // Load symbol mappings
  bool load_symbol_mappings(const std::string &filepath);

  // Trade reading
  std::optional<TradeData> poll_trade();
  std::vector<TradeData> read_all_trades();

  // Orderbook reading
  std::optional<OrderbookData> poll_orderbook();
  std::vector<OrderbookData> read_all_orderbooks();

  // Price lookups (cached)
  std::optional<double> get_latest_price(const std::string &exchange,
                                         const std::string &symbol);

  // Orderbook lookups (cached)
  std::optional<OrderbookData> get_latest_orderbook(const std::string &exchange,
                                                    const std::string &symbol);

  // Recent trades lookup
  std::vector<TradeData> get_recent_trades(const std::string &exchange,
                                           const std::string &symbol,
                                           size_t limit = 100);

  // Multi-exchange price comparison
  std::map<std::string, double>
  get_all_exchange_prices(const std::string &symbol);

  // Multi-exchange orderbook comparison
  std::map<std::string, OrderbookData>
  get_all_exchange_orderbooks(const std::string &symbol);

  // Statistics
  uint64_t get_trades_read() const { return trades_read_; }
  uint64_t get_orderbooks_read() const { return orderbooks_read_; }
  uint64_t get_current_time_us() const;

  // Buffer monitoring
  std::pair<uint64_t, uint64_t> get_buffer_utilization() const;
  uint64_t get_lost_count() const;

private:
  std::unique_ptr<HotSpineReader> reader_;
  SymbolRegistry &registry_;

  // Caches
  std::map<std::string, PriceCache> price_cache_; // "exchange:symbol" -> price
  std::map<std::string, OrderbookCache> orderbook_cache_;
  std::map<std::string, std::deque<TradeData>> trade_history_; // Recent trades

  // Statistics
  uint64_t trades_read_ = 0;
  uint64_t orderbooks_read_ = 0;

  // Configuration
  size_t max_trade_history_ = 1000;
  uint64_t cache_max_age_us_ = 5'000'000; // 5 seconds

  // Helper functions
  std::string make_cache_key(const std::string &exchange,
                             const std::string &symbol) const;

  TradeData resolve_trade(const HotTrade &trade);
  OrderbookData resolve_orderbook(const HotOrderbookSnapshot &snapshot);

  void update_caches(const TradeData &trade);
  void update_caches(const OrderbookData &orderbook);
  void cleanup_stale_caches();
};

} // namespace BTQuant