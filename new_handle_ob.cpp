void MarketDataProcessor::handleOrderbookMessage(const ccapi::Message &msg) {

  const auto &cid_list = msg.getCorrelationIdList();
  const std::string cid = cid_list.empty() ? "" : cid_list[0];
  auto parts = split(cid, ":");

  std::string exchange = parts.size() > 0 ? parts[0] : "";
  std::string symbol = parts.size() > 1 ? parts[1] : "";
  std::string market_type = parts.size() > 2 ? parts[2] : "spot";

  if (exchange.empty() || symbol.empty()) {
    std::cerr
        << "[" << getCurrentTimestamp()
        << "][ERROR] handleOrderbookMessage: empty exchange/symbol in CID: "
        << cid << std::endl;
    return;
  }

  const std::string key = exchange + ":" + symbol + ":" + market_type;
  const auto &elements = msg.getElementList();
  if (elements.empty())
    return;

  auto tp = msg.getTime();
  int64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      tp.time_since_epoch())
                      .count();

  std::vector<HotSpine::HotOrderbookLevel> hot_bids;
  std::vector<HotSpine::HotOrderbookLevel> hot_asks;

  for (const auto &el : elements) {
    auto bid_p_s = getAny(el, {"BID_PRICE", "BEST_BID_PRICE"});
    auto bid_q_s = getAny(el, {"BID_SIZE", "BEST_BID_SIZE"});
    auto ask_p_s = getAny(el, {"ASK_PRICE", "BEST_ASK_PRICE"});
    auto ask_q_s = getAny(el, {"ASK_SIZE", "BEST_ASK_SIZE"});

    if (!bid_p_s.empty() && !bid_q_s.empty()) {
      bool okp = false, okq = false;
      double p = safeParseDouble("bid_price", bid_p_s, okp);
      double q = safeParseDouble("bid_size", bid_q_s, okq);
      if (okp && okq && q > 0.0) {
        hot_bids.push_back({p, q});
      }
    }

    if (!ask_p_s.empty() && !ask_q_s.empty()) {
      bool okp = false, okq = false;
      double p = safeParseDouble("ask_price", ask_p_s, okp);
      double q = safeParseDouble("ask_size", ask_q_s, okq);
      if (okp && okq && q > 0.0) {
        hot_asks.push_back({p, q});
      }
    }
  }

  if (hot_bids.empty() && hot_asks.empty())
    return;

  MarketData::OrderbookSnapshot ob;
  ob.timestamp_us = ts_us;
  ob.exchange = exchange;
  ob.symbol = symbol;
  ob.market_type = market_type;

  if (!enable_exclusive_hotspine_) {
    // Only build JSON if we are NOT in exclusive mode (avoid overhead)
    auto build_side_json =
        [](const std::vector<HotSpine::HotOrderbookLevel> &side) {
          std::ostringstream oss;
          oss << "[";
          for (std::size_t i = 0; i < side.size(); ++i) {
            if (i > 0)
              oss << ",";
            oss << "[" << side[i].price << "," << side[i].size << "]";
          }
          oss << "]";
          return oss.str();
        };
    ob.bids_json = build_side_json(hot_bids);
    ob.asks_json = build_side_json(hot_asks);
    ob.checksum.clear();

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    orderbook_buffer_.push_back(ob);
  }

  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    pair_stats_[key].orderbooks++;
    ++stats_.orderbooks_received;
  }

  if (hotspine_writer_) {
    hotspine_writer_->writeOrderbook(ob, hot_bids, hot_asks);
  }

  if (!enable_exclusive_hotspine_) {
    flushOrderbooksIfNeeded();
  }
}
