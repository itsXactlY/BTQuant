bool HotSpineWriter::writeOrderbook(const MarketData::OrderbookSnapshot& ob, 
                                    const std::vector<HotSpine::HotOrderbookLevel>& bids, 
                                    const std::vector<HotSpine::HotOrderbookLevel>& asks) {
  if (!isHealthy() || orderbooks_buffer_ == nullptr) {
    write_errors_++;
    return false;
  }
  HotSpine::HotOrderbookSnapshot hot_ob;
  hot_ob.ts_exchange = static_cast<uint64_t>(ob.timestamp_us);
  hot_ob.ts_local = getCurrentTimestampMicros();
  hot_ob.symbol_id = getSymbolId(ob.exchange, ob.symbol, ob.market_type);
  
  hot_ob.bids_count = static_cast<uint8_t>(std::min<size_t>(bids.size(), 20));
  for (size_t i = 0; i < hot_ob.bids_count; ++i) {
    hot_ob.bids[i] = bids[i];
  }
  
  hot_ob.asks_count = static_cast<uint8_t>(std::min<size_t>(asks.size(), 20));
  for (size_t i = 0; i < hot_ob.asks_count; ++i) {
    hot_ob.asks[i] = asks[i];
  }

  uint64_t write_idx = header_->orderbook_write_index;
  if (header_->orderbook_capacity == 0) return false;
  uint64_t next_idx = (write_idx + 1) % header_->orderbook_capacity;
  if (next_idx == header_->orderbook_read_index) {
    header_->orderbook_lost_count++;
    write_errors_++;
    return false;
  }
  orderbooks_buffer_[write_idx] = hot_ob;
  header_->orderbook_write_index = next_idx;
  return true;
}
