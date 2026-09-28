#ifndef BTQUANT_VPVR_WIDGET_HPP
#define BTQUANT_VPVR_WIDGET_HPP

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Volume Profile / Visible Range (VPVR) — horizontal histogram of total
// volume traded at each price level. Highlights:
//   * POC (Point of Control) — price with the most traded volume.
//   * Value Area High / Low — prices that bound 70% of total volume
//     (single-pass expanding from POC outward until 70% reached).
//   * VWAP line — horizontal marker at the volume-weighted average price.
//
// Aggregates on-the-fly from snap.recent_trades. For higher-frequency
// production data the MarketDataProcessor could grow a per-candle price-
// level histogram; for now the rolling 256-trade buffer is enough.
class VPVRWidget {
public:
    VPVRWidget();
    ~VPVRWidget();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    void setPriceBucketTicks(int t) { m_priceBucketTicks = t; }
    void setValueAreaPct(double pct) { m_valueAreaPct = pct; }  // 0.0–1.0, default 0.70

    bool showWindow = true;

private:
    int m_priceBucketTicks = 1;
    double m_valueAreaPct = 0.70;
    bool m_initialized = false;

    ::btquant::MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
