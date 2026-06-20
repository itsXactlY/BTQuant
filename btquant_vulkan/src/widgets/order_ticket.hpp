#ifndef BTQUANT_ORDER_TICKET_HPP
#define BTQUANT_ORDER_TICKET_HPP

#include <functional>
#include <string>

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Order Ticket widget — interactive ticket form for placing new orders.
// Pulls live reference price from MarketDataProcessor (latest trade), lets
// the user set side (buy/sell), order type (market/limit), quantity, and
// (for limit orders) a limit price. Computes estimated fill cost, fee, and
// total notional in real time. Submit fires the registered callback and
// logs the intent via BTQ_LOG_INFO — no real exchange wiring in the MVP.
//
// Pure math (fee calc + fill estimate) is exposed for testing.
class OrderTicket {
public:
    // Submit callback — fired when the user confirms the ticket. The
    // string is a human-readable summary suitable for logging.
    using SubmitFn = std::function<void(const std::string& summary)>;
    void setSubmitFn(SubmitFn fn) { m_submit = std::move(fn); }

    // Bind a live data source for the reference price field. Passing
    // nullptr disconnects (the form falls back to the manual ref price).
    void setMarketData(::btquant::MarketDataProcessor* data) { m_data = data; }

    void render();

    // ---- Pure math (test surface) ----

    // Fee in quote currency. feeBps is in basis points (10 = 0.10%).
    static double computeFee(double size, double price, double feeBps);

    // Fill estimate for a market order. Buy fills at ask, sell at bid.
    // For a limit order fills at the limit if price crosses, else ref.
    // Returns the *effective* price per unit.
    static double estimateFillPrice(bool isBuy, bool isLimit,
                                    double limitPrice, double refPrice,
                                    double slippageBps);

    // Total cost to the trader = |size| * effective_price + fee.
    // Buy: positive cost. Sell: negative (proceeds) — caller decides sign.
    static double computeTotalCost(double size, double effectivePrice,
                                   double feeBps);

    // Accessors for the current draft state (for hotkey rebinding etc.).
    bool isOpen() const  { return m_open; }
    void setOpen(bool v) { m_open = v; }
    bool isBuy()  const  { return m_sideIsBuy; }
    bool isLimit() const { return m_typeIsLimit; }
    double quantity() const;
    double limitPrice() const;
    double referencePrice() const;

    // Side setter — used by hotkeys (Alt+B / Alt+S) to flip the
    // ticket's side without going through the render path. Calling
    // setSideBuy(true) makes the next submit a BUY; false = SELL.
    void setSideBuy(bool v) { m_sideIsBuy = v; }

    // Build the submit summary string from the current draft + fire
    // the registered callback. Returns false if the draft can't be
    // submitted (qty or fill price is 0) or if no submit callback is
    // wired. The render path now calls this from the Submit button so
    // the hotkey path and the click path share one code path.
    bool submit();

private:
    void refreshRefPrice();

    bool m_open = false;
    bool m_sideIsBuy  = true;
    bool m_typeIsLimit = false;
    char m_qty  [32] = "0.10";
    char m_limit[32] = "0.00";      // only used when limit
    char m_feeBps[32] = "10";       // 10 bps = 0.10% taker fee
    char m_slipBps[32] = "5";       // 5 bps market slippage estimate

    ::btquant::MarketDataProcessor* m_data = nullptr;
    SubmitFn m_submit;
};

} // namespace btquant::ui

#endif
