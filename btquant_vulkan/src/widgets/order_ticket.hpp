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

    // Last live trade price the refresh path observed (0.0 when no
    // data). Public so tests can assert the plumbing without driving
    // the render loop.
    double liveRefPrice() const { return m_liveRefPrice; }

    // Side setter — used by hotkeys (Alt+B / Alt+S) to flip the
    // ticket's side without going through the render path. Calling
    // setSideBuy(true) makes the next submit a BUY; false = SELL.
    void setSideBuy(bool v) { m_sideIsBuy = v; }

    // Alt-submits-opposite toggle. When on (default), an Alt+click
    // on the Submit button — or Alt+Enter — flips the side first,
    // then submits. The current side is restored after submit so a
    // subsequent normal click submits the original side again. Opt
    // out by calling setAltSubmitsOpposite(false); some traders
    // don't want the Alt key to silently flip on a fat-finger.
    bool altSubmitsOpposite() const         { return m_altSubmitsOpposite; }
    void setAltSubmitsOpposite(bool v)     { m_altSubmitsOpposite = v; }

    // Pure helper: given the current side and whether the user is
    // holding Alt at submit time, what side will actually be submitted?
    // Used by the render path and by tests; safe without an ImGui
    // context (no internal state, just a ternary).
    static bool effectiveSideOnSubmit(bool currentSideIsBuy, bool altDown) {
        // Alt+submit flips the side, plain submit keeps it. This is
        // the inverse of the trader-mental model on a hot-path order
        // ("I want to flatten fast" → Alt-click whichever side is
        // showing).
        return altDown ? !currentSideIsBuy : currentSideIsBuy;
    }

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
    bool m_altSubmitsOpposite = true;   // opt-out: setAltSubmitsOpposite(false)
    char m_qty  [32] = "0.10";
    char m_limit[32] = "0.00";      // only used when limit
    char m_feeBps[32] = "10";       // 10 bps = 0.10% taker fee
    char m_slipBps[32] = "5";       // 5 bps market slippage estimate

    // Cached latest trade price from the data spine. Updated each
    // frame by refreshRefPrice(). For limit orders the live price is
    // shown as a hint but never overwrites the user-edited limit. For
    // market orders it seeds m_limit on the first render so the
    // greyed-out "Ref price (auto)" field has a sensible value.
    double m_liveRefPrice = 0.0;

    ::btquant::MarketDataProcessor* m_data = nullptr;
    SubmitFn m_submit;
};

} // namespace btquant::ui

#endif
