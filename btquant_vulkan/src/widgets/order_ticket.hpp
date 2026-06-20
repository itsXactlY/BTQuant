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

    // Tag accessor — the strategy label that flows into the journal
    // via the submit callback (Sprint #49). Empty string = untagged.
    const char* tag() const { return m_tag; }

    // ---- Last-used tag memory ----
    // After every successful submit, m_tag is copied into m_lastTag
    // (Sprint #52). On the next render of the ticket, the first
    // frame after a fresh open restores m_tag from m_lastTag so the
    // trader doesn't have to retype the strategy label. Opt out via
    // setRememberLastTag(false).
    const char* lastTag() const              { return m_lastTag; }
    bool rememberLastTag() const             { return m_rememberLastTag; }
    void setRememberLastTag(bool v)          { m_rememberLastTag = v; }

    // Test helper — primes m_lastTag to a specific value, simulating
    // "the trader submitted a fill with this tag earlier". Lets tests
    // exercise the auto-restore path without driving the render loop.
    void setLastTagForTest(const char* tag) {
        std::snprintf(m_lastTag, sizeof(m_lastTag), "%s", tag ? tag : "");
    }

    // ---- Last-submitted draft memory (Sprint #57) ----
    //
    // After every successful submit, the current qty + side + type +
    // fee + slippage are copied into the m_last* buffers (the limit
    // price is intentionally NOT copied — limits depend on live
    // market state, not trader preference, so we always pull the
    // fresh ref price instead).
    //
    // On the first render frame after a fresh open AND when ALL the
    // corresponding current buffers are empty AND last* are non-empty,
    // the draft is restored from m_last*. The static latch ensures
    // the restore fires at most once per open cycle. Opt out via
    // setRememberLastDraft(false).
    //
    // resetDraft() preserves m_last* (so the trader can clear the
    // current draft and still get the saved values back on the next
    // open — same contract as m_lastTag).
    double lastQty() const;
    double lastFeeBps() const;
    double lastSlipBps() const;
    bool   lastSideIsBuy() const;
    bool   lastTypeIsLimit() const;
    bool   rememberLastDraft() const;
    void   setRememberLastDraft(bool v);

    // Test helpers — prime the m_last* buffers and the side/type
    // state directly so tests can exercise the auto-restore path
    // without driving the render loop.
    void setLastDraftForTest(double qty, double feeBps, double slipBps,
                             bool isBuy, bool isLimit) {
        std::snprintf(m_lastQty,    sizeof(m_lastQty),    "%.4f", qty);
        std::snprintf(m_lastFeeBps, sizeof(m_lastFeeBps), "%.2f", feeBps);
        std::snprintf(m_lastSlipBps,sizeof(m_lastSlipBps),"%.2f", slipBps);
        m_lastSideIsBuy   = isBuy;
        m_lastTypeIsLimit = isLimit;
    }
    void clearLastDraftForTest() {
        m_lastQty[0] = m_lastFeeBps[0] = m_lastSlipBps[0] = '\0';
        m_lastSideIsBuy   = true;
        m_lastTypeIsLimit = false;
    }

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

    // ---- Persistent-draft controls ----
    // By default, submit() does NOT clear the draft — qty / side /
    // type / limit stay so a scalper can re-fire with one click. Opt
    // in to auto-clear via setClearAfterSubmit(true). The "Reset"
    // button is always available to manually clear.
    bool clearAfterSubmit() const         { return m_clearAfterSubmit; }
    void setClearAfterSubmit(bool v)     { m_clearAfterSubmit = v; }

    // Number of successful submit() calls since construction (or the
    // last explicit reset). Cheap accessor — no recompute cost.
    int  submitCount() const              { return m_submitCount; }
    void resetSubmitCount()               { m_submitCount = 0; }

    // Manually clear the draft to defaults (qty=0.10, side=BUY,
    // type=market, limit=0.00). Mirrors the buffers' initial state
    // so the next render shows a fresh ticket.
    void resetDraft();

    // Test surface for the reset path — the buffers themselves are
    // private, so this re-reads the parsed values to confirm a reset
    // took effect. Returns true if qty reads back as the default
    // (0.10) AND side is BUY AND type is market. Limit price is
    // allowed to be 0.00 (the default for market orders).
    bool isDraftAtDefaults() const;

private:
    void refreshRefPrice();

    bool m_open = false;
    bool m_sideIsBuy  = true;
    bool m_typeIsLimit = false;
    bool m_altSubmitsOpposite = true;   // opt-out: setAltSubmitsOpposite(false)
    bool m_clearAfterSubmit  = false;   // opt-in: setClearAfterSubmit(true)
    bool m_rememberLastTag   = true;    // opt-out: setRememberLastTag(false)
    bool m_rememberLastDraft = true;    // opt-out: setRememberLastDraft(false)
    int  m_submitCount = 0;             // monotonic counter since construction
    char m_qty  [32] = "0.10";
    char m_limit[32] = "0.00";      // only used when limit
    char m_feeBps[32] = "10";       // 10 bps = 0.10% taker fee
    char m_slipBps[32] = "5";       // 5 bps market slippage estimate
    char m_tag[32] = "";            // strategy label (Sprint #49)
    char m_lastTag[32] = "";        // most recent submitted tag (Sprint #52)

    // Last-submitted draft memory (Sprint #57). Mirrors the fields
    // the trader picks deliberately (qty / side / type / fee / slip);
    // the limit price is intentionally NOT mirrored because it
    // depends on live market state. m_last* are seeded by submit()
    // AFTER the optional resetDraft() clears the current buffers,
    // so a clear-after-submit user still gets the next open
    // pre-filled from what they just submitted.
    char m_lastQty[32]    = "";
    char m_lastFeeBps[32] = "";
    char m_lastSlipBps[32]= "";
    bool m_lastSideIsBuy   = true;
    bool m_lastTypeIsLimit = false;

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
