# import time
# from ccapi import EventHandler, SessionOptions, SessionConfigs, Session, Subscription, Event
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# if __name__ == "__main__":
#     option = SessionOptions()
#     config = SessionConfigs()
#     session = Session(option, config)

#     exchange = "okx"
#     instrument = "BTC-USDT"
#     subscription = Subscription(
#         exchange,
#         instrument,
#         "MARKET_DEPTH",
#         "MARKET_DEPTH_MAX=400&CONFLATE_INTERVAL_MILLISECONDS=100",
#     )
#     session.subscribe(subscription)

#     # --- Figure with 4 subplots (2x2) ---
#     sns.set(style="darkgrid")
#     fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
#     ax_depth = axes[0, 0]       # ECDF depth
#     ax_cum = axes[0, 1]         # cumulative volume curves
#     ax_imb = axes[1, 0]         # imbalance over time
#     ax_spread = axes[1, 1]      # spread over time

#     startTime = time.time()
#     time_hist = []
#     imbalance_hist = []
#     spread_hist = []

#     while True:
#         bids = {"price": [], "size": []}
#         asks = {"price": [], "size": []}
#         eventList = session.getEventQueue().purge()

#         if eventList:
#             event = eventList[-1]
#             if event.getType() == Event.Type_SUBSCRIPTION_DATA:
#                 for message in event.getMessageList():
#                     for element in message.getElementList():
#                         elementNameValueMap = element.getNameValueMap()
#                         for name, value in elementNameValueMap.items():
#                             if name == "BID_PRICE":
#                                 bids["price"].append(float(value))
#                             if name == "BID_SIZE":
#                                 bids["size"].append(float(value))
#                             if name == "ASK_PRICE":
#                                 asks["price"].append(float(value))
#                             if name == "ASK_SIZE":
#                                 asks["size"].append(float(value))

#                     if not bids["price"] or not asks["price"]:
#                         continue  # nothing to plot yet

#                     # --- basic microstructure stats ---
#                     best_bid = max(bids["price"])
#                     best_ask = min(asks["price"])
#                     spread = best_ask - best_bid
#                     total_bid_vol = sum(bids["size"])
#                     total_ask_vol = sum(asks["size"])
#                     denom = total_bid_vol + total_ask_vol
#                     imbalance = (total_bid_vol - total_ask_vol) / denom if denom > 0 else 0.0

#                     t_rel = time.time() - startTime
#                     time_hist.append(t_rel)
#                     imbalance_hist.append(imbalance)
#                     spread_hist.append(spread)

#                     # --- sort by price for cumulative curves ---
#                     bid_sorted_idx = np.argsort(bids["price"])
#                     ask_sorted_idx = np.argsort(asks["price"])
#                     bid_prices = np.array(bids["price"])[bid_sorted_idx]
#                     bid_sizes = np.array(bids["size"])[bid_sorted_idx]
#                     ask_prices = np.array(asks["price"])[ask_sorted_idx]
#                     ask_sizes = np.array(asks["size"])[ask_sorted_idx]
#                     bid_cum = np.cumsum(bid_sizes[::-1])[::-1]   # from best bid downwards
#                     ask_cum = np.cumsum(ask_sizes)              # from best ask upwards

#                     # --- 1) ECDF depth (top-left) ---
#                     ax_depth.clear()
#                     ax_depth.set_title(
#                         f"{instrument} Depth ECDF\n{exchange.title()} @ {message.getTimeISO()}"
#                     )
#                     ax_depth.set_xlabel("Price")
#                     ax_depth.set_ylabel("Volume (ECDF count)")

#                     sns.ecdfplot(
#                         x="price",
#                         weights="size",
#                         stat="count",
#                         complementary=True,
#                         data={"price": bids["price"], "size": bids["size"]},
#                         ax=ax_depth,
#                         color="g",
#                         legend=False,
#                     )
#                     sns.ecdfplot(
#                         x="price",
#                         weights="size",
#                         stat="count",
#                         data={"price": asks["price"], "size": asks["size"]},
#                         ax=ax_depth,
#                         color="r",
#                         legend=False,
#                     )

#                     # --- 2) Cumulative volume curves (top-right) ---
#                     ax_cum.clear()
#                     ax_cum.set_title("Cumulative Depth (Size vs Price)")
#                     ax_cum.set_xlabel("Price")
#                     ax_cum.set_ylabel("Cumulative Size")

#                     # bids: depth from best bid downward in price
#                     ax_cum.step(
#                         bid_prices,
#                         bid_cum,
#                         where="post",
#                         color="g",
#                         label="Bids",
#                     )
#                     # asks: depth from best ask upward in price
#                     ax_cum.step(
#                         ask_prices,
#                         ask_cum,
#                         where="post",
#                         color="r",
#                         label="Asks",
#                     )
#                     ax_cum.legend(loc="best")

#                     # --- 3) Order book imbalance over time (bottom-left) ---
#                     ax_imb.clear()
#                     ax_imb.set_title("Order Book Imbalance")
#                     ax_imb.set_xlabel("Time since start (s)")
#                     ax_imb.set_ylabel("Imbalance")
#                     ax_imb.plot(time_hist, imbalance_hist, color="b")
#                     ax_imb.axhline(0.0, color="gray", linestyle="--", linewidth=1)

#                     # --- 4) Spread over time (bottom-right) ---
#                     ax_spread.clear()
#                     ax_spread.set_title("Best Bid/Ask Spread")
#                     ax_spread.set_xlabel("Time since start (s)")
#                     ax_spread.set_ylabel("Spread (price units)")
#                     ax_spread.plot(time_hist, spread_hist, color="m")

#         plt.pause(0.05)
#         if time.time() - startTime > 99999999999:
#             break

#     session.stop()
#     print("Bye")


import time
from ccapi import SessionOptions, SessionConfigs, Session, Subscription, Event
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------
# Exchange universe (your reduced, more liquid set)
# ---------------------------------------------------
EXCHANGE_INSTRUMENT = {
    # USDT-margined spot / perp venues
    "binance": "BTCUSDT",
    "binance-usds-futures": "BTCUSDT",
    "binance-coin-futures": "BTCUSDT",
    "okx": "BTC-USDT",
    "gateio": "BTC_USDT",
    "gateio-perpetual-futures": "BTC_USDT",
    "bitget": "BTCUSDT",
    "bitget-futures": "BTCUSDT",

    # USD-quoted venues
    "coinbase": "BTC-USD",
    "bitstamp": "btcusd",
    "kraken": "XBT/USD",
    "kraken-futures": "PI_XBTUSD",
    "bitfinex": "tBTCUSD",
    "bitmex": "XBTUSD",
    "bybit": "BTCUSDT",
    "deribit": "BTC-PERPETUAL",
}

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DEPTH_OPTIONS = "MARKET_DEPTH_MAX=200&CONFLATE_INTERVAL_MILLISECONDS=200"
SIZE_GRID = [10_000, 50_000, 100_000, 250_000]  # notionals in quote currency
MAX_TIME = 99999999999  # effectively infinite

# ---------------------------------------------------
# Helper: compute VWAP execution price for a given notional
# ---------------------------------------------------
def vwap_for_notional(order_book, side, notional_target, ref_mid):
    """
    Walk a single book side to fill `notional_target` in quote currency.
    side: "buy" uses asks; "sell" uses bids.
    Returns VWAP execution price, or None if insufficient depth.
    """
    if side == "buy":
        prices = np.array(order_book["asks"]["price"])
        sizes = np.array(order_book["asks"]["size"])
        # buy: sort asks ascending
        idx = np.argsort(prices)
    else:
        prices = np.array(order_book["bids"]["price"])
        sizes = np.array(order_book["bids"]["size"])
        # sell: sort bids descending
        idx = np.argsort(prices)[::-1]

    prices = prices[idx]
    sizes = sizes[idx]

    remaining = notional_target
    cost = 0.0
    filled_qty = 0.0

    for p, q in zip(prices, sizes):
        max_notional_here = p * q
        if max_notional_here <= 0:
            continue
        take_notional = min(remaining, max_notional_here)
        if take_notional <= 0:
        #     nothing more to fill
            break
        take_qty = take_notional / p
        cost += take_qty * p
        filled_qty += take_qty
        remaining -= take_notional
        if remaining <= 1e-9:
            break

    if remaining > 1e-6 or filled_qty <= 0:
        return None

    vwap_price = cost / filled_qty
    return vwap_price

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    option = SessionOptions()
    config = SessionConfigs()
    session = Session(option, config)

    # Build multi-exchange subscriptions with explicit correlation IDs
    subscription_list = []
    for ex, instr in EXCHANGE_INSTRUMENT.items():
        corr_id = f"{ex}|{instr}"
        sub = Subscription(ex, instr, "MARKET_DEPTH", DEPTH_OPTIONS, corr_id)
        subscription_list.append(sub)

    for sub in subscription_list:
        session.subscribe(sub)

    # Per-exchange books
    books = {
        ex: {
            "bids": {"price": [], "size": []},
            "asks": {"price": [], "size": []},
            "last_time": None,
        }
        for ex in EXCHANGE_INSTRUMENT.keys()
    }

    sns.set(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    ax_depth = axes[0, 0]   # global ECDF vs NBBO mid (bps)
    ax_spread = axes[0, 1]  # per-exchange effective spread (bps)
    ax_impact = axes[1, 0]  # best cross-venue slippage vs size
    ax_disp = axes[1, 1]    # cross-exchange mid dispersion over time

    start_time = time.time()
    time_hist = []
    nbbo_spread_hist = []
    dispersion_time = []
    dispersion_hist = []

    while True:
        # ---------------------------------------------------
        # Ingest new events and update books
        # ---------------------------------------------------
        event_list = session.getEventQueue().purge()
        if event_list:
            event = event_list[-1]
            if event.getType() == Event.Type_SUBSCRIPTION_DATA:
                for message in event.getMessageList():
                    corr_id_list = message.getCorrelationIdList()
                    if not corr_id_list:
                        continue
                    corr_id = corr_id_list[0]
                    if isinstance(corr_id, (list, tuple)):
                        if not corr_id:
                            continue
                        corr_id = corr_id[0]

                    try:
                        ex, instr = corr_id.split("|", 1)
                    except ValueError:
                        continue
                    if ex not in books:
                        continue

                    bids = {"price": [], "size": []}
                    asks = {"price": [], "size": []}

                    for element in message.getElementList():
                        nv = element.getNameValueMap()
                        for name, value in nv.items():
                            v = float(value)
                            if name == "BID_PRICE":
                                bids["price"].append(v)
                            elif name == "BID_SIZE":
                                bids["size"].append(v)
                            elif name == "ASK_PRICE":
                                asks["price"].append(v)
                            elif name == "ASK_SIZE":
                                asks["size"].append(v)

                    if bids["price"] and bids["size"]:
                        books[ex]["bids"] = bids
                    if asks["price"] and asks["size"]:
                        books[ex]["asks"] = asks
                    books[ex]["last_time"] = message.getTimeISO()

        # ---------------------------------------------------
        # Build cross-exchange snapshot metrics
        # ---------------------------------------------------
        all_bid_prices = []
        all_bid_sizes = []
        all_ask_prices = []
        all_ask_sizes = []

        per_ex_best_bid = {}
        per_ex_best_ask = {}

        for ex, data in books.items():
            bids = data["bids"]
            asks = data["asks"]
            if not bids["price"] or not asks["price"]:
                continue

            all_bid_prices.extend(bids["price"])
            all_bid_sizes.extend(bids["size"])
            all_ask_prices.extend(asks["price"])
            all_ask_sizes.extend(asks["size"])

            per_ex_best_bid[ex] = max(bids["price"])
            per_ex_best_ask[ex] = min(asks["price"])

        if (
            all_bid_prices
            and all_ask_prices
            and per_ex_best_bid
            and per_ex_best_ask
        ):
            # NBBO
            nbbo_best_bid = max(per_ex_best_bid.values())
            nbbo_best_ask = min(per_ex_best_ask.values())
            nbbo_mid = 0.5 * (nbbo_best_bid + nbbo_best_ask)
            now_rel = time.time() - start_time
            nbbo_spread = nbbo_best_ask - nbbo_best_bid
            time_hist.append(now_rel)
            nbbo_spread_hist.append(nbbo_spread)

            # -----------------------------------------------
            # 1) Global depth ECDF vs NBBO mid (bps)
            # -----------------------------------------------
            bid_bps = [1e4 * (p - nbbo_mid) / nbbo_mid for p in all_bid_prices]
            ask_bps = [1e4 * (p - nbbo_mid) / nbbo_mid for p in all_ask_prices]

            ax_depth.clear()
            ax_depth.set_title("Global Depth ECDF vs NBBO mid (bps)")
            ax_depth.set_xlabel("Price offset (bps)")
            ax_depth.set_ylabel("Volume ECDF (count)")

            sns.ecdfplot(
                x=bid_bps,
                weights=all_bid_sizes,
                stat="count",
                complementary=True,
                ax=ax_depth,
                color="g",
                legend=False,
            )
            sns.ecdfplot(
                x=ask_bps,
                weights=all_ask_sizes,
                stat="count",
                ax=ax_depth,
                color="r",
                legend=False,
            )

            # -----------------------------------------------
            # 2) Per-exchange effective spread vs NBBO (bps)
            # -----------------------------------------------
            ax_spread.clear()
            ax_spread.set_title("Per-exchange Effective Spread vs NBBO (bps)")
            ax_spread.set_ylabel("Spread (bps)")

            ex_names = []
            ex_spreads_bps = []
            mid_offsets_bps = []

            for ex in per_ex_best_bid:
                b = per_ex_best_bid[ex]
                a = per_ex_best_ask[ex]
                m_ex = 0.5 * (b + a)
                eff_spread_bps = 1e4 * (a - b) / nbbo_mid
                mid_offset_bps = 1e4 * (m_ex - nbbo_mid) / nbbo_mid

                ex_names.append(ex)
                ex_spreads_bps.append(eff_spread_bps)
                mid_offsets_bps.append(mid_offset_bps)

            ax_spread.bar(ex_names, ex_spreads_bps, color="b")
            ax_spread.set_xticklabels(ex_names, rotation=90, fontsize=8)

            # -----------------------------------------------
            # 3) Best cross-venue buy slippage vs notional size
            # -----------------------------------------------
            ax_impact.clear()
            ax_impact.set_title("Best Cross-venue Buy Slippage vs Size")
            ax_impact.set_xlabel("Notional size (quote)")
            ax_impact.set_ylabel("Slippage (bps vs NBBO mid)")

            best_buy_slippage = []
            for notional in SIZE_GRID:
                best_bps = None
                for ex, data in books.items():
                    bids = data["bids"]
                    asks = data["asks"]
                    if not bids["price"] or not asks["price"]:
                        continue
                    exec_price = vwap_for_notional(
                        data,
                        side="buy",
                        notional_target=notional,
                        ref_mid=nbbo_mid,
                    )
                    if exec_price is None:
                        continue
                    slippage_bps = 1e4 * (exec_price - nbbo_mid) / nbbo_mid
                    if best_bps is None or slippage_bps < best_bps:
                        best_bps = slippage_bps
                best_buy_slippage.append(best_bps if best_bps is not None else np.nan)

            ax_impact.plot(SIZE_GRID, best_buy_slippage, marker="o", color="c")

            # -----------------------------------------------
            # 4) Cross-exchange mid dispersion (std of bps)
            # -----------------------------------------------
            mid_offsets_bps_arr = np.array(mid_offsets_bps)
            if mid_offsets_bps_arr.size > 1:
                dispersion = float(np.std(mid_offsets_bps_arr))
                dispersion_time.append(now_rel)
                dispersion_hist.append(dispersion)

            ax_disp.clear()
            ax_disp.set_title("Cross-exchange Mid Dispersion (std of bps)")
            ax_disp.set_xlabel("Time since start (s)")
            ax_disp.set_ylabel("Std of mid offsets (bps)")
            if dispersion_time:
                ax_disp.plot(dispersion_time, dispersion_hist, color="m")

        plt.pause(0.1)
        if time.time() - start_time > MAX_TIME:
            break

    session.stop()
    print("Bye")
