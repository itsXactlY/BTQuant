from datetime import datetime, timedelta    

# Prototype till new backend

class fetch_historical_candles():
    def _fetch(self):
        import pytz
        # data0 is your <Broker>Data -> has ._store and .symbol
        store = self.data._store
        symbol = store.symbol
        now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        since = now - timedelta(minutes=5)
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(now.timestamp() * 1000)
        klines = store.fetch_ohlcv(symbol, '1m', since=since_ms, until=until_ms)
        if not klines:
            return []
        last5 = klines[-5:]
        out = []
        for k in last5:
            ts = datetime.fromtimestamp(k[0] / 1000, tz=pytz.UTC)
            o, h, l, c, v = map(float, k[1:6])
            out.append(dict(
                dt=ts,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            ))
        return out

    def _log_view(self):
        mins = self._fetch()
        if mins:
            for i, mb in enumerate(mins):
                print(
                    f"  [{i}] {mb['dt']:%Y-%m-%d %H:%M} | "
                    f"O={mb['open']:.2f} H={mb['high']:.2f} "
                    f"L={mb['low']:.2f} C={mb['close']:.2f} "
                    f"V={mb['volume']:.6f}"
                )

    def next(self):
        super().next()

        if self.p.backtest:
            return

        if len(self.data) < 2:
            return

        dt_curr = self.data.datetime.datetime(0)
        dt_prev = self.data.datetime.datetime(-1)

        curr_key = (dt_curr.year, dt_curr.month, dt_curr.day, dt_curr.hour, dt_curr.minute)
        prev_key = (dt_prev.year, dt_prev.month, dt_prev.day, dt_prev.hour, dt_prev.minute)

        if curr_key == prev_key:
            return
        self.report_positions()
        self._log_view()