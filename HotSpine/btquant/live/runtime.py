from btquant.hotspine.reader import HotSpineReader
from btquant.live.strategy_adapter import BTStrategyAdapter

class LiveRuntime:
    def __init__(self, strategy_cls):
        self.adapter = BTStrategyAdapter(strategy_cls)
        self.reader = HotSpineReader()

    def run(self):
        while True:
            t = self.reader.poll()
            if t:
                self.adapter.on_trade(t)
