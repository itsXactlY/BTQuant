import time

class LatencyProbe:
    def __init__(self):
        self.max_ns = 0

    def observe(self, exchange_ts):
        now = time.time_ns()
        delta = now - exchange_ts
        self.max_ns = max(self.max_ns, delta)
