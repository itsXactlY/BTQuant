#!/usr/bin/env bash
set -euo pipefail

ROOT=hotspine
PY=btquant
CPU_CORE=3

echo ">>> Creating directory layout"

mkdir -p $ROOT/{cpp,include,build}
mkdir -p $PY/{live,hotspine,metrics}

########################################
# 1️⃣ HotSpine C++ ring buffer
########################################

cat > $ROOT/include/hotspine.h <<'EOF'
#pragma once
#include <cstddef>
#include <stdint.h>
#include <atomic>

struct Trade {
    uint64_t ts_ns;
    double price;
    double size;
};

constexpr size_t RING_SIZE = 1 << 20;

struct Ring {
    std::atomic<uint64_t> write_idx;
    std::atomic<uint64_t> read_idx;
    Trade buffer[RING_SIZE];
};

extern "C" {
    Ring* hotspine_init();
    void hotspine_push(Ring*, Trade);
    int hotspine_poll(Ring*, Trade*);
}
EOF

cat > $ROOT/cpp/hotspine.cpp <<'EOF'
#include "hotspine.h"
#include <cstring>
#include <sched.h>
#include <unistd.h>

static Ring ring;

Ring* hotspine_init() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    ring.write_idx.store(0);
    ring.read_idx.store(0);
    return &ring;
}

void hotspine_push(Ring* r, Trade t) {
    uint64_t w = r->write_idx.load(std::memory_order_relaxed);
    r->buffer[w % RING_SIZE] = t;
    r->write_idx.store(w + 1, std::memory_order_release);
}

int hotspine_poll(Ring* r, Trade* out) {
    uint64_t r_idx = r->read_idx.load(std::memory_order_relaxed);
    uint64_t w_idx = r->write_idx.load(std::memory_order_acquire);

    if (r_idx == w_idx) return 0;

    *out = r->buffer[r_idx % RING_SIZE];
    r->read_idx.store(r_idx + 1, std::memory_order_release);
    return 1;
}
EOF

echo ">>> Building HotSpine shared library"

g++ -O3 -march=native -shared -fPIC \
    $ROOT/cpp/hotspine.cpp \
    -I$ROOT/include \
    -o $ROOT/build/libhotspine.so

########################################
# 2️⃣ Python HotSpine reader
########################################

cat > $PY/hotspine/reader.py <<'EOF'
import ctypes, time

lib = ctypes.CDLL("hotspine/build/libhotspine.so")

class Trade(ctypes.Structure):
    _fields_ = [
        ("ts_ns", ctypes.c_uint64),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
    ]

lib.hotspine_init.restype = ctypes.c_void_p
lib.hotspine_poll.argtypes = [ctypes.c_void_p, ctypes.POINTER(Trade)]

class HotSpineReader:
    def __init__(self):
        self.ring = lib.hotspine_init()
        self.trade = Trade()

    def poll(self):
        if lib.hotspine_poll(self.ring, ctypes.byref(self.trade)):
            return self.trade
        return None
EOF

########################################
# 3️⃣ BT Strategy Adapter
########################################

cat > $PY/live/strategy_adapter.py <<'EOF'
class BTStrategyAdapter:
    def __init__(self, strategy_cls):
        self.strategy = strategy_cls.__new__(strategy_cls)
        self.strategy.datas = []
        self.strategy.broker = self
        self.strategy.position = 0
        strategy_cls.__init__(self.strategy)

    def buy(self, size=None, price=None):
        print("BUY", size, price)

    def sell(self, size=None, price=None):
        print("SELL", size, price)

    def on_trade(self, trade):
        self.strategy.data = trade
        self.strategy.next()
EOF

########################################
# 4️⃣ Live runtime (NO SQL)
########################################

cat > $PY/live/runtime.py <<'EOF'
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
EOF

########################################
# 5️⃣ Metrics
########################################

cat > $PY/metrics/latency.py <<'EOF'
import time

class LatencyProbe:
    def __init__(self):
        self.max_ns = 0

    def observe(self, exchange_ts):
        now = time.time_ns()
        delta = now - exchange_ts
        self.max_ns = max(self.max_ns, delta)
EOF

########################################
# 6️⃣ Hard SQL OFF
########################################

cat > $PY/live/guard.py <<'EOF'
import os

if os.environ.get("BTQ_LIVE") == "1":
    raise RuntimeError("SQL access forbidden in live mode")
EOF

echo ">>> HotSpine deployment complete"