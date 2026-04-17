#!/usr/bin/env python3
"""
Mock Data Producer v3 — Proper struct alignment.
"""
import mmap, os, struct, time, json, random, signal, sys

SHM_PATH = "/dev/shm/btquant"
SYMBOLS_FILE = "/dev/shm/btquant_symbols.json"

HEADER_SIZE = 80
TRADE_SIZE = 40
OB_SIZE = 6464       # alignas(64): 6424 -> 6464
OB_HEADER = 24       # uint64+uint64+uint32+uint8+uint8+uint8[2]
OB_DATA = 6400       # 200*16 + 200*16
OB_PAD = 40          # 6464 - 24 - 6400
TRADE_CAP = 50000
OB_CAP = 2000
TOTAL_SIZE = HEADER_SIZE + TRADE_CAP * TRADE_SIZE + OB_CAP * OB_SIZE

MAGIC = 0x42545155

SYMBOLS = [
    (1,  "BTCUSDT",  83500.0,  0.0003, 0.01),
    (2,  "ETHUSDT",  1580.0,   0.0005, 0.01),
    (3,  "SOLUSDT",  125.0,    0.0008, 0.001),
    (4,  "DOGEUSDT", 0.165,    0.0015, 0.00001),
    (5,  "XRPUSDT",  2.05,     0.0008, 0.0001),
    (6,  "ADAUSDT",  0.62,     0.001,  0.0001),
    (7,  "BNBUSDT",  590.0,    0.0007, 0.01),
    (8,  "LINKUSDT", 12.5,     0.001,  0.001),
    (9,  "AVAXUSDT", 19.5,     0.001,  0.001),
    (10, "SUIUSDT",  2.15,     0.0012, 0.0001),
    (11, "PEPEUSDT", 0.0000072,0.002,  0.0000001),
    (12, "WIFUSDT",  0.42,     0.0015, 0.0001),
    (13, "NEARUSDT", 2.85,     0.001,  0.001),
    (14, "DOTUSDT",  3.95,     0.001,  0.001),
    (15, "ARBUSDT",  0.32,     0.0012, 0.0001),
    (16, "FILUSDT",  2.90,     0.001,  0.001),
    (17, "UNIUSDT",  5.80,     0.001,  0.001),
    (18, "APTUSDT",  5.20,     0.001,  0.001),
    (19, "OPUSDT",   0.72,     0.0012, 0.0001),
    (20, "MKRUSDT",  1250.0,   0.0008, 0.1),
]

running = True
def stop(s, f):
    global running
    running = False
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

def make_ob(sid, mid, tick, ts_ns):
    """Build one orderbook snapshot padded to OB_SIZE."""
    ob = bytearray(OB_SIZE)
    # Header: 24 bytes
    struct.pack_into('<QQIBB2s', ob, 0, ts_ns, ts_ns + 50, sid, 50, 50, b'\x00' * 2)
    # 200 bids at offset 24
    for j in range(200):
        if j < 50:
            struct.pack_into('<dd', ob, OB_HEADER + j * 16,
                mid - (j + 1) * tick * random.uniform(0.5, 1.5),
                random.lognormvariate(2, 1.0))
        else:
            struct.pack_into('<dd', ob, OB_HEADER + j * 16, 0.0, 0.0)
    # 200 asks at offset 24 + 3200 = 3224
    ask_off = OB_HEADER + 200 * 16
    for j in range(200):
        if j < 50:
            struct.pack_into('<dd', ob, ask_off + j * 16,
                mid + (j + 1) * tick * random.uniform(0.5, 1.5),
                random.lognormvariate(2, 1.0))
        else:
            struct.pack_into('<dd', ob, ask_off + j * 16, 0.0, 0.0)
    # Remaining 40 bytes are zero (padding) — already zero from bytearray
    return bytes(ob)

def main():
    global running
    print("=" * 50)
    print("  BTQuant Mock Data Producer v3")
    print("=" * 50)

    # Symbols
    with open(SYMBOLS_FILE, 'w') as f:
        for sid, name, _, _, _ in SYMBOLS:
            f.write(json.dumps({"id": sid, "exchange": "binance", "symbol": name}) + '\n')
    print(f"[OK] {len(SYMBOLS)} symbols")

    # SHM
    with open(SHM_PATH, 'wb') as f:
        f.write(b'\x00' * TOTAL_SIZE)
    fd = os.open(SHM_PATH, os.O_RDWR)
    shm = mmap.mmap(fd, TOTAL_SIZE)
    os.close(fd)

    # Header
    header = bytearray(80)
    struct.pack_into('<IIQQQQQQQQ', header, 0,
        MAGIC, 2, TRADE_CAP, 0, 0, 0, 0, 0, 0, OB_CAP)
    shm[:80] = header

    trades_off = HEADER_SIZE
    obs_off = trades_off + TRADE_CAP * TRADE_SIZE

    # Pre-fill 20000 trades
    print(f"[FILL] 20000 trades...", end=' ', flush=True)
    t0 = time.time()
    prices = {s[0]: s[2] for s in SYMBOLS}
    ts = int(time.time() * 1_000_000)
    buf = bytearray(TRADE_SIZE * 20000)
    for i in range(20000):
        sym = SYMBOLS[i % len(SYMBOLS)]
        sid, _, base, vol, tick = sym
        prices[sid] = max(tick, prices[sid] * (1.0 + random.gauss(0, vol)))
        price = prices[sid]
        size = max(0.001, random.lognormvariate(0, 1.2) * 0.1)
        side = random.randint(0, 1)
        ts += random.randint(100, 50000)
        struct.pack_into('<QQddIB3s', buf, i * TRADE_SIZE,
            ts, ts + random.randint(0, 100), price, size, sid, side, b'\x00' * 3)
    shm[trades_off:trades_off + len(buf)] = buf
    shm[16:24] = struct.pack('<Q', 20000)
    print(f"{time.time()-t0:.2f}s")

    # Pre-fill 500 orderbooks
    print(f"[FILL] 500 orderbooks...", end=' ', flush=True)
    t0 = time.time()
    for i in range(500):
        sym = SYMBOLS[i % len(SYMBOLS)]
        ob = make_ob(sym[0], prices.get(sym[0], sym[2]), sym[4], ts + i * 10000)
        pos = obs_off + i * OB_SIZE
        shm[pos:pos + OB_SIZE] = ob
    shm[40:48] = struct.pack('<Q', 500)
    print(f"{time.time()-t0:.2f}s")

    # Stream — realistic throughput (~200 trades/s)
    print(f"[STREAM] Live data at ~200 trades/s... Ctrl+C to stop")
    wi = 20000
    owi = 500
    count = 0
    last_stat = time.time()
    last_ob = time.time()

    while running:
        now_ns = time.time_ns()
        # Trades batch
        batch = random.randint(3, 8)
        buf = bytearray(TRADE_SIZE * batch)
        for i in range(batch):
            sym = SYMBOLS[random.randint(0, len(SYMBOLS) - 1)]
            sid, _, _, vol, tick = sym
            prices[sid] = max(tick, prices[sid] * (1.0 + random.gauss(0, vol)))
            p = prices[sid]
            sz = max(0.001, random.lognormvariate(0, 1.2) * 0.1)
            s = random.randint(0, 1)
            now_ns += random.randint(100, 10000)
            struct.pack_into('<QQddIB3s', buf, i * TRADE_SIZE,
                now_ns, now_ns + random.randint(0, 100), p, sz, sid, s, b'\x00' * 3)
        pos = trades_off + (wi % TRADE_CAP) * TRADE_SIZE
        end = pos + len(buf)
        limit = trades_off + TRADE_CAP * TRADE_SIZE
        if end <= limit:
            shm[pos:end] = buf
        else:
            first = limit - pos
            shm[pos:limit] = buf[:first]
            shm[trades_off:trades_off + len(buf) - first] = buf[first:]
        wi += batch
        shm[16:24] = struct.pack('<Q', wi)
        count += batch

        # OB every 50ms
        if time.time() - last_ob > 0.05:
            sym = SYMBOLS[random.randint(0, min(4, len(SYMBOLS) - 1))]
            ob = make_ob(sym[0], prices.get(sym[0], sym[2]), sym[4], time.time_ns())
            opos = obs_off + (owi % OB_CAP) * OB_SIZE
            shm[opos:opos + OB_SIZE] = ob
            owi += 1
            shm[40:48] = struct.pack('<Q', owi)
            last_ob = time.time()

        time.sleep(0.05)  # ~200 trades/s max
        # Stats
        if time.time() - last_stat > 2:
            elapsed = time.time() - last_stat
            tps = count / elapsed
            sys.stdout.write(f"\r[STREAM] {tps:,.0f} trades/s | {owi} OBs | BTC=${prices.get(1,0):,.2f}      ")
            sys.stdout.flush()
            count = 0
            last_stat = time.time()

    shm.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
