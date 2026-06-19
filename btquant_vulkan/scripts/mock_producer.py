#!/usr/bin/env python3
"""
BTQuant mock data producer.

Writes a realistic stream of market data to /dev/shm/btquant_hotspine using the
expected binary format (per /home/alca/projects/PubBTQuant/btquant_vulkan/CLAUDE.md):

  Header @ 0x0000: "UQTB" magic (4 bytes) + version (2 bytes LE) + symbol_count (2 bytes LE)
  Symbols @ 0x1000: array of 128-byte records:
      bid_price (double, 8 bytes)
      ask_price (double, 8 bytes)
      bid_size  (double, 8 bytes)
      ask_size  (double, 8 bytes)
      timestamp (uint64, 8 bytes)
      seq       (uint32, 4 bytes)
      flags     (uint32, 4 bytes)
      padding to 128 bytes

Symbols JSON at /dev/shm/btquant_symbols.json:
  {"symbols": [{"exchange": "binance", "symbol": "BTC/USDT"}, ...]}

Usage:
  python3 mock_producer.py [--symbol BTC/USDT] [--exchange binance] [--interval 100]
"""

import argparse
import json
import math
import os
import random
import struct
import time
from pathlib import Path

HOTSPINE_PATH = "/dev/shm/btquant_hotspine"
SYMBOLS_JSON_PATH = "/dev/shm/btquant_symbols.json"
HEADER_MAGIC = b"UQTB"
HEADER_SIZE = 0x1000
SYMBOL_RECORD_SIZE = 128


def write_symbols_json(symbols: list[tuple[str, str]]) -> None:
    payload = {"symbols": [
        {"id": i, "exchange": ex, "symbol": sym}
        for i, (ex, sym) in enumerate(symbols)
    ]}
    Path(SYMBOLS_JSON_PATH).write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="BTQuant mock hotspine producer")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--interval-ms", type=int, default=100,
                        help="Update interval (default 100 ms = 10 Hz)")
    parser.add_argument("--price-start", type=float, default=67500.0,
                        help="Starting mid price (default 67500.0 — realistic BTC)")
    parser.add_argument("--duration-sec", type=int, default=0,
                        help="Run duration in seconds (0 = forever)")
    args = parser.parse_args()

    # Synthetic state — geometric Brownian motion around price_start.
    price = args.price_start
    seq = 0
    spread = 0.5  # half-spread
    symbols = [(args.exchange, args.symbol)]

    # Write symbols JSON first (DataSpine reads this on open).
    write_symbols_json(symbols)
    print(f"[mock_producer] wrote {len(symbols)} symbols to {SYMBOLS_JSON_PATH}")

    # Create + truncate the SHM file.
    Path(HOTSPINE_PATH).unlink(missing_ok=True)
    fd = os.open(HOTSPINE_PATH, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
    # Pre-allocate: header + 1 symbol record
    os.ftruncate(fd, HEADER_SIZE + SYMBOL_RECORD_SIZE)
    print(f"[mock_producer] created {HOTSPINE_PATH} ({HEADER_SIZE + SYMBOL_RECORD_SIZE} bytes)")

    # Write header
    os.lseek(fd, 0, os.SEEK_SET)
    header = HEADER_MAGIC + struct.pack("<HH", 1, 1) + b"\x00" * (HEADER_SIZE - 8)
    os.write(fd, header)
    print(f"[mock_producer] wrote header (UQTB v1, 1 symbol)")

    start = time.monotonic()
    try:
        while True:
            if args.duration_sec and (time.monotonic() - start) > args.duration_sec:
                print(f"[mock_producer] duration reached, exiting")
                break

            # GBM-style price evolution: dS = sigma * S * dW
            dt = args.interval_ms / 1000.0
            sigma = 0.0002
            dW = random.gauss(0, math.sqrt(dt))
            price = max(price * math.exp(sigma * dW), 1.0)

            # Spread varies with volatility (wider when noise is high).
            current_spread = spread * (1.0 + abs(dW) * 50.0)
            bid = price - current_spread / 2
            ask = price + current_spread / 2

            # Bid/ask sizes — random in [0.05, 5.0] lots.
            bid_size = random.uniform(0.05, 5.0)
            ask_size = random.uniform(0.05, 5.0)

            ts_us = int(time.time() * 1_000_000)
            # Write symbol record at HEADER_SIZE + idx * SYMBOL_RECORD_SIZE
            offset = HEADER_SIZE + 0 * SYMBOL_RECORD_SIZE
            os.lseek(fd, offset, os.SEEK_SET)
            # Layout (matches HotSpineEntry in data_spine.hpp):
            #   4 doubles (bid, ask, bid_size, ask_size) + uint64 (timestamp)
            #   + 2×uint32 (seq, flags) = 48 bytes, padded to SYMBOL_RECORD_SIZE (128)
            payload = struct.pack("<ddddQII",
                                  bid, ask, bid_size, ask_size,
                                  ts_us, seq, 0)
            payload += b"\x00" * (SYMBOL_RECORD_SIZE - len(payload))
            os.write(fd, payload)
            seq += 1

            if seq % 20 == 0:
                print(f"[mock_producer] seq={seq} price={price:.2f} "
                      f"bid={bid:.2f} ask={ask:.2f} "
                      f"bid_size={bid_size:.2f} ask_size={ask_size:.2f}")

            time.sleep(args.interval_ms / 1000.0)
    except KeyboardInterrupt:
        print(f"\n[mock_producer] interrupted at seq={seq}")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
