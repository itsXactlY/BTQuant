#!/usr/bin/env python3
"""
Example HotSpine strategy for btq_live_runtime

This demonstrates how to use the HotSpine reader with a simple trading strategy.
"""

import sys
import os
import time
import signal
import threading

# Add the dependencies to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dependencies'))

from backtrader.hotspine.reader import HotSpineReader, HotTrade, HotSpineRuntime


class SimpleMovingAverageStrategy:
    """
    Example strategy that calculates a simple moving average
    and makes trading decisions based on price crossing the SMA.
    """
    
    def __init__(self):
        self.sma_period = 10
        self.sma_value = 0
        self.price_history = []
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.trade_count = 0
        self.start_time = time.time()
        
    def next(self):
        """Called for each new trade"""
        if not hasattr(self, 'data') or not self.data:
            return
        
        trade = self.data
        self.trade_count += 1
        
        # Add price to history
        self.price_history.append(trade.price)
        
        # Keep only the last N prices
        if len(self.price_history) > self.sma_period:
            self.price_history = self.price_history[-self.sma_period:]
        
        # Calculate SMA
        if len(self.price_history) == self.sma_period:
            self.sma_value = sum(self.price_history) / self.sma_period
        
        # Simple trading logic: buy when price crosses above SMA, sell when crosses below
        if len(self.price_history) == self.sma_period and self.position == 0:
            if trade.price > self.sma_value * 1.001:  # 0.1% above SMA
                print(f"📈 BUY signal: Price {trade.price:.2f} > SMA {self.sma_value:.2f}")
                self.position = 1
                self.broker.buy(size=1.0, price=trade.price)
            elif trade.price < self.sma_value * 0.999:  # 0.1% below SMA
                print(f"📉 SELL signal: Price {trade.price:.2f} < SMA {self.sma_value:.2f}")
                self.position = -1
                self.broker.sell(size=1.0, price=trade.price)
        
        # Print stats periodically
        if self.trade_count % 100 == 0:
            elapsed = time.time() - self.start_time
            rate = self.trade_count / elapsed if elapsed > 0 else 0
            print(f"Processed {self.trade_count} trades (rate: {rate:.1f}/sec) | "
                  f"Price: {trade.price:.2f} | SMA: {self.sma_value:.2f} | Position: {'LONG' if self.position == 1 else 'SHORT' if self.position == -1 else 'FLAT'}")


class PrintAllTradesStrategy:
    """
    Simple strategy that just prints all incoming trades
    """
    
    def __init__(self):
        self.trade_count = 0
        self.start_time = time.time()
        
    def next(self):
        """Called for each new trade"""
        if not hasattr(self, 'data') or not self.data:
            return
        
        trade = self.data
        self.trade_count += 1
        
        # Print trade info
        side = "BUY" if trade.side == 0 else "SELL"
        print(f"Trade #{self.trade_count}: {side} {trade.size} @ {trade.price} (symbol: {trade.symbol_id})")
        
        # Print stats periodically
        if self.trade_count % 100 == 0:
            elapsed = time.time() - self.start_time
            rate = self.trade_count / elapsed if elapsed > 0 else 0
            print(f"--- Processed {self.trade_count} trades at {rate:.1f} trades/sec ---")


def run_example():
    """Run the example HotSpine strategy"""
    print("HotSpine Example Strategy")
    print("=" * 40)
    print("This example demonstrates HotSpine reader integration with btq_live_runtime")
    print()
    
    # Let user choose strategy
    print("Choose a strategy:")
    print("1. Simple Moving Average Strategy (trading signals)")
    print("2. Print All Trades Strategy (debug/monitoring)")
    print("3. Run performance test")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        strategy_class = SimpleMovingAverageStrategy
        print("\nRunning Simple Moving Average Strategy...")
    elif choice == "2":
        strategy_class = PrintAllTradesStrategy
        print("\nRunning Print All Trades Strategy...")
    elif choice == "3":
        run_performance_test()
        return
    else:
        print("Invalid choice")
        return
    
    # Let user choose mode
    print("\nChoose runtime mode:")
    print("1. Single trade mode (lowest latency)")
    print("2. Batch mode (higher throughput)")
    
    mode_choice = input("Enter choice (1-2): ").strip()
    batch_mode = mode_choice == "2"
    
    print(f"\nStarting HotSpine runtime with {'batch' if batch_mode else 'single'} mode...")
    print("Press Ctrl+C to stop...\n")
    
    # Create and run the runtime
    try:
        runtime = HotSpineRuntime(strategy_class)
        runtime.run(batch_mode=batch_mode)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"❌ Runtime error: {e}")


def run_performance_test():
    """Run a performance test to measure HotSpine reader throughput"""
    print("\nRunning HotSpine Performance Test...")
    print("=" * 40)
    
    class PerformanceTestStrategy:
        def __init__(self):
            self.trade_count = 0
            self.start_time = time.time()
            self.last_report_time = self.start_time
            
        def next(self):
            self.trade_count += 1
            
            # Report progress every second
            current_time = time.time()
            if current_time - self.last_report_time >= 1.0:
                elapsed = current_time - self.start_time
                recent_rate = 1.0 / (current_time - self.last_report_time)
                overall_rate = self.trade_count / elapsed if elapsed > 0 else 0
                
                print(f"Rate: {recent_rate:.0f}/sec (avg: {overall_rate:.0f}/sec) | Total: {self.trade_count}")
                self.last_report_time = current_time
    
    print("Testing single trade mode...")
    try:
        runtime = HotSpineRuntime(PerformanceTestStrategy)
        
        # Run for 5 seconds
        def run_for_duration():
            runtime.run(batch_mode=False)
        
        thread = threading.Thread(target=run_for_duration, daemon=True)
        thread.start()
        
        time.sleep(5)
        print("\nSingle trade mode test completed")
        
    except Exception as e:
        print(f"Single trade mode test failed: {e}")
    
    print("\nTesting batch mode...")
    try:
        runtime = HotSpineRuntime(PerformanceTestStrategy)
        
        # Run for 5 seconds
        def run_for_duration():
            runtime.run(batch_mode=True)
        
        thread = threading.Thread(target=run_for_duration, daemon=True)
        thread.start()
        
        time.sleep(5)
        print("\nBatch mode test completed")
        
    except Exception as e:
        print(f"Batch mode test failed: {e}")


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n⏹️  Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    run_example()