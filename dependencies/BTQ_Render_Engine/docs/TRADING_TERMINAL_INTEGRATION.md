# BTQuant Trading Terminal - Integration Guide

## Overview

Das BTQuant Trading Terminal ist eine professionelle DOM-basierte Trading-Plattform, die auf dem BTQ_Render_Engine aufbaut. Es bietet Echtzeit-Marktdaten-Visualisierung mit GPU-beschleunigtem Rendering und C++26 Parallelisierung.

## Architektur

### Komponenten

```
BTQ_Render_Engine/
├── include/components/
│   ├── dom_surface_panel.hpp      # DOM Surface Heatmap Panel
│   ├── footprint_panel.hpp        # Footprint Chart Panel
│   ├── chart_panel.hpp            # Candlestick Chart Panel
│   ├── orderbook_panel.hpp        # Orderbook Panel
│   └── volume_profile_panel.hpp   # Volume Profile Panel
├── src/components/
│   ├── dom_surface_panel.cpp
│   ├── footprint_panel.cpp
│   ├── chart_panel.cpp
│   └── volume_profile_panel.cpp
├── src/main_trading_terminal.cpp  # Hauptanwendung
└── shaders/
    ├── lob_heatmap.comp           # DOM Heatmap Compute Shader
    ├── footprint_vert.glsl        # Footprint Vertex Shader
    └── footprint_frag.glsl        # Footprint Fragment Shader
```

### Rendering Pipeline

1. **Data Input**: HotSpine Shared Memory → Market Data Processor
2. **Parallel Updates**: std::execution::par für Panel-Updates
3. **GPU Rendering**: Vulkan/OpenGL Backend
4. **Display**: 60 FPS+ mit <1ms Frame Time

## Integration mit HotSpine

### HotSpine Shared Memory Setup

Das Terminal liest Echtzeit-Marktdaten aus HotSpine Shared Memory:

```cpp
#include "hotspine_data_bridge.hpp"

// HotSpine Data Bridge initialisieren
BTQ::Data::HotSpineDataBridge hotspine_bridge;
hotspine_bridge.initialize("/dev/shm/hotspine_trades");

// Marktdaten abonnieren
hotspine_bridge.subscribe("ETHUSDT", [](const auto& update) {
    // Update Orderbook
    // Update DOM Surface
    // Update Footprint
});
```

### HotSpine Data Format

```cpp
struct HotSpineTrade {
    uint64_t timestamp;
    double price;
    double size;
    Side side;  // BID or ASK
    std::string symbol;
};
```

### Performance-Optimierung

- **Lock-Free Updates**: `std::atomic` für Orderbook-Updates
- **Batch Reading**: 1000+ Trades pro Batch
- **Zero-Copy**: Direkter Shared Memory Zugriff

## Integration mit BigBrainCentral

### SQL Query Client

```cpp
#include "bigbraincentral_client.hpp"

// Client initialisieren
auto sql = BigBrainCentral::Client::get();

// Historische Candle-Daten abfragen
std::vector<OHLCV> candles = sql->query_candles(
    "ETHUSDT",      // Symbol
    "15m",          // Timeframe
    now() - 24h,    // Startzeit
    now()           // Endzeit
);

// Indikatoren berechnen
std::vector<double> sma_20 = sql->calculate_sma(candles, 20);
std::vector<double> ema_9 = sql->calculate_ema(candles, 9);
```

### Supported Queries

- `query_candles(symbol, timeframe, start, end)` - Historische OHLCV-Daten
- `query_trades(symbol, start, end)` - Historische Trades
- `calculate_sma(data, period)` - Simple Moving Average
- `calculate_ema(data, period)` - Exponential Moving Average
- `calculate_rsi(data, period)` - Relative Strength Index

## Panel-Konfiguration

### DOM Surface Panel

```cpp
#include "components/dom_surface_panel.hpp"

// DOM Surface Panel erstellen
auto dom_panel = std::make_shared<BTQ::Components::DOMSurfacePanel>();
dom_panel->set_bounds({480, 0, 672, 720});
dom_panel->set_symbol("ETHUSDT");
dom_panel->set_price_range(3000.0, 3100.0);
dom_panel->set_time_window(2048);  // 2048 Zeitpunkte

// Large Order Detection konfigurieren
dom_panel->set_large_order_threshold(10.0);  // 10x Median
dom_panel->set_whale_detection_enabled(true);
```

### Footprint Panel

```cpp
#include "components/footprint_panel.hpp"

// Footprint Panel erstellen
auto footprint_panel = std::make_shared<BTQ::Components::FootprintPanel>();
footprint_panel->set_bounds({480, 720, 672, 360});
footprint_panel->set_symbol("ETHUSDT");
footprint_panel->set_grid_size(60, 100);  // 60 Minuten × 100 Price Levels
footprint_panel->set_color_mode(ColorMode::Delta);  // Delta-basierte Farben
```

### Candlestick Panel

```cpp
#include "components/chart_panel.hpp"

// Candlestick Panel erstellen
auto chart_panel = std::make_shared<BTQ::Components::ChartPanel>();
chart_panel->set_bounds({0, 0, 480, 720});
chart_panel->set_symbol("ETHUSDT");
chart_panel->set_timeframe("15m");

// Indikatoren hinzufügen
chart_panel->add_indicator<BTQ::Indicators::SMA>(20);
chart_panel->add_indicator<BTQ::Indicators::EMA>(9);
chart_panel->add_indicator<BTQ::Indicators::RSI>(14);

// Fibonacci Retracements
chart_panel->enable_fibonacci(true);
chart_panel->set_fibonacci_levels({0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0});
```

## Performance Benchmarks

### Benchmark-Ergebnisse

| Metrik | Ergebnis | Target | Status |
|--------|----------|--------|--------|
| DOM Heatmap Generation | 0.263 ms | <5.0 ms | ✓ PASSED |
| Footprint Aggregation | 0.006 ms | <2.0 ms | ✓ PASSED |
| Candlestick Rendering | 0.002 ms | <3.0 ms | ✓ PASSED |
| Full Frame Simulation | 0.282 ms | <16.0 ms | ✓ PASSED |

### Geschätzte FPS: 3544 (Target: 60 FPS)

### Performance-Optimierungen

1. **Lock-Free Orderbook Updates**
   ```cpp
   std::atomic_ref<OrderbookLevel> level(hotspine_data[price_idx]);
   level.store({price, size}, std::memory_order_release);
   ```

2. **Parallel Indicator Calculation**
   ```cpp
   std::for_each(std::execution::par_unseq, indicators.begin(), indicators.end(),
       [&candles](auto& indicator) {
           indicator.calculate(candles);
       });
   ```

3. **GPU Compute for Heatmap**
   - Offload Heatmap-Generierung zu GPU Compute Shadern
   - Direkte Texture-Updates ohne CPU → GPU Transfer
   - Target: 60 FPS mit 13,970 Orderbook Updates/sec

4. **Memory Pools**
   ```cpp
   std::pmr::unsynchronized_pool_resource trade_pool;
   std::pmr::vector<Trade> trades(&trade_pool);
   ```

## Build & Run

### Voraussetzungen

- C++26 Compiler (GCC 14+, Clang 18+)
- CMake 3.28+
- Vulkan SDK oder OpenGL 4.6+
- HotSpine Shared Memory
- BigBrainCentral SQL Client

### Build

```bash
cd dependencies/BTQ_Render_Engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Run

```bash
./btquant_terminal
```

### Konfiguration

Die Konfiguration wird über `btquant.yaml` gesteuert:

```yaml
terminal:
  window:
    title: "BTQuant Advanced Terminal"
    width: 1920
    height: 1080
    vsync: true
  
  panels:
    - type: candlestick
      bounds: [0, 0, 480, 720]
      symbol: "ETHUSDT"
      timeframe: "15m"
    
    - type: dom_surface
      bounds: [480, 0, 672, 720]
      symbol: "ETHUSDT"
      price_range: [3000.0, 3100.0]
    
    - type: footprint
      bounds: [480, 720, 672, 360]
      symbol: "ETHUSDT"
      grid_size: [60, 100]
  
  hotspine:
    shm_path: "/dev/shm/hotspine_trades"
    batch_size: 1000
  
  bigbraincentral:
    host: "localhost"
    port: 5432
    database: "btquant"
```

## Troubleshooting

### Keine HotSpine-Daten

```bash
# Prüfen ob Shared Memory existiert
ls -la /dev/shm/hotspine_trades

# HotSpine Dump anzeigen
./shm_dump
```

### Performance-Probleme

```bash
# Benchmark ausführen
./benchmark_trading_terminal

# Performance Monitor aktivieren
export BTQ_PERFORMANCE_MONITOR=1
./btquant_terminal
```

### Vulkan-Fehler

```bash
# Vulkan-Validation Layer aktivieren
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
./btquant_terminal
```

## API Reference

### DOMSurfacePanel

```cpp
class DOMSurfacePanel {
public:
    void set_symbol(const std::string& symbol);
    void set_price_range(double min, double max);
    void set_time_window(size_t samples);
    void set_large_order_threshold(double multiplier);
    void set_whale_detection_enabled(bool enabled);
    
    std::vector<LargeOrder> get_large_orders() const;
    std::array<double, 200> get_bid_liquidity() const;
    std::array<double, 200> get_ask_liquidity() const;
};
```

### FootprintPanel

```cpp
class FootprintPanel {
public:
    void set_symbol(const std::string& symbol);
    void set_grid_size(size_t cols, size_t rows);
    void set_color_mode(ColorMode mode);
    
    std::vector<std::vector<Cell>> get_grid() const;
    double get_max_delta() const;
};
```

### ChartPanel

```cpp
class ChartPanel {
public:
    void set_symbol(const std::string& symbol);
    void set_timeframe(const std::string& timeframe);
    
    template<typename Indicator>
    void add_indicator(typename Indicator::Config config);
    
    void enable_fibonacci(bool enabled);
    void set_fibonacci_levels(const std::vector<double>& levels);
    
    std::vector<OHLCV> get_candles() const;
};
```

## License

BTQuant Trading Terminal - Copyright 2024

## Support

Für Fragen und Support: support@btquant.io
