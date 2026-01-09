# BTQuant Manipulation Detector - Dynamic Configuration Architecture

## Overview

This document describes the refactored dynamic configuration system for the BTQuant Manipulation Detector. The system has been enhanced to eliminate all hardcoded values and support runtime discovery of exchanges, symbols, and detector configurations.

## Key Improvements

### 1. Dynamic Configuration Loading

The system now uses a hierarchical configuration discovery mechanism:

```
Configuration Discovery Order:
1. Environment variables (highest priority)
2. User config directory: ~/.config/btquant/
3. Project config directory: ./config/
4. Built-in defaults (lowest priority)
```

#### Environment Variable Overrides

| Variable | Description | Default |
|----------|-------------|---------|
| `BTQUANT_CONFIG_PATH` | Path to main config directory | `./config/` |
| `BTQUANT_SHARED_MEMORY_PATH` | Path to shared memory segments | `/btquant_hotspine` |
| `BTQUANT_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARN, ERROR) | `INFO` |
| `BTQUANT_POLL_INTERVAL_MS` | Monitoring polling interval in milliseconds | `1000` |
| `BTQUANT_BUFFER_SIZE` | Shared memory buffer size | `65536` |

### 2. Plugin-Based Detector Architecture

New detectors can be added without modifying core code by:

1. Creating a detector class inheriting from `DetectorPlugin`
2. Implementing required virtual methods
3. Registering via `DetectorRegistry::registerDetector()`

#### Detector Plugin Interface

```cpp
class DetectorPlugin {
public:
    virtual ~DetectorPlugin() = default;
    virtual std::string getName() const = 0;
    virtual void configure(const YAML::Node& config) = 0;
    virtual std::vector<ManipulationAlert> analyze(
        const MarketSnapshot& snapshot) = 0;
    virtual void onOrderBookUpdate(const OrderBook& book) = 0;
};
```

#### Built-in Detectors

| Detector | Description | Config File |
|----------|-------------|-------------|
| `liquidity_imbalance` | Detects liquidity gaps in order books | `detectors/liquidity_imbalance.yaml` |
| `spoofing` | Identifies fake orders to manipulate prices | `detectors/spoofing.yaml` |
| `spread_arbitrage` | Detects cross-exchange arbitrage opportunities | `detectors/spread_arbitrage.yaml` |
| `stop_hunt` | Identifies stop-loss hunting patterns | `detectors/stop_hunt.yaml` |
| `whale_frontrun` | Detects large order front-running | `detectors/whale_frontrun.yaml` |

### 3. Runtime Symbol and Exchange Discovery

The system no longer requires hardcoded symbol mappings. Instead:

- **Exchange Discovery**: Queries exchange APIs at startup to discover available trading pairs
- **Symbol Mapping**: Loaded dynamically from `symbol_mapping.json` or fetched from exchange
- **ID Assignment**: Exchange-specific IDs are assigned at runtime

#### Symbol Mapping Configuration (`config/symbol_mapping.json`)

```json
{
  "exchanges": {
    "binance": {
      "base_id": 1,
      "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    },
    "bybit": {
      "base_id": 401,
      "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    }
  },
  "shared_memory": {
    "segment_name": "btquant_hotspine",
    "buffer_size": 65536
  }
}
```

### 4. Dynamic HotSpineReader

The HotSpineReader now supports dynamic configuration:

```cpp
HotSpineReader::Config config;
config.sharedMemoryPath = "/btquant_hotspine";
config.bufferSize = 65536;
config.pollIntervalMs = 1000;
config.batchSize = 100;

HotSpineReader reader(config);
reader.attach();  // Dynamic attachment to shared memory
```

### 5. Configuration Schema Validation

All configuration files are validated against schemas:

```cpp
ConfigValidator validator;
validator.addRequiredField("exchanges", YAML::NodeType::Map);
validator.addRequiredField("detectors", YAML::NodeType::Map);
validator.addOptionalField("logging", YAML::NodeType::Map);

ValidationResult result = validator.validate(config);
if (!result.isValid) {
    throw ConfigValidationError(result.errorMessages);
}
```

### 6. Comprehensive Logging System

Logging is dynamically configurable at runtime:

```yaml
logging:
  level: DEBUG
  output: both  # console, file, or both
  file_path: ./logs/btquant.log
  max_file_size_mb: 100
  max_kept_files: 10
```

#### Log Levels

- `DEBUG`: Detailed information for debugging
- `INFO`: General operational information
- `WARN`: Warning conditions
- `ERROR`: Error conditions
- `CRITICAL`: Critical errors requiring immediate attention

## Configuration Files Structure

```
config/
├── btquant.yaml              # Main configuration file
├── symbol_mapping.json       # Exchange-symbol mappings
├── detectors/
│   ├── liquidity_imbalance.yaml
│   ├── spoofing.yaml
│   ├── spread_arbitrage.yaml
│   ├── stop_hunt.yaml
│   └── whale_frontrun.yaml
└── logging.yaml              # Logging configuration
```

## Example Main Configuration (`config/btquant.yaml`)

```yaml
# BTQuant Manipulation Detector Configuration
# All values can be overridden via environment variables

# Shared Memory Configuration
shared_memory:
  path: "/btquant_hotspine"
  buffer_size: 65536
  segment_name: "btquant_hotspine"

# Monitoring Configuration
monitoring:
  poll_interval_ms: 1000
  batch_size: 100
  max_order_book_depth: 50
  price_precision: 8
  volume_precision: 8

# Exchange Configuration (can be auto-discovered)
exchanges:
  binance:
    enabled: true
    base_id: 1
  bybit:
    enabled: true
    base_id: 401
  okx:
    enabled: true
    base_id: 301
  gate:
    enabled: true
    base_id: 501
  kraken:
    enabled: true
    base_id: 201
  coinbase:
    enabled: true
    base_id: 101

# Detector Configuration
detectors:
  liquidity_imbalance:
    enabled: true
    threshold_percent: 5.0
    min_volume_btc: 1.0
  spoofing:
    enabled: true
    order_age_threshold_seconds: 60
    size_to_bbo_ratio: 0.5
  spread_arbitrage:
    enabled: true
    min_spread_percent: 0.5
    min_volume_btc: 0.1
  stop_hunt:
    enabled: true
    price_deviation_percent: 2.0
    volume_spike_threshold: 5.0
  whale_frontrun:
    enabled: true
    min_order_size_btc: 10.0
    front_run_detection_window_ms: 500

# Logging Configuration
logging:
  level: INFO
  output: console
  file_path: ./logs/btquant.log
  max_file_size_mb: 100
  max_kept_files: 10

# Alert Configuration
alerts:
  enabled: true
  channels:
    - console
    - file
  min_severity: INFO
```

## Adding New Exchanges

To add a new exchange without code changes:

1. Create a new section in `config/symbol_mapping.json`:

```json
{
  "exchanges": {
    "new_exchange": {
      "base_id": 1001,
      "symbols": ["BTC-USDT", "ETH-USDT"]
    }
  }
}
```

2. Ensure the exchange data producer writes to the shared memory segment
3. Restart the monitor - the new exchange will be discovered automatically

## Adding New Detectors

To create a new manipulation detector:

1. Create a new header file in `include/detectors/`:

```cpp
// include/detectors/new_detector.hpp
#pragma once
#include "detector_plugin.hpp"

class NewDetector : public DetectorPlugin {
public:
    std::string getName() const override { return "new_detector"; }
    void configure(const YAML::Node& config) override;
    std::vector<ManipulationAlert> analyze(const MarketSnapshot& snapshot) override;
    void onOrderBookUpdate(const OrderBook& book) override;
};
```

2. Implement the detector in `src/detectors/new_detector.cpp`
3. Add configuration file `config/detectors/new_detector.yaml`
4. Register the detector in the main application:

```cpp
DetectorRegistry::registerDetector<NewDetector>("new_detector");
```

## Performance Considerations

### Dynamic Buffer Sizing

Buffer sizes are automatically calculated based on:

- Available system memory
- Number of exchanges
- Expected throughput

```cpp
auto bufferSize = DynamicBufferCalculator::calculateOptimalSize(
    systemMemoryMB: 4096,
    exchangeCount: 6,
    expectedTps: 10000
);
```

### Polling Optimization

The monitoring loop uses adaptive polling:

- Base interval: configurable (default 1000ms)
- Adaptive adjustment based on data rate
- Batch processing for efficiency

## Error Handling and Graceful Degradation

The system implements comprehensive error handling:

1. **Optional Component Failures**: Non-critical detectors failing won't crash the system
2. **Fallback Configurations**: Missing config files use built-in defaults
3. **Connection Retries**: Automatic reconnection with exponential backoff
4. **Circuit Breaker**: Stops polling failed exchanges after threshold

## Testing

### Unit Tests

```bash
cd tests/new
cmake --build build --target unit_tests
./build/unit_tests
```

### Integration Tests

```bash
./BUILD_AND_RUN.sh --integration-test
```

### Configuration Validation

```bash
./BUILD_AND_RUN.sh --validate-config
```

## Migration from Hardcoded Configuration

If you have existing hardcoded symbol mappings, migrate them to `config/symbol_mapping.json`:

```json
{
  "version": "1.0",
  "exchanges": {
    "binance": {
      "base_id": 1,
      "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    }
  }
}
```

## Troubleshooting

### Configuration Not Loading

1. Check environment variables: `echo $BTQUANT_CONFIG_PATH`
2. Verify config file syntax: `yaml lint config/btquant.yaml`
3. Enable debug logging: `export BTQUANT_LOG_LEVEL=DEBUG`

### Exchange Not Appearing

1. Verify exchange is enabled in `btquant.yaml`
2. Check symbol mapping exists in `symbol_mapping.json`
3. Ensure data producer is writing to shared memory

### Detector Not Working

1. Check detector is enabled in config
2. Verify required thresholds are set
3. Enable debug logging for detector output

## API Reference

See [API Reference Documentation](../docs/technical/api-reference.md) for detailed API documentation.

## License

This project is proprietary software. See LICENSE file for details.
