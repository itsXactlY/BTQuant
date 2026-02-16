# ChartSuperNode

The `ChartSuperNode` is a central coordination component for all chart-related activities in the PubBTQuant trading platform. It serves as the primary hub for managing multiple chart instances, synchronizing data across charts, and coordinating advanced analytical features.

## Features

- **Centralized Chart Management**: Manages multiple chart instances with unified control
- **Synchronization**: Provides mechanisms for synchronizing views, crosshairs, and data across multiple charts
- **Advanced Analytics Integration**: Built-in support for TPO (Time-Price Opportunity) analysis and VWAP calculations
- **Indicator Calculation**: Efficient calculation and caching of technical indicators
- **Thread-Safe Operations**: Atomic operations and mutex protection for multi-threaded environments
- **Performance Monitoring**: Built-in metrics for tracking update times and resource usage

## Architecture

The ChartSuperNode sits at the center of the chart ecosystem and coordinates with:

- `ChartManager`: For low-level chart instance management
- `TPOEngine`: For advanced time-price analysis
- `MarketDataProcessor`: For real-time market data feeds
- `HotSpineDataBridge`: For shared memory data access
- Multiple `ChartPanel` instances: For UI rendering

## Usage

```cpp
// Create the ChartSuperNode with required dependencies
auto super_node = std::make_shared<BTQuant::ChartSuperNode>(bridge, processor);

// Initialize the node
super_node->initialize();

// Create a chart
uint32_t chart_id = super_node->create_chart("BTC-USDT", "Binance", 1, 
                                             RenderEngine::TimeFrame::TF_1MIN);

// Register for synchronization callbacks
super_node->set_chart_sync_callback([](const ChartSuperNode::SyncData& data) {
    // Handle chart synchronization
});

// Perform regular updates
super_node->update();
```

## Thread Safety

The ChartSuperNode implements thread-safe operations using:
- Atomic variables for state management
- Mutex protection for shared resources
- RAII-style locking mechanisms
- Safe execution templates for critical sections