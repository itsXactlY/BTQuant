# HotSpine Enhancement Plan: Market Type Filtering and Symbol Whitelisting

## Overview

This plan addresses the requirements to ensure that only spot market subscriptions are processed and stored in the HotSpine system. The enhancements will include market type validation, symbol whitelisting, and improved SQL integration.

## Current State Analysis

### Key Findings

1. **Configuration Validation**: The CCAPI configuration correctly specifies spot markets only
2. **Subscription Logic**: ExchangeConnectionManager and MarketDataCollector respect the configuration
3. **Hotpath and SQL Delivery**: Need enhancements to ensure only intended subscriptions (spot markets) are processed

### Current Implementation Gaps

1. **HotTrade Structure**: Missing market_type field to identify spot vs futures trades
2. **HotSpineConfig**: No market type filtering or symbol whitelisting configuration
3. **HotSpineReader**: No market type validation to filter non-spot trades
4. **HotSpineSQLIntegration**: Symbol mapping is placeholder, no market type filtering
5. **HotSpineRuntime**: No subscription filtering against whitelist

## Proposed Architecture

### 1. Enhanced HotTrade Structure

```python
class HotTrade(ctypes.Structure):
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),
        ("ts_local", ctypes.c_uint64),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
        ("symbol_id", ctypes.c_uint32),
        ("side", ctypes.c_uint8),
        ("market_type", ctypes.c_uint8),  # NEW: 0=spot, 1=futures, 2=other
    ]
```

### 2. Extended HotSpineConfig

```python
@dataclass
class HotSpineConfig:
    # ... existing fields ...
    
    # NEW: Market type filtering
    market_type_filter: str = "spot"  # "spot", "futures", or "all"
    
    # NEW: Symbol whitelisting
    symbol_whitelist: Optional[List[str]] = None  # List of allowed symbol IDs
    
    # NEW: Enhanced symbol mapping
    symbol_mapping: Optional[Dict[int, Dict[str, str]]] = None  # symbol_id -> {symbol, market_type}
```

### 3. Market Type Validation in HotSpineReader

```python
def poll_trade(self) -> Optional[HotTrade]:
    trade = self._lib.hotspine_reader_poll_trade(self._reader_ptr, ctypes.byref(trade))
    
    # NEW: Market type validation
    if trade and self.config.market_type_filter != "all":
        market_type = self._get_market_type_from_symbol(trade.symbol_id)
        if market_type != self.config.market_type_filter:
            self._metrics['filtered_trades'] += 1
            return None
    
    return trade

def _get_market_type_from_symbol(self, symbol_id: int) -> str:
    """Get market type from symbol mapping or default to spot"""
    if self.config.symbol_mapping and symbol_id in self.config.symbol_mapping:
        return self.config.symbol_mapping[symbol_id].get('market_type', 'spot')
    return 'spot'
```

### 4. Enhanced SQL Integration

```python
def _convert_hottrade_to_dict(self, trade: HotTrade) -> Dict[str, Any]:
    # Use symbol mapping for accurate symbol representation
    symbol_info = self._get_symbol_info(trade.symbol_id)
    
    return {
        'timestamp': trade.ts_exchange * 1000,
        'exchange': 'hotspine',
        'symbol': symbol_info['symbol'] if symbol_info else f'symbol_{trade.symbol_id}',
        'market_type': symbol_info['market_type'] if symbol_info else 'spot',
        'trade_id': f'hotspine_{trade.ts_exchange}_{trade.symbol_id}',
        'price': float(trade.price),
        'quantity': float(trade.size),
        'side': 'buy' if trade.side == 0 else 'sell',
        'is_buyer_maker': None
    }

def _get_symbol_info(self, symbol_id: int) -> Optional[Dict[str, str]]:
    """Get symbol information from mapping"""
    if self.config.symbol_mapping and symbol_id in self.config.symbol_mapping:
        return self.config.symbol_mapping[symbol_id]
    return None
```

### 5. Subscription Filtering in HotSpineRuntime

```python
def on_trade(self, trade: HotTrade):
    # NEW: Symbol whitelist validation
    if self.config.symbol_whitelist:
        symbol_str = f"symbol_{trade.symbol_id}"
        if symbol_str not in self.config.symbol_whitelist:
            self._runtime_metrics['filtered_trades'] += 1
            return
    
    # NEW: Market type validation at runtime level
    market_type = self._get_market_type_from_symbol(trade.symbol_id)
    if market_type != self.config.market_type_filter:
        self._runtime_metrics['filtered_trades'] += 1
        return
    
    # Process trade normally
    self._process_trade(trade)
```

## Implementation Plan

### Phase 1: Core Enhancements

1. **Update HotTrade Structure**
   - Add market_type field to HotTrade ctypes structure
   - Update all HotTrade usage to handle the new field
   - Ensure backward compatibility

2. **Enhance HotSpineConfig**
   - Add market_type_filter field
   - Add symbol_whitelist field
   - Enhance symbol_mapping to include market_type
   - Update environment variable loading
   - Add validation for new fields

3. **Implement Market Type Validation in HotSpineReader**
   - Add market type filtering logic in poll_trade()
   - Add helper method to get market type from symbol
   - Add metrics for filtered trades
   - Update batch reading to respect filtering

### Phase 2: SQL Integration Enhancements

4. **Enhance SQL Integration**
   - Update _convert_hottrade_to_dict() to use symbol mapping
   - Add market type to stored trade data
   - Add symbol mapping validation
   - Update debugging methods to include market type

5. **Add Symbol Management Utilities**
   - Create symbol mapping helper methods
   - Add symbol whitelist validation
   - Create utility methods for symbol lookup

### Phase 3: Runtime Enhancements

6. **Add Subscription Filtering in HotSpineRuntime**
   - Add symbol whitelist validation in on_trade()
   - Add market type validation at runtime level
   - Add metrics for filtered trades
   - Update error handling for filtered trades

7. **Enhance Metrics and Monitoring**
   - Add filtered_trades metric to HotSpineReader
   - Add filtered_trades metric to HotSpineRuntime
   - Update monitoring to track filtering statistics
   - Add logging for filtered trades

### Phase 4: Testing and Documentation

8. **Create Test Cases**
   - Test market type filtering with different configurations
   - Test symbol whitelisting functionality
   - Test backward compatibility
   - Test error handling for invalid configurations

9. **Update Documentation**
   - Update README with new configuration options
   - Add examples for market type filtering
   - Add examples for symbol whitelisting
   - Update API documentation

10. **Create Example Configurations**
    - Create example configuration files
    - Add environment variable examples
    - Create integration test examples

## Configuration Examples

### Spot Market Only Configuration

```python
config = HotSpineConfig(
    market_type_filter="spot",
    symbol_whitelist=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    symbol_mapping={
        1: {"symbol": "BTCUSDT", "market_type": "spot"},
        2: {"symbol": "ETHUSDT", "market_type": "spot"},
        3: {"symbol": "BNBUSDT", "market_type": "spot"}
    }
)
```

### Environment Variable Configuration

```bash
export HOTSPINE_MARKET_TYPE_FILTER="spot"
export HOTSPINE_SYMBOL_WHITELIST='["BTCUSDT", "ETHUSDT"]'
export HOTSPINE_SYMBOL_MAPPING='{"1": {"symbol": "BTCUSDT", "market_type": "spot"}, "2": {"symbol": "ETHUSDT", "market_type": "spot"}}'
```

## Backward Compatibility

All changes will maintain backward compatibility:

1. **HotTrade Structure**: New field will be added at the end to maintain memory layout
2. **Configuration**: New fields will have sensible defaults
3. **API**: Existing methods will continue to work without modification
4. **Behavior**: Default behavior will be to allow all trades (current behavior)

## Metrics and Monitoring

New metrics will be added to track filtering:

- `filtered_trades`: Number of trades filtered by market type or symbol whitelist
- `filtering_efficiency`: Percentage of trades filtered vs total trades
- `market_type_distribution`: Distribution of market types in received trades

## Error Handling

Enhanced error handling will be added:

- Invalid market type configurations
- Invalid symbol whitelist configurations
- Missing symbol mappings
- Conflicting configurations

## Testing Strategy

Comprehensive tests will be created:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test the complete data flow
3. **Performance Tests**: Ensure filtering doesn't impact performance
4. **Regression Tests**: Ensure existing functionality still works
5. **Edge Case Tests**: Test with invalid configurations and edge cases

## Timeline

The implementation will follow this sequence:

1. Core structure updates (HotTrade, HotSpineConfig)
2. Reader-level filtering
3. SQL integration enhancements
4. Runtime-level filtering
5. Testing and validation
6. Documentation and examples

## Success Criteria

The implementation will be considered successful when:

1. Only spot market trades are processed when configured
2. Symbol whitelisting effectively filters trades
3. SQL storage includes accurate market type information
4. Performance impact is minimal (<5% overhead)
5. All existing functionality continues to work
6. Comprehensive test coverage is achieved
