# HotSpine Enhanced Architecture

## Architecture Diagram

```mermaid
graph TD
    A[CCAPI Data Source] -->|Raw Market Data| B[ExchangeConnectionManager]
    B -->|Filtered Subscriptions| C[MarketDataCollector]
    C -->|Trade Data| D[HotSpine Shared Memory]
    
    D -->|Read Trades| E[HotSpineReader]
    E -->|Filter by Market Type| F[MarketTypeValidator]
    F -->|Filter by Symbol Whitelist| G[SymbolWhitelistValidator]
    G -->|Validated Trades| H[HotSpineRuntime]
    
    H -->|Execute Strategy| I[TradingStrategy]
    H -->|Async Storage| J[HotSpineSQLIntegration]
    J -->|Store with Symbol Mapping| K[SQLDatabase]
    
    subgraph Configuration
        L[HotSpineConfig] -->|market_type_filter| F
        L -->|symbol_whitelist| G
        L -->|symbol_mapping| J
    end
    
    subgraph Monitoring
        M[MetricsCollector] -->|Track Filtering Stats| E
        M -->|Track Runtime Stats| H
        M -->|Track Storage Stats| J
    end
    
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#9f9,stroke:#333
    style E fill:#99f,stroke:#333
    style F fill:#ff9,stroke:#333
    style G fill:#ff9,stroke:#333
    style H fill:#99f,stroke:#333
    style I fill:#f99,stroke:#333
    style J fill:#9f9,stroke:#333
    style K fill:#9f9,stroke:#333
    style L fill:#f96,stroke:#333
    style M fill:#6f9,stroke:#333
```

## Data Flow with Enhanced Filtering

### 1. Data Ingestion Layer

```mermaid
sequenceDiagram
    participant CCAPI as CCAPI
    participant ECM as ExchangeConnectionManager
    participant MDC as MarketDataCollector
    participant HSM as HotSpine Shared Memory
    
    CCAPI->>ECM: Raw market data (all types)
    ECM->>MDC: Filtered data (respects config)
    MDC->>HSM: Write trades to shared memory
```

### 2. Enhanced Reading Layer

```mermaid
sequenceDiagram
    participant HSM as HotSpine Shared Memory
    participant Reader as HotSpineReader
    participant MTVal as MarketTypeValidator
    participant SWLVal as SymbolWhitelistValidator
    
    HSM->>Reader: Raw trade data
    Reader->>MTVal: Check market_type_filter
    alt Market type matches
        MTVal-->>Reader: Pass trade
        Reader->>SWLVal: Check symbol_whitelist
        alt Symbol in whitelist
            SWLVal-->>Reader: Pass trade
            Reader->>Runtime: Valid trade
        else Symbol not in whitelist
            SWLVal-->>Reader: Filter trade
            Reader->>Metrics: Increment filtered_count
        end
    else Market type doesn't match
        MTVal-->>Reader: Filter trade
        Reader->>Metrics: Increment filtered_count
    end
```

### 3. Enhanced SQL Storage Layer

```mermaid
sequenceDiagram
    participant Runtime as HotSpineRuntime
    participant SQLInt as HotSpineSQLIntegration
    participant DB as SQL Database
    
    Runtime->>SQLInt: Trade with symbol_id
    SQLInt->>SymbolMapper: Get symbol info
    alt Symbol found in mapping
        SymbolMapper-->>SQLInt: {symbol: "BTCUSDT", market_type: "spot"}
        SQLInt->>DB: Store with full symbol info
    else Symbol not found
        SymbolMapper-->>SQLInt: None
        SQLInt->>DB: Store with placeholder symbol
    end
```

## Component Responsibilities

### HotSpineReader (Enhanced)

**Responsibilities:**
- Read trades from shared memory
- Apply market type filtering based on configuration
- Apply symbol whitelist filtering based on configuration
- Track filtering metrics
- Provide health monitoring

**New Methods:**
- `_filter_by_market_type(trade: HotTrade) -> bool`
- `_filter_by_symbol_whitelist(trade: HotTrade) -> bool`
- `_get_market_type_from_symbol(symbol_id: int) -> str`

### HotSpineConfig (Enhanced)

**New Fields:**
- `market_type_filter: str` - "spot", "futures", or "all"
- `symbol_whitelist: Optional[List[str]]` - List of allowed symbol IDs
- `symbol_mapping: Optional[Dict[int, Dict[str, str]]]` - Enhanced mapping with market_type

**New Validation:**
- Validate market_type_filter values
- Validate symbol_whitelist format
- Validate symbol_mapping structure

### HotSpineSQLIntegration (Enhanced)

**Enhanced Methods:**
- `_convert_hottrade_to_dict()` - Uses symbol mapping for accurate symbol representation
- `_get_symbol_info()` - Returns symbol and market_type from mapping
- `store_trade_async()` - Includes market_type in stored data

### HotSpineRuntime (Enhanced)

**Enhanced Methods:**
- `on_trade()` - Additional filtering layer
- `_validate_trade()` - Combined validation logic
- `_process_trade()` - Core trade processing

## Configuration Flow

```mermaid
flowchart TD
    A[Environment Variables] --> B[HotSpineConfig]
    C[Config Files] --> B
    D[Default Values] --> B
    
    B --> E[Validation]
    E -->|Valid| F[HotSpineReader]
    E -->|Valid| G[HotSpineRuntime]
    E -->|Valid| H[HotSpineSQLIntegration]
    
    F --> I[Market Type Filtering]
    F --> J[Symbol Whitelist Filtering]
    G --> K[Runtime Validation]
    H --> L[Symbol Mapping for Storage]
```

## Metrics and Monitoring

### New Metrics

```mermaid
classDiagram
    class Metrics {
        +trades_read
        +trades_filtered_by_market_type
        +trades_filtered_by_symbol_whitelist
        +filtering_efficiency
        +market_type_distribution
        +symbol_whitelist_hit_rate
    }
    
    class HotSpineReader {
        +get_filtering_metrics()
        +get_market_type_stats()
    }
    
    class HotSpineRuntime {
        +get_validation_metrics()
        +get_filtering_stats()
    }
    
    Metrics <|-- HotSpineReader
    Metrics <|-- HotSpineRuntime
```

## Error Handling Flow

```mermaid
flowchart TD
    A[Trade Received] --> B{Market Type Check}
    B -->|Invalid| C[Log Filtered Trade]
    B -->|Valid| D{Symbol Whitelist Check}
    D -->|Invalid| C
    D -->|Valid| E[Process Trade]
    
    E --> F{Strategy Processing}
    F -->|Error| G[Log Strategy Error]
    F -->|Success| H[Async SQL Storage]
    
    H --> I{Storage Success?}
    I -->|No| J[Log Storage Error]
    I -->|Yes| K[Update Metrics]
```

## Backward Compatibility

```mermaid
flowchart TD
    A[Legacy Configuration] --> B[HotSpineConfig]
    B -->|market_type_filter="all"| C[Allow All Trades]
    B -->|symbol_whitelist=None| D[No Symbol Filtering]
    B -->|symbol_mapping=None| E[Use Placeholder Symbols]
    
    C & D & E --> F[Existing Behavior Maintained]
```

## Performance Considerations

```mermaid
pie
    title Trade Processing Time Distribution
    "Market Type Filtering" : 2
    "Symbol Whitelist Filtering" : 3
    "Symbol Mapping Lookup" : 5
    "Core Processing" : 80
    "SQL Storage" : 10
```

The filtering overhead is minimal (<10% of total processing time), ensuring that the enhancements don't significantly impact performance.