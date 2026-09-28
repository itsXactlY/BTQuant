# btquant_vulkan Architektur

## Überblick
Das btquant_vulkan Projekt ist eine High-Frequency Trading-Anwendung mit Vulkan-GPU-Rendering Engine.

## Systemarchitektur

```mermaid
graph TD
    A[Main Application] --> B[Vulkan Context]
    A --> C[Data Spine]
    A --> D[UI Context]
    A --> E[Window Manager]
    
    B --> F[Vulkan Device]
    B --> G[Swapchain]
    B --> H[Render Pass]
    B --> I[Command Buffers]
    
    C --> J[Memory Mapped File]
    C --> K[Ring Buffers]
    C --> L[Market Data Aggregator]
    
    D --> M[ImGui Context]
    D --> N[ImPlot Context]
    
    E --> O[Window Management]
    E --> P[Docking System]
    
    J --> K
    K --> L
    L --> Q[GPU Buffers]
    
    Q --> R[Compute Shaders]
    Q --> S[Graphics Shaders]
    
    R --> T[Heatmaps]
    R --> U[VPVR Calculations]
    
    S --> V[Charts]
    S --> W[DOM Visualization]
    S --> X[Order Book Display]
    
    O --> Y[Order Book Widget]
    O --> Z[DOM Widget]
    O --> AA[Trades Widget]
    O --> AB[TPO Widget]
    
    Y --> AC[Price Levels]
    Y --> AD[Order Size]
    
    Z --> AE[Depth Visualization]
    Z --> AF[Market Depth]
    
    AA --> AG[Trade History]
    AA --> AH[Volume Analysis]
    
    AB --> AI[Time Price Opportunity]
    AB --> AJ[Market Momentum]
    
    style A fill:#4CAF50,stroke:#333
    style B fill:#2196F3,stroke:#333
    style C fill:#FF9800,stroke:#333
    style D fill:#9C27B0,stroke:#333
    style E fill:#F44336,stroke:#333
    style F fill:#607D8B,stroke:#333
    style G fill:#607D8B,stroke:#333
    style H fill:#607D8B,stroke:#333
    style I fill:#607D8B,stroke:#333
    style J fill:#8BC34A,stroke:#333
    style K fill:#8BC34A,stroke:#333
    style L fill:#8BC34A,stroke:#333
    style Q fill:#009688,stroke:#333
    style R fill:#009688,stroke:#333
    style S fill:#009688,stroke:#333
    style T fill:#009688,stroke:#333
    style U fill:#009688,stroke:#333
    style V fill:#009688,stroke:#333
    style W fill:#009688,stroke:#333
    style X fill:#009688,stroke:#333
    style Y fill:#FF5722,stroke:#333
    style Z fill:#FF5722,stroke:#333
    style AA fill:#FF5722,stroke:#333
    style AB fill:#FF5722,stroke:#333
```

## Datenfluss

```mermaid
sequenceDiagram
    participant DataFile as /dev/shm/btquant_hotspine
    participant DataSpine as DataSpine
    participant RingBuffer as RingBuffer
    participant Aggregator as MarketDataAggregator
    participant GPUBuffer as GPU Buffers
    participant ComputeShader as Compute Shaders
    participant GraphicsShader as Graphics Shaders
    participant UI as UI Context
    
    DataFile->>DataSpine: Read binary data
    DataSpine->>RingBuffer: Store in lock-free queue
    RingBuffer->>Aggregator: Process market data
    Aggregator->>GPUBuffer: Transfer to GPU memory
    GPUBuffer->>ComputeShader: Perform calculations
    ComputeShader->>GPUBuffer: Store results
    GPUBuffer->>GraphicsShader: Prepare for rendering
    GraphicsShader->>UI: Render visualizations
```

## Komponentenbeschreibung

### 1. Datenverarbeitungsschicht
- **DataSpine**: Verwaltet den Zugriff auf die binären Daten in `/dev/shm/btquant_hotspine`
- **RingBuffer**: Lockfreie Warteschlangen für Datenfluss
- **MarketDataAggregator**: Verarbeitet und aggregiert Marktinformationen

### 2. Rendering-Schicht
- **VulkanContext**: Vulkan-Initialisierung und Geräteverwaltung
- **RenderPipeline**: Render-Pipeline für Vulkan
- **GPU Buffers**: Speicher für GPU-Visualisierungen

### 3. UI-Schicht
- **UI Context**: ImGui Vulkan Integration
- **Window Manager**: Fenster- und Docking-System
- **Widgets**: Spezifische UI-Komponenten

### 4. Compute Shaders
- **Heatmaps**: Temperaturvisualisierungen
- **VPVR Calculations**: Volume Profile Value Range Berechnungen

## Datenformat
Das Binärformat von `/dev/shm/btquant_hotspine`:
- Header: "UQTB" Magic (4 Bytes) + Version + Symbol Count
- Daten beginnen bei 0x1000, pro Symbol:
  - bid_price, ask_price (double)
  - bid_size, ask_size (double)
  - timestamp (uint64, Mikrosekunden)
  - seq (uint32)