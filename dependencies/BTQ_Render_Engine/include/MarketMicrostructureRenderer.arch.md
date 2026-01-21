# Market Microstructure Renderer Architecture

## Overview

The Market Microstructure Renderer is a zero-latency trading visualization system built on Vulkan 1.3, designed to replicate the visual capabilities of Optimus Flow for high-frequency trading analysis.

## System Architecture

### Core Components

```
┌───────────────────────────────────────────────────────────────────────┐
│ Market Microstructure Renderer                                       │
├───────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐
│  │ LOB Heatmap  │  │ Footprint    │  │ TPO (Time Price Opportunity) │
│  │ (Compute     │  │ Chart        │  │ Profile                      │
│  │ Shader)      │  │ (Instanced   │  │ (Compute Shader + Atomic)    │
│  │              │  │ Rendering)   │  │                              │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘
│           │                │                         │
│           └────────────────┴─────────────────────────┘
│                              │
│                      ┌─────────────┐
│                      │ Vulkan Sync │
│                      │  Context    │
│                      └─────────────┘
│                              │
│           ┌──────────────────┴──────────────────┐
│           │                                     │
│  ┌──────────────┐                      ┌──────────────┐
│  │ Hotspine SSBO│                      │ Hotspine Data │
│  │  Manager     │                      │  Structures  │
│  └──────────────┘                      └──────────────┘
│           │                                     │
│           └────────────────┬─────────────────────┘
│                            │
│                    ┌──────────────┐
│                    │ Vulkan Core  │
│                    │  Resources   │
│                    └──────────────┘
└───────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. LOB Heatmap (Limit Order Book Surface)

**File:** [`dependencies/BTQ_Render_Engine/shaders/lob_heatmap.comp`](dependencies/BTQ_Render_Engine/shaders/lob_heatmap.comp)

**Technique:** Compute Shader texture writing

- **Input:** Order book snapshot from Hotspine ring buffer (SSBO)
- **Output:** Cyclic heatmap texture where:
  - X-axis = Time (rolling window)
  - Y-axis = Price levels
  - Color = Liquidity density (cool to warm gradient)
- **Texture Dimensions:** 1024x512 by default
- **Color Scheme:** Blue → Red (low to high liquidity)

**Key Features:**
- Dynamic rolling window using cyclic buffer
- GPU-side histogram calculation
- No CPU texture regeneration
- Direct SSBO to Image2D transfer

### 2. Volumetric Footprint Charts

**Files:** 
- Vertex: [`dependencies/BTQ_Render_Engine/shaders/footprint_vert.glsl`](dependencies/BTQ_Render_Engine/shaders/footprint_vert.glsl)
- Fragment: [`dependencies/BTQ_Render_Engine/shaders/footprint_frag.glsl`](dependencies/BTQ_Render_Engine/shaders/footprint_frag.glsl)

**Technique:** Instanced rendering with quad instances

**CandleCluster Structure:**
```cpp
struct CandleCluster {
    float centerX;          // X coordinate (time)
    float centerY;          // Y coordinate (price)
    float width;            // Time duration
    float height;           // Price range
    uint32_t bidVolume;     // Total bid volume
    uint32_t askVolume;     // Total ask volume
    uint32_t tradeCount;    // Number of trades
    float vwap;             // Volume-weighted average price
    bool hasTrades;         // Trade activity indicator
};
```

**Text Rendering:**
- Signed Distance Field (SDF) font atlas
- Geometry shader batching
- Distance field rendering for crisp text at any size
- Configurable SDF spread and font atlas dimensions

### 3. TPO Profile (Time Price Opportunity)

**File:** [`dependencies/BTQ_Render_Engine/shaders/tpo_profile.comp`](dependencies/BTQ_Render_Engine/shaders/tpo_profile.comp)

**Technique:** Compute shader with atomic operations

**Histogram Bucketing:**
- Price level → Histogram bucket
- Uses `atomicAdd` for GPU-side increment
- Real-time updates without CPU synchronization
- Configurable bucket count and time window
- Supports rolling window resets

## Synchronization Strategy

**File:** [`dependencies/BTQ_Render_Engine/include/components/VulkanSynchronization.h`](dependencies/BTQ_Render_Engine/include/components/VulkanSynchronization.h)

### Zero-Latency Design Principles

1. **Double-Buffered Command Buffers:** 
   - 3 frames in flight for maximum GPU utilization
   - Timeline semaphores for fine-grained synchronization
   - Fences for CPU-GPU signaling

2. **Ring Buffer Management:**
   - Dual slot ring buffer for data updates
   - Lock-free atomic operations for buffer acquisition/release
   - 1MB slots for high-throughput data

3. **Memory Barriers:**
   - Buffer-to-Compute shader transitions
   - Compute-to-Image transitions
   - Image-to-Fragment shader transitions
   - Pipeline barriers for texture reading/writing

### Timeline Semaphore Usage

```cpp
class TimelineSemaphore {
    TimelineValue getCurrentValue();
    bool waitForValue(TimelineValue value);
    bool signalValue(TimelineValue value);
};
```

### Render Pass Synchronization

```cpp
struct RenderPassSyncState {
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;
    TimelineValue lastCompletedValue;
};
```

## Hotspine Data Structures

**File:** [`dependencies/BTQ_Render_Engine/include/trading/HotspineData.h`](dependencies/BTQ_Render_Engine/include/trading/HotspineData.h)

### GPU-CPU Memory Alignment

```cpp
// Standard layout, trivially copyable types
template<typename T>
concept GpuAlignable = std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>;

// Explicit alignment
struct alignas(16) OrderBookLevel {
    float price;
    uint32_t askQuantity;
    uint32_t bidQuantity;
    uint32_t numOrders;
};
```

### Variable-Length Arrays

```cpp
struct alignas(16) HotspineOrderBookSnapshot {
    uint32_t currentTimeIndex;
    uint32_t priceLevelsCount;
    float basePrice;
    float priceRange;
    OrderBookLevel levels[];  // Must be last member
};

// Calculate required buffer size
constexpr size_t calculateOrderBookBufferSize(uint32_t priceLevelCount) {
    return sizeof(HotspineOrderBookSnapshot) + priceLevelCount * sizeof(OrderBookLevel);
}
```

## Configuration

### Renderer Configuration

```cpp
struct RendererConfig {
    LOBHeatmapConfig lobHeatmap;
    FootprintChartConfig footprintChart;
    TPOProfileConfig tpoProfile;
};

struct LOBHeatmapConfig {
    uint32_t width = 1024;
    uint32_t height = 512;
    float maxLiquidity = 100000.0f;
    bool invertYAxis = true;
};

struct FootprintChartConfig {
    uint32_t maxClusters = 4096;
    float cellMinSize = 2.0f;
    float cellMaxSize = 20.0f;
    bool showLabels = true;
};

struct TPOProfileConfig {
    uint32_t bucketCount = 256;
    float priceResolution = 0.1f;
    uint32_t timeWindowMs = 30000;
    bool resetOnUpdate = true;
};
```

## Pipeline Management

### Compute Pipelines

```
LOB Heatmap Compute Pipeline
├─ Set 0, Binding 0: Order Book Snapshot (SSBO)
├─ Set 0, Binding 1: Heatmap Output (Image2D)
├─ Set 1, Binding 0: Heatmap Parameters (UBO)
└─ Work Group Size: 16x16

TPO Profile Compute Pipeline
├─ Set 0, Binding 0: Trade Ticks (SSBO)
├─ Set 0, Binding 1: Histogram Buckets (SSBO)
├─ Set 1, Binding 0: Histogram Parameters (UBO)
└─ Work Group Size: 64

Footprint Chart Graphics Pipeline
├─ Set 0, Binding 0: Candle Clusters (SSBO)
├─ Set 1, Binding 0: View Parameters (UBO)
├─ Set 2, Binding 0: Font Atlas (Texture2D)
├─ Set 2, Binding 1: Text Parameters (UBO)
└─ Topology: Triangles (Instanced Quads)
```

## Performance Characteristics

### Expected FPS: 60+ FPS at 1080p

### Memory Usage:
- Order Book SSBO: ~16KB per snapshot
- Trade Ticks SSBO: ~64KB per batch (4096 ticks)
- Heatmap Texture: 1024x512x32bpp = ~2MB
- TPO Histogram: 256 buckets = ~1KB
- Font Atlas: 512x512x8bpp = ~256KB

### GPU Memory Budget:
- Minimum: 128MB VRAM
- Recommended: 256MB VRAM

## Integration Points

### Data Feed Integration

```cpp
// Example integration with Hotspine data source
void onOrderBookUpdate(const HotspineOrderBookSnapshot& snapshot) {
    renderer->updateLOBData(snapshot);
}

void onTradeUpdate(const HotspineTradeTicks& trades) {
    renderer->updateTradeData(trades);
}

void onFootprintUpdate(std::span<const CandleCluster> clusters) {
    renderer->updateFootprintClusters(clusters);
}
```

### Render Loop Integration

```cpp
void renderLoop() {
    vk::VulkanSyncContext syncContext(device, cmdPool);
    vk::TimelineSemaphore timelineSemaphore(device);
    
    while (!shouldExit) {
        const uint32_t currentFrame = frameCounter % MAX_FRAMES_IN_FLIGHT;
        
        // Wait for previous frame to complete and prepare for next
        syncContext.prepareFrame(currentFrame, timelineSemaphore);
        
        // Acquire command buffer
        auto cmdBuffer = syncContext.acquireCommandBuffer();
        beginCommandBuffer(cmdBuffer);
        
        // Render market microstructure
        renderer->render(cmdBuffer, currentFrame, syncContext, timelineSemaphore);
        
        // Submit command buffer
        syncContext.submitCommandBuffer(queue, cmdBuffer, timelineSemaphore,
                                      currentFrame + 1, currentFrame);
        
        // Present to screen
        presentFrame(syncContext, currentFrame);
        
        frameCounter++;
    }
}
```

## Debugging and Profiling

### Synchronization Debug

```cpp
class SyncDebugUtils {
    static uint64_t measureExecutionTime(VkDevice device,
                                       VkCommandBuffer cmdBuffer,
                                       VkQueue queue,
                                       const char* name);
    static void logSyncState(const VulkanSyncContext& context);
};
```

### Renderer Statistics

```cpp
struct RendererStats {
    uint32_t framesRendered = 0;
    uint32_t lobUpdates = 0;
    uint32_t tradeUpdates = 0;
    uint32_t footprintCellsRendered = 0;
    double averageFrameTimeMs = 0.0;
    uint64_t lastUpdateTimeNs = 0;
};

RendererStats stats = renderer->getStats();
```

## Future Optimizations

### Potential Improvements

1. **Asynchronous Compute Shader Execution:**
   - Separate queue families for compute and graphics
   - Time-slice compute and rendering phases

2. **Sparse Texture Sampling:**
   - Level-of-detail (LOD) for heatmap texture
   - Mipmapped texture sampling

3. **Instanced Text Rendering:**
   - Geometry shader text batching
   - GPU-side text layout calculation

4. **Ray Tracing Enhancements:**
   - Vulkan Ray Tracing (VK_EXT_ray_tracing)
   - Real-time reflection and lighting effects

5. **Machine Learning Integration:**
   - ML-based liquidity prediction visualization
   - Anomaly detection overlays

## Build Configuration

### CMake Integration

```cmake
# Shader compilation
find_program(GLSLC glslc)
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert"
                  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag")

foreach(SHADER ${SHADERS})
    string(REPLACE ".glsl" ".spv" OUTPUT ${SHADER})
    string(REPLACE ".comp" ".spv" OUTPUT ${OUTPUT})
    
    add_custom_command(
        OUTPUT ${OUTPUT}
        COMMAND ${GLSLC} ${SHADER} -o ${OUTPUT}
        DEPENDS ${SHADER}
    )
endforeach()

# Target configuration
add_library(BTQ_Render_Engine STATIC
    src/components/MarketMicrostructureRenderer.cpp
    src/components/VulkanSynchronization.cpp
    include/components/MarketMicrostructureRenderer.h
    include/components/VulkanSynchronization.h
    include/trading/HotspineData.h
    ${COMPILED_SHADERS}
)

target_link_libraries(BTQ_Render_Engine Vulkan::Vulkan)
```

## Summary

The Market Microstructure Renderer provides a comprehensive, zero-latency trading visualization system that directly matches the capabilities of Optimus Flow using Vulkan 1.3's advanced features. The architecture is designed for high-throughput data processing, with GPU-side computations eliminating expensive CPU-GPU transfers. The system supports real-time visualization of market microstructure through:

1. **LOB Heatmap:** Historical liquidity evolution
2. **Volumetric Footprint:** Candle clusters with bid/ask volume
3. **TPO Profile:** Time spent at price levels histogram

All components are designed for maximum performance and are suitable for high-frequency trading applications.