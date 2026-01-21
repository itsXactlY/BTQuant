# Market Microstructure Renderer Merge Plan

## Overview
This plan outlines the steps to merge the Market Microstructure Renderer features from `clean_render_wip_1` into `clean_render_wip_2`. The goal is to implement a zero-latency trading visualization system with LOB Heatmap, Footprint Charts, and TPO Profile using Vulkan 1.3.

## Current State Analysis (clean_render_wip_2)
- **VulkanCore**: Well-implemented with semaphore/fence synchronization (2 frames in flight)
- **VulkanDashboard**: Main dashboard with ImGui integration
- **HotSpineDataBridge**: Zero-copy shared memory reader for market data
- **MarketDataProcessor**: Data storage and processing
- **QuantWorkspaceComponent**: Modular UI framework with panels
- **No shaders directory**: Missing compute and graphics shaders for microstructure rendering

## Target Architecture
```mermaid
graph TD
    A[Market Microstructure Renderer] --> B[LOB Heatmap (Compute Shader)]
    A --> C[Footprint Chart (Instanced Rendering)]
    A --> D[TPO Profile (Compute + Atomic)]
    B --> E[Vulkan Sync Context]
    C --> E
    D --> E
    E --> F[Hotspine SSBO Manager]
    E --> G[Hotspine Data Structures]
    F --> H[Vulkan Core Resources]
    G --> H
```

## Implementation Steps

### Phase 1: Infrastructure Setup
1. **Create shaders directory structure**
   - `dependencies/BTQ_Render_Engine/shaders/` - For GLSL source files
   - `dependencies/BTQ_Render_Engine/shaders/spirv/` - For compiled SPIR-V files

2. **Implement HotspineData.h** (GPU-aligned data structures)
   - Define `GpuAlignable` concept for standard layout types
   - Implement `OrderBookLevel` with 16-byte alignment
   - Create `HotspineOrderBookSnapshot` with variable-length array
   - Add `CandleCluster` structure for footprint charts

3. **Implement VulkanSynchronization.h/cpp**
   - Create `TimelineSemaphore` class with timeline value tracking
   - Implement `VulkanSyncContext` for render pass synchronization
   - Add `SyncDebugUtils` for performance measurement
   - Support 3 frames in flight for maximum GPU utilization

### Phase 2: Core Renderer Implementation
4. **Create MarketMicrostructureRenderer.h** (Main interface)
   - Define `RendererConfig` structure with component configurations
   - Declare `MarketMicrostructureRenderer` class with render/update methods
   - Add `RendererStats` for performance tracking
   - Define integration points for data feed and render loop

5. **Implement MarketMicrostructureRenderer.cpp** (Core logic)
   - Initialize compute and graphics pipelines
   - Manage SSBOs (Shader Storage Buffer Objects) for Hotspine data
   - Handle texture updates for heatmap rendering
   - Implement render loop integration

### Phase 3: Shader Development
6. **Add LOB Heatmap compute shader** (`lob_heatmap.comp`)
   - GPU-side histogram calculation
   - Cyclic heatmap texture writing
   - SSBO to Image2D transfer
   - Blue-to-red color gradient for liquidity visualization

7. **Add Footprint Chart shaders**
   - `footprint_vert.glsl` - Instanced quad rendering
   - `footprint_frag.glsl` - Bid/ask volume coloring and text rendering
   - SDF (Signed Distance Field) font rendering support

8. **Add TPO Profile compute shader** (`tpo_profile.comp`)
   - Atomic operations for histogram bucketing
   - Price level binning with `atomicAdd`
   - Rolling window histogram reset functionality

9. **Compile shaders to SPIR-V**
   - Add CMake custom commands for shader compilation
   - Support GLSL to SPIR-V compilation during build
   - Verify shader compilation in CI/CD

### Phase 4: Pipeline Management
10. **Integrate with VulkanCore**
    - Extend VulkanCore with pipeline management methods
    - Add compute pipeline creation and management
    - Extend descriptor pool with compute shader descriptors
    - Implement pipeline barriers for shader transitions

11. **Implement SSBO Manager**
    - Create `HotspineSSBOManager` for buffer management
    - Handle double-buffered SSBO updates
    - Implement lock-free atomic operations for buffer acquisition
    - Support 1MB slot sizes for high-throughput data

### Phase 5: Data Integration
12. **Connect with HotspineDataBridge**
    - Extend HotSpineDataBridge to provide SSBO-compatible data
    - Implement data conversion from HotTrade/HotOrderbookSnapshot to GPU formats
    - Add callback mechanisms for data updates
    - Optimize data transfer for low latency

13. **Update MarketDataProcessor**
    - Add support for storing and processing footprint chart clusters
    - Extend with TPO profile histogram data
    - Implement data aggregation for heatmap generation

### Phase 6: UI Integration
14. **Add MarketMicrostructurePanel**
    - Create `MarketMicrostructureComponent` inherited from UIComponent
    - Implement ImGui controls for renderer configuration
    - Add panel to QuantWorkspaceComponent
    - Support resizable panel and docking

15. **Implement Renderer Configuration UI**
    - LOB Heatmap settings (width, height, max liquidity)
    - Footprint Chart settings (cell size, label visibility)
    - TPO Profile settings (bucket count, time window)
    - Real-time statistics display

### Phase 7: Build Configuration
16. **Update CMakeLists.txt**
    - Add shader compilation to build process
    - Include new source files in dashboard_advanced target
    - Add compute shader dependencies
    - Ensure GLSL compiler (glslc) is available

17. **Add shader compilation script**
    - Create `build_shaders.sh` for manual shader compilation
    - Support incremental compilation
    - Verify shader compatibility with Vulkan 1.3

### Phase 8: Testing and Optimization
18. **Test with main_realtime_dashboard.cpp**
    - Add Market Microstructure Renderer to main dashboard
    - Test integration with Hotspine data source
    - Verify rendering performance and stability

19. **Add synchronization debug utilities**
    - Implement `SyncDebugUtils::measureExecutionTime()`
    - Add `SyncDebugUtils::logSyncState()`
    - Support rendering statistics tracking

20. **Optimize compute shader performance**
    - Profile shader execution times
    - Optimize work group sizes
    - Implement asynchronous compute queue execution
    - Test with real market data

### Phase 9: Documentation
21. **Update MarketMicrostructureRenderer.arch.md**
    - Add implementation details
    - Include pipeline diagrams
    - Document shader interfaces
    - Add performance characteristics

22. **Add usage examples**
    - Example integration with Hotspine data source
    - Render loop integration example
    - Configuration and customization examples

## Critical Components from clean_render_wip_1
- **Shaders**: LOB heatmap, footprint, and TPO profile shaders
- **GPU data structures**: HotspineData.h with alignment requirements
- **Synchronization**: Timeline semaphore implementation
- **Renderer core**: MarketMicrostructureRenderer class
- **Pipeline management**: Compute and graphics pipeline setup

## Risk Mitigation
1. **Shader compilation**: Ensure GLSL compiler availability in CI
2. **Vulkan version compatibility**: Verify Vulkan 1.3 support
3. **Performance**: Test on different GPU architectures
4. **Data alignment**: Ensure GPU-CPU memory layout compatibility
5. **Synchronization**: Test with various frame rates and data rates

## Success Criteria
- LOB Heatmap rendering at 60+ FPS
- Footprint Chart with 4096+ clusters
- TPO Profile with 256+ histogram buckets
- Zero-latency data processing (<0.5ms)
- Stable operation with real Hotspine data

## Completion Timeline
This is a multi-phase implementation. Each phase should be tested and validated before proceeding to the next.
