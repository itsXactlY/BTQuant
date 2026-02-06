# BTQ Render Engine - Developer Guide

## Table of Contents
1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Architecture Overview](#architecture-overview)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Development Best Practices](#development-best-practices)
6. [Testing](#testing)
7. [Performance Considerations](#performance-considerations)

## Overview

The BTQ Render Engine is a sophisticated trading terminal designed to provide professional-grade market analysis tools. This platform offers a comprehensive suite of analytical panels including charts, footprint analysis, order book visualization, volume profiles, and much more.

The application is built with performance in mind, utilizing Vulkan graphics and advanced rendering techniques to deliver smooth, real-time market data visualization even with high-frequency data feeds.

### Key Technologies
- **Language**: C++23/26
- **Graphics**: Vulkan API with ImGui for UI
- **Build System**: CMake
- **Threading**: TBB (Threading Building Blocks) and custom thread schedulers
- **UI Framework**: ImGui with docking branch
- **Plotting**: ImPlot for charting capabilities

## Setup Instructions

### Prerequisites

Before building the BTQ Render Engine, ensure you have the following packages installed:

```bash
# On Ubuntu/Debian systems
sudo apt update
sudo apt install build-essential cmake vulkan-sdk libglfw3-dev libvulkan-dev libtbb-dev git

# Install modern GCC or Clang compiler that supports C++26
sudo apt install gcc-13 g++-13
```

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/your-repo/BTQ_Render_Engine.git
cd BTQ_Render_Engine
```

2. Navigate to the main project directory:
```bash
cd dependencies/BTQ_Render_Engine
```

3. Create build directory:
```bash
mkdir build && cd build
```

4. Configure the build (ensure you're using a compiler that supports C++26):
```bash
cmake .. -DCMAKE_CXX_COMPILER=g++-13
```

5. Compile the application:
```bash
make -j$(nproc)
```

6. Run the application:
```bash
./realtime_dashboard
```

### Alternative Executables

The build system creates multiple executables:
- `realtime_dashboard` - Main trading dashboard application
- `dashboard_advanced` - Advanced version with additional features
- Various test executables for specific components

### Development Environment Setup

For optimal development experience:

1. **IDE Setup**:
   - VSCode with C/C++ extension
   - CLion or Visual Studio Code with C++ extensions
   - Ensure your IDE supports C++26

2. **Code Formatting**:
   - The project uses `.clang-format` for consistent code style
   - Format code before committing: `clang-format -i filename.cpp`

3. **Shader Compilation**:
   - Shaders are automatically compiled during the build process
   - GLSL shaders are located in the `shaders/` directory
   - SPIR-V binaries are generated in `shaders/spirv/`

## Architecture Overview

### High-Level Architecture

The BTQ Render Engine follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UI Layer      │    │  Business Logic  │    │   Data Layer    │
│                 │    │                  │    │                 │
│ • ImGui         │◄──►│ • Panels         │◄──►│ • HotSpine      │
│ • ImPlot        │    │ • Indicators     │    │ • Market Data   │
│ • Components    │    │ • Analytics      │    │ • Persistence   │
└─────────────────┘    │ • Trading        │    └─────────────────┘
                       └──────────────────┘
                              ▲
                              │
                       ┌──────────────────┐
                       │  Rendering Layer │
                       │                  │
                       │ • Vulkan Core    │
                       │ • Shaders        │
                       │ • Frame Pacing   │
                       └──────────────────┘
```

### Core Components

#### 1. Data Layer
- **UnifiedDataPipeline**: Processes and normalizes incoming data streams
- **CacheManager**: Manages data caching and retrieval
- **SymbolRegistry**: Maintains available trading symbols and metadata

#### 2. Business Logic Layer
- **Panel Manager**: Orchestrates different trading panels
- **Indicator Engine**: Calculates technical indicators
- **Analytics Module**: Performs market microstructure analysis
- **Trading Interface**: Handles order management and execution

#### 3. Rendering Layer
- **VulkanCore**: Low-level Vulkan graphics management
- **ImGui Integration**: UI rendering with Vulkan backend
- **Frame Pacer**: Ensures smooth rendering performance
- **LOD System**: Level-of-detail for performance optimization

#### 4. UI Components
- **Chart Panel**: Price chart visualization
- **Footprint Panel**: Volume analysis at price levels
- **Order Book (DOM)**: Market depth visualization
- **Time & Sales**: Chronological trade display
- **Volume Profile**: Volume distribution analysis
- **Watchlist**: Multiple symbol monitoring

### Threading Model

The application uses a sophisticated multi-threading approach:

- **Main Thread**: UI rendering and user interaction
- **Task Scheduler**: Distributes computational work across threads
- **Lock-Free Queues**: Efficient inter-thread communication
- **TBB Integration**: Parallel algorithms for data processing
- **Memory Pools**: Reduce allocation overhead in performance-critical paths

### Key Directories

```
dependencies/BTQ_Render_Engine/
├── include/              # Header files organized by module
│   ├── components/       # Panel interfaces and base classes
│   ├── data/            # Data structures and interfaces
│   ├── rendering/       # Vulkan and rendering abstractions
│   ├── performance/     # Performance monitoring utilities
│   └── ...
├── src/                 # Source files organized by module
│   ├── components/      # Panel implementations
│   ├── data/           # Data processing implementations
│   ├── rendering/      # Rendering implementations
│   ├── threading/      # Threading utilities
│   └── ...
├── shaders/             # GLSL shader files
├── tests/               # Unit and integration tests
├── docs/                # Documentation
├── CMakeLists.txt       # Build configuration
└── default_layout.json  # Default dashboard layout
```

## Contribution Guidelines

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Ensure all tests pass
5. Submit a pull request

### Code Standards

#### C++ Coding Standards
- Use C++23/26 features where appropriate
- Follow RAII principles for resource management
- Use smart pointers instead of raw pointers when possible
- Prefer const correctness
- Use range-based for loops when iterating containers
- Follow the SOLID principles

#### Naming Conventions
- Class names: `PascalCase` (e.g., `ChartPanel`, `DataManager`)
- Function names: `camelCase` (e.g., `updateMarketData`, `renderFrame`)
- Variable names: `camelCase` (e.g., `symbolName`, `volumeProfile`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `MAX_ORDER_BOOK_LEVELS`)
- Private members: Leading underscore (e.g., `_dataBuffer`, `_renderer`)

#### File Organization
- One class per header/source file pair when possible
- Header guards: `#pragma once`
- Include order: related header, C library, C++ library, other libraries, project headers
- Use forward declarations when possible to reduce compilation dependencies

### Testing Requirements

All contributions must include:
- Unit tests for new functionality
- Integration tests where appropriate
- Performance benchmarks for performance-sensitive code
- Manual testing verification for UI components

### Pull Request Process

1. Ensure your code follows the project's style and standards
2. Add tests for new functionality
3. Update documentation as needed
4. Verify all existing tests still pass
5. Describe your changes in the pull request
6. Reference any related issues

### Development Workflow

#### Feature Development
1. Create a new branch from `main`
2. Implement the feature with tests
3. Ensure code quality and performance
4. Submit pull request for review
5. Address feedback and iterate

#### Bug Fixes
1. Create a test that reproduces the bug
2. Fix the issue
3. Verify the test now passes
4. Submit pull request with bug description

### Code Review Checklist

When submitting or reviewing code, consider:
- Does the code follow the established patterns?
- Are there appropriate unit tests?
- Is the performance acceptable?
- Is the code maintainable and readable?
- Are edge cases handled properly?
- Does the change affect existing functionality?

## Development Best Practices

### Performance Optimization
- Profile before optimizing
- Use memory pools for frequent allocations
- Minimize dynamic allocations in rendering loops
- Use SIMD instructions where beneficial
- Implement LOD (Level of Detail) for large datasets

### Memory Management
- Use RAII for automatic resource management
- Prefer stack allocation over heap when possible
- Use smart pointers appropriately
- Implement custom allocators for performance-critical paths

### Error Handling
- Use exceptions for exceptional conditions
- Return error codes for expected failure cases
- Log errors with sufficient context
- Fail gracefully when possible

### Concurrency
- Use thread-safe data structures
- Minimize shared mutable state
- Use atomic operations for simple synchronization
- Prefer lock-free algorithms when appropriate

## Testing

### Running Tests

Execute all tests:
```bash
cd build
ctest
```

Run specific test executable:
```bash
./tests/test_specific_feature
```

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test interactions between modules
- **Performance Tests**: Benchmark critical paths
- **Regression Tests**: Ensure bugs don't reappear

### Writing Tests

Follow the AAA pattern (Arrange, Act, Assert):
```cpp
TEST_CASE("Market data processing") {
    // Arrange
    MarketDataProcessor processor;
    auto testData = createMockData();
    
    // Act
    auto result = processor.process(testData);
    
    // Assert
    REQUIRE(result.isValid());
    REQUIRE(result.size() == expectedSize);
}
```

## Performance Considerations

### Critical Paths

The following areas require special attention to performance:
- Market data ingestion and processing
- Real-time rendering loop
- UI responsiveness
- Memory allocation/deallocation

### Profiling Tools

The application includes built-in profiling capabilities:
- CPU Profiler for identifying bottlenecks
- Memory Tracker for allocation analysis
- Frame Time Graph for rendering performance
- Performance Monitor for system metrics

### Optimization Techniques Used

- Lock-free data structures for inter-thread communication
- Memory pools to reduce allocation overhead
- Level-of-Detail (LOD) rendering for large datasets
- Batch rendering for similar objects
- Culling of off-screen elements
- Incremental data updates

---

## Getting Help

For additional support:
- Check the existing documentation in the `docs/` directory
- Review the PRDs and implementation summaries in the root directory
- Examine existing code examples and tests
- Reach out to the development team for complex architectural questions