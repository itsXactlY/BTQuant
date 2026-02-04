# Loading States Implementation

This implementation provides comprehensive loading state management for the BTQ Render Engine, including progress indicators and prevention of UI freezing during operations.

## Features

- **Loading State Management**: Track and manage loading operations with start, update, and finish methods
- **Progress Tracking**: Monitor progress of long-running operations with percentage updates
- **UI Rendering**: Display progress bars, spinners, and loading overlays in the UI
- **Thread Safety**: Safe to use across multiple threads with mutex protection
- **Async Operations**: Execute operations asynchronously while showing loading indicators
- **Integration**: Works seamlessly with the existing TaskScheduler for async operations

## Components

### LoadingStateManager Class

The main class that manages loading states:

```cpp
class LoadingStateManager {
public:
    // Start a loading operation
    void startLoading(const std::string& operation_name, float initial_progress = 0.0f);
    
    // Update progress
    void updateProgress(float new_progress, const std::string& status_message = "");
    
    // Finish loading
    void finishLoading();
    
    // Check if loading is in progress
    bool isLoading() const;
    
    // Get current progress
    float getProgress() const;
    
    // Execute with loading indicator
    void executeWithLoading(const std::string& operation_name, std::function<void()> task_func);
    
    // Render loading UI elements
    void renderLoadingOverlay();
    void renderProgressBar(float fraction, const ImVec2& size_arg, const char* overlay);
    void renderLoadingSpinner(const char* label, float radius, int segments);
};
```

## Usage Examples

### Basic Usage
```cpp
btq::ui::LoadingStateManager loading_manager;

// Start loading
loading_manager.startLoading("Processing data...", 0.0f);

// Update progress
loading_manager.updateProgress(0.5f, "Halfway through...");

// Finish loading
loading_manager.finishLoading();
```

### Execute with Loading Indicator
```cpp
// Execute a long-running task with automatic loading state management
loading_manager.executeWithLoading("Calculating indicators...", []() {
    // Long-running operation here
    std::this_thread::sleep_for(std::chrono::seconds(2));
});
```

### Async Execution
```cpp
// Execute asynchronously with loading indicator
auto future = loading_manager.executeAsyncWithLoading("Background processing...", []() {
    // Background operation here
});
```

### UI Integration
```cpp
// In your UI rendering loop
if (loading_manager.isLoading()) {
    loading_manager.renderLoadingOverlay();  // Shows modal loading window
}
```

## Integration with Task Scheduler

The loading states implementation integrates with the existing TaskScheduler to provide progress tracking for computational tasks:

```cpp
btq::TaskScheduler scheduler;
btq::ui::LoadingStateManager loading_manager;

// Execute calculation with loading indicator
loading_manager.startLoading("Calculating SMA...", 0.0f);
auto future = scheduler.calculate_sma_async(prices, 20);

// Update progress periodically
while (loading_manager.isLoading() && /* calculation not complete */) {
    loading_manager.updateProgress(/* calculated progress */);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

auto result = future.get();
loading_manager.finishLoading();
```

## Thread Safety

The LoadingStateManager is thread-safe and can be accessed from multiple threads simultaneously. Internal mutexes protect shared state, and condition variables enable waiting for operations to complete.

## UI Elements

### Loading Overlay
A modal window that appears during loading operations, showing:
- Operation name/message
- Progress percentage
- Progress bar
- Elapsed time

### Progress Bar
A standard progress bar that can be integrated into existing UI layouts.

### Spinner
An animated spinner for indeterminate loading states.

## Benefits

1. **Prevents UI Freezing**: Long-running operations execute in background threads
2. **Visual Feedback**: Clear progress indicators keep users informed
3. **Responsive UI**: Main thread remains responsive during operations
4. **Consistent Experience**: Uniform loading state management across the application
5. **Easy Integration**: Simple API for adding loading states to existing operations