# BTQ Render Engine Error Handling System

This directory contains the comprehensive error handling system for the BTQ Render Engine.

## Overview

The error handling system provides:

1. **Comprehensive Error Codes**: A complete set of error codes covering all major error scenarios in the engine
2. **Rich Error Information**: Detailed error information including code, message, details, file, line, and function
3. **Modern C++ Error Handling**: Using `std::expected` for robust error propagation
4. **User-Friendly Error Messages**: Human-readable error messages with troubleshooting steps
5. **Flexible Error Reporting**: Multiple error reporting mechanisms (console, file, custom reporters)
6. **Graceful Degradation**: Systems continue to operate even when errors occur

## Key Components

### 1. Result Type (`btq::Result<T>`)

A type alias for `std::expected<T, ErrorInfo>` that represents either a successful value of type `T` or an error.

```cpp
btq::Result<int> divide(int a, int b) {
    if (b == 0) {
        return btq::make_error<int>(btq::ErrorCode::kDivisionByZero, "Cannot divide by zero");
    }
    return a / b;
}
```

### 2. Error Codes (`btq::ErrorCode`)

Comprehensive enum of error codes covering:
- Rendering errors (Vulkan initialization, shader compilation, etc.)
- Resource management errors
- Network/data errors
- File system errors
- Configuration errors
- Threading errors
- UI/interaction errors
- Trading-specific errors
- Analytics/calculation errors

### 3. Error Information (`btq::ErrorInfo`)

Structure containing detailed error information:
- Error code
- Message
- Details
- File and line where error occurred
- Function name

### 4. Error Macros

- `BTQ_MAKE_ERROR(code, msg)` - Create an error
- `BTQ_TRY(expr)` - Propagate errors from expressions
- `BTQ_CATCH_TO_ERROR(block, error_code, msg)` - Convert exceptions to errors

### 5. Error Reporting System

- Console reporter for immediate feedback
- File reporter for persistent logging
- Custom reporter interface for specialized needs
- Global error reporter singleton

### 6. User-Friendly Error Messages

- Human-readable error messages
- Troubleshooting steps for common issues
- Context-aware error descriptions

## Usage Examples

### Creating Errors

```cpp
auto error = BTQ_MAKE_ERROR(btq::ErrorCode::kInvalidArgument, "Invalid parameter");
```

### Returning Errors from Functions

```cpp
btq::Result<std::string> load_file(const std::string& path) {
    if (path.empty()) {
        return btq::make_error<std::string>(
            btq::ErrorCode::kFileNotFound,
            "File path is empty",
            "Path parameter was empty"
        );
    }
    // Load file...
    return "file contents";
}
```

### Propagating Errors

```cpp
btq::Result<int> process_data() {
    auto file_result = load_file("data.txt");
    std::string data = BTQ_TRY(file_result);  // Returns early if error
    
    // Process data...
    return 42;
}
```

### Error Reporting

```cpp
// Initialize error reporting
btq::GlobalErrorReporter::initialize();

// Add file logging
btq::GlobalErrorReporter::add_reporter(
    std::make_unique<btq::FileErrorReporter>("errors.log")
);

// Report errors
btq::report_error(error_info, btq::ErrorSeverity::kError);
btq::report_message("Operation completed", btq::ErrorSeverity::kInfo);
```

## Best Practices

1. **Always Handle Errors**: Use `BTQ_TRY` for error propagation or explicitly check `has_value()`
2. **Provide Context**: Include meaningful error messages and details
3. **Use Appropriate Error Codes**: Select the most specific error code available
4. **Log Important Errors**: Use the error reporting system for significant issues
5. **Provide User-Friendly Messages**: Use the user-friendly error message system for end-user facing errors

## Thread Safety

The error handling system is designed to be thread-safe. The global error reporter can be safely accessed from multiple threads.

## Integration

The error handling system integrates seamlessly with existing code and follows the same patterns already established in the BTQ Render Engine (such as the use of `std::expected` seen in the vulkan_dashboard_advanced.hpp).