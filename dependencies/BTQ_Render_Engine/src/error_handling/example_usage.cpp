#include "error_handling/result.hpp"
#include "error_handling/error_reporter.hpp"
#include "error_handling/user_friendly_errors.hpp"
#include <iostream>
#include <vector>
#include <string>

namespace btq {

// Example function that demonstrates error handling
Result<int> divide_numbers(int numerator, int denominator) {
    if (denominator == 0) {
        return make_error<int>(
            ErrorCode::kDivisionByZero,
            "Attempted to divide by zero",
            "Denominator value was 0",
            __FILE__,
            __LINE__,
            __FUNCTION__
        );
    }
    
    return numerator / denominator;
}

// Example function that demonstrates success case
Result<std::string> load_configuration(const std::string& config_path) {
    // Simulate loading configuration
    if (config_path.empty()) {
        return make_error<std::string>(
            ErrorCode::kConfigNotFound,
            "Configuration file path is empty",
            "No configuration file specified",
            __FILE__,
            __LINE__,
            __FUNCTION__
        );
    }
    
    // Simulate successful loading
    return "Configuration loaded successfully from: " + config_path;
}

// Example function that demonstrates error propagation
Result<double> calculate_complex_metric(const std::vector<double>& data) {
    if (data.empty()) {
        return make_error<double>(
            ErrorCode::kInvalidDataFormat,
            "Cannot calculate metric from empty data",
            "Input vector is empty",
            __FILE__,
            __LINE__,
            __FUNCTION__
        );
    }
    
    // Try to perform calculation that might fail
    double sum = 0.0;
    for (double val : data) {
        if (val != val) { // Check for NaN
            return make_error<double>(
                ErrorCode::kMathDomainError,
                "Invalid value (NaN) found in data",
                "Encountered NaN value during calculation",
                __FILE__,
                __LINE__,
                __FUNCTION__
            );
        }
        sum += val;
    }
    
    // Simulate potential overflow
    if (sum > 1e308) { // Approximate double max
        return make_error<double>(
            ErrorCode::kCalculationOverflow,
            "Calculation would result in overflow",
            "Sum exceeds maximum representable value",
            __FILE__,
            __LINE__,
            __FUNCTION__
        );
    }
    
    return sum / data.size(); // Return average
}

// Example demonstrating user-friendly error messages
void demonstrate_user_friendly_errors() {
    std::cout << "=== Demonstrating User-Friendly Error Messages ===" << std::endl;
    
    // Example 1: Division by zero
    auto result1 = divide_numbers(10, 0);
    if (!result1) {
        std::cout << "Error occurred: " << to_string(result1.error()) << std::endl;
        std::cout << "User-friendly message: " 
                  << UserFriendlyErrorMessage::generate_user_friendly_message(result1.error().code) 
                  << std::endl;
        std::cout << std::endl;
    }
    
    // Example 2: Invalid configuration
    auto result2 = load_configuration("");
    if (!result2) {
        std::cout << "Error occurred: " << to_string(result2.error()) << std::endl;
        std::cout << "User-friendly message: " 
                  << UserFriendlyErrorMessage::generate_user_friendly_message(result2.error().code) 
                  << std::endl;
        std::cout << std::endl;
    }
    
    // Example 3: Complex calculation with empty data
    std::vector<double> empty_data;
    auto result3 = calculate_complex_metric(empty_data);
    if (!result3) {
        std::cout << "Error occurred: " << to_string(result3.error()) << std::endl;
        std::cout << "User-friendly message: " 
                  << UserFriendlyErrorMessage::generate_user_friendly_message(result3.error().code) 
                  << std::endl;
        std::cout << std::endl;
    }
}

// Example demonstrating error reporting
void demonstrate_error_reporting() {
    std::cout << "=== Demonstrating Error Reporting ===" << std::endl;
    
    // Initialize global error reporter
    GlobalErrorReporter::initialize();
    
    // Add a file reporter for persistent logging
    try {
        GlobalErrorReporter::add_reporter(std::make_unique<FileErrorReporter>("error_demo.log"));
    } catch (const std::exception& e) {
        std::cout << "Could not add file reporter: " << e.what() << std::endl;
    }
    
    // Generate some errors to report
    auto result = divide_numbers(5, 0);
    if (!result) {
        report_error(result.error(), ErrorSeverity::kError);
    }
    
    auto config_result = load_configuration("");
    if (!config_result) {
        report_error(config_result.error(), ErrorSeverity::kWarning);
    }
    
    report_message("Demonstration of error reporting completed", ErrorSeverity::kInfo);
    
    std::cout << "Errors reported to console and log file." << std::endl;
}

// Example demonstrating error handling with BTQ_TRY macro
Result<double> complex_calculation_with_try_macro(const std::vector<double>& inputs) {
    // Use BTQ_TRY to propagate errors from other functions
    auto config_result = load_configuration("example_config.json");
    std::string config = BTQ_TRY(std::move(config_result)); // This will return early if error occurs
    
    // Perform some calculation
    if (inputs.empty()) {
        return make_error<double>(
            ErrorCode::kInvalidArgument,
            "Input vector cannot be empty",
            "Size: 0",
            __FILE__,
            __LINE__,
            __FUNCTION__
        );
    }
    
    // Use BTQ_TRY again for another operation that might fail
    auto metric_result = calculate_complex_metric(inputs);
    double metric = BTQ_TRY(std::move(metric_result)); // This will return early if error occurs
    
    // If we get here, both operations succeeded
    return metric * 2.0; // Some transformation
}

// Run all error handling demonstrations
void run_error_handling_examples() {
    std::cout << "Running BTQ Render Engine Error Handling Examples" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    demonstrate_user_friendly_errors();
    demonstrate_error_reporting();
    
    std::cout << "\n=== Demonstrating BTQ_TRY Macro ===" << std::endl;
    
    // Test with empty input (should fail at calculate_complex_metric)
    std::vector<double> empty_input;
    auto complex_result = complex_calculation_with_try_macro(empty_input);
    if (!complex_result) {
        std::cout << "Complex calculation failed as expected: " 
                  << to_string(complex_result.error()) << std::endl;
    }
    
    // Test with valid input
    std::vector<double> valid_input = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto valid_result = complex_calculation_with_try_macro(valid_input);
    if (valid_result) {
        std::cout << "Complex calculation succeeded: " << valid_result.value() << std::endl;
    } else {
        std::cout << "Complex calculation failed unexpectedly: " 
                  << to_string(valid_result.error()) << std::endl;
    }
    
    std::cout << "\n=== Demonstrating Exception to Error Conversion ===" << std::endl;
    
    // Example of converting exceptions to errors
    auto exception_handling_result = BTQ_CATCH_TO_ERROR([]() {
        // Simulate a function that might throw
        throw std::runtime_error("Simulated runtime error");
        return 42; // This won't be reached
    }, ErrorCode::kUnknownError, "Exception occurred in simulated function");
    
    if (!exception_handling_result) {
        std::cout << "Exception properly converted to error: " 
                  << to_string(exception_handling_result.error()) << std::endl;
    }
    
    std::cout << "\nError handling examples completed." << std::endl;
}

} // namespace btq