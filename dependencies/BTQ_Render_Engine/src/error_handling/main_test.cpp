#include "error_handling/result.hpp"
#include "error_handling/error_reporter.hpp"
#include "error_handling/user_friendly_errors.hpp"
#include <iostream>

int main() {
    std::cout << "Testing BTQ Render Engine Error Handling System" << std::endl;
    
    // Test basic error creation
    auto error = BTQ_MAKE_ERROR(btq::ErrorCode::kInvalidArgument, "Test error message");
    std::cout << "Created error: " << btq::to_string(error) << std::endl;
    
    // Test Result with success
    btq::Result<int> success_result = 42;
    if (success_result.has_value()) {
        std::cout << "Success result value: " << success_result.value() << std::endl;
    }
    
    // Test Result with error
    auto error_result = btq::make_error<int>(
        btq::ErrorCode::kDivisionByZero,
        "Division by zero occurred",
        "Attempted to divide by zero"
    );

    if (!error_result.has_value()) {
        std::cout << "Error result: " << btq::to_string(error_result.error()) << std::endl;
        std::cout << "User-friendly message: "
                  << btq::UserFriendlyErrorMessage::generate_user_friendly_message(error_result.error().code)
                  << std::endl;
    }

    // Test error reporting
    btq::GlobalErrorReporter::initialize();
    btq::report_error(error, btq::ErrorSeverity::kError);
    btq::report_message("Test message reported successfully", btq::ErrorSeverity::kInfo);
    
    std::cout << "Error handling system test completed successfully!" << std::endl;
    
    return 0;
}