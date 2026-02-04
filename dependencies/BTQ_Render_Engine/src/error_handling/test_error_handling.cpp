#include "error_handling/result.hpp"
#include "error_handling/error_reporter.hpp"
#include "error_handling/user_friendly_errors.hpp"
#include <cassert>
#include <iostream>
#include <string>

namespace btq {
namespace test {

void test_basic_error_creation() {
    std::cout << "Testing basic error creation..." << std::endl;
    
    auto error = BTQ_MAKE_ERROR(ErrorCode::kInvalidArgument, "Test error message");
    assert(error.code == ErrorCode::kInvalidArgument);
    assert(error.message == "Test error message");
    
    std::cout << "✓ Basic error creation test passed" << std::endl;
}

void test_result_success() {
    std::cout << "Testing Result with success..." << std::endl;
    
    Result<int> success_result = 42;
    assert(success_result.has_value());
    assert(success_result.value() == 42);
    
    std::cout << "✓ Result success test passed" << std::endl;
}

void test_result_error() {
    std::cout << "Testing Result with error..." << std::endl;
    
    auto error_info = BTQ_MAKE_ERROR(ErrorCode::kOutOfRange, "Out of range test");
    Result<int> error_result = Result<int>(std::unexpect, error_info);
    
    assert(!error_result.has_value());
    assert(error_result.error().code == ErrorCode::kOutOfRange);
    
    std::cout << "✓ Result error test passed" << std::endl;
}

void test_error_code_to_string() {
    std::cout << "Testing error code to string conversion..." << std::endl;

    std::string error_str = btq::to_string(ErrorCode::kDivisionByZero);
    assert(error_str == "Division by Zero");

    std::string unknown_str = btq::to_string(ErrorCode::kUnknownError);
    assert(unknown_str == "Unknown Error");

    std::cout << "✓ Error code to string test passed" << std::endl;
}

void test_error_info_to_string() {
    std::cout << "Testing ErrorInfo to string conversion..." << std::endl;

    ErrorInfo error(ErrorCode::kFileNotFound, "File not found", "Detailed info", "test.cpp", 42, "test_func");
    std::string error_str = btq::to_string(error);

    // Just check that it produces some output
    assert(!error_str.empty());

    std::cout << "✓ ErrorInfo to string test passed" << std::endl;
}

void test_make_error_helper() {
    std::cout << "Testing make_error helper function..." << std::endl;
    
    auto error_result = make_error<int>(ErrorCode::kInvalidArgument, "Invalid arg test");
    assert(!error_result.has_value());
    assert(error_result.error().code == ErrorCode::kInvalidArgument);
    assert(error_result.error().message == "Invalid arg test");
    
    std::cout << "✓ make_error helper test passed" << std::endl;
}

void test_user_friendly_messages() {
    std::cout << "Testing user-friendly error messages..." << std::endl;
    
    std::string user_msg = UserFriendlyErrorMessage::generate_user_friendly_message(ErrorCode::kDivisionByZero);
    assert(user_msg.find("Division by zero") != std::string::npos);
    
    std::string detailed_msg = UserFriendlyErrorMessage::generate_detailed_error_message(ErrorCode::kOutOfMemory);
    assert(detailed_msg.find("Insufficient memory") != std::string::npos);
    assert(detailed_msg.find("Troubleshooting:") != std::string::npos);
    
    std::cout << "✓ User-friendly messages test passed" << std::endl;
}

void test_error_propagation_with_btq_try() {
    std::cout << "Testing BTQ_TRY macro for error propagation..." << std::endl;
    
    // Function that returns an error
    auto failing_func = []() -> Result<int> {
        return make_error<int>(ErrorCode::kInvalidArgument, "Intentional failure");
    };
    
    // Function that uses BTQ_TRY to propagate the error
    auto calling_func = [&failing_func]() -> Result<int> {
        int value = BTQ_TRY(failing_func());  // This should propagate the error
        return value + 10;  // This line should not be reached
    };
    
    auto result = calling_func();
    assert(!result.has_value());
    assert(result.error().code == ErrorCode::kInvalidArgument);
    
    std::cout << "✓ BTQ_TRY macro test passed" << std::endl;
}

void test_global_error_reporter() {
    std::cout << "Testing global error reporter..." << std::endl;
    
    // Initialize the global reporter
    GlobalErrorReporter::initialize();
    
    // Report a message
    report_message("Test message for global reporter");
    
    // Report an error
    auto error = BTQ_MAKE_ERROR(ErrorCode::kUnknownError, "Test error for global reporter");
    report_error(error, ErrorSeverity::kWarning);
    
    std::cout << "✓ Global error reporter test passed" << std::endl;
}

void run_all_tests() {
    std::cout << "Running BTQ Render Engine Error Handling Tests" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    test_basic_error_creation();
    test_result_success();
    test_result_error();
    test_error_code_to_string();
    test_error_info_to_string();
    test_make_error_helper();
    test_user_friendly_messages();
    test_error_propagation_with_btq_try();
    test_global_error_reporter();
    
    std::cout << "\n✓ All error handling tests passed!" << std::endl;
}

} // namespace test
} // namespace btq

int main() {
    btq::test::run_all_tests();
    return 0;
}