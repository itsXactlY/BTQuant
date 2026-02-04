#include "error_handling/crash_reporter.hpp"
#include "error_handling/error_reporter.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Initializing crash reporting system..." << std::endl;
    
    // Initialize crash reporter
    btq::CrashReporter::initialize("./test_crash_dumps", "https://telemetry.example.com");
    btq::CrashReporter::set_app_version("2.0.0");
    
    // Initialize global error reporter
    btq::GlobalErrorReporter::initialize();
    
    std::cout << "Crash reporting enabled: " << btq::CrashReporter::is_crash_reporting_enabled() << std::endl;
    std::cout << "User opt-out status: " << btq::CrashReporter::is_user_opt_out() << std::endl;
    
    // Test user opt-out
    std::cout << "\nSetting user opt-out to true..." << std::endl;
    btq::CrashReporter::set_user_opt_out(true);
    std::cout << "Crash reporting enabled after opt-out: " << btq::CrashReporter::is_crash_reporting_enabled() << std::endl;
    
    // Test with opt-out disabled
    std::cout << "\nSetting user opt-out to false..." << std::endl;
    btq::CrashReporter::set_user_opt_out(false);
    std::cout << "Crash reporting enabled after opt-in: " << btq::CrashReporter::is_crash_reporting_enabled() << std::endl;
    
    // Test error reporting with telemetry
    std::cout << "\nTesting error reporting with telemetry..." << std::endl;
    auto error = BTQ_MAKE_ERROR_WITH_DETAILS(
        btq::ErrorCode::kInvalidArgument,
        "Test error for telemetry",
        "This is a test error to verify telemetry functionality"
    );
    
    btq::report_error(error, btq::ErrorSeverity::kError);
    
    // Test message reporting
    btq::report_message("Test message for crash reporter integration", btq::ErrorSeverity::kInfo);
    
    // Test crash dump generation
    std::cout << "\nTesting crash dump generation..." << std::endl;
    std::string dump_path = btq::CrashReporter::generate_crash_dump("Test crash dump generation");
    std::cout << "Crash dump generated at: " << dump_path << std::endl;
    
    // Test telemetry event
    std::cout << "\nTesting telemetry event..." << std::endl;
    btq::CrashReporter::send_telemetry_event("test_event", "Test telemetry functionality");
    
    std::cout << "\nCrash reporting test completed successfully!" << std::endl;
    
    return 0;
}