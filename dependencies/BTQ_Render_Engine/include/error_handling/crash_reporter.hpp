#pragma once

#include "error_handling/result.hpp"
#include "error_handling/error_reporter.hpp"
#include <string>
#include <atomic>
#include <csignal>

#ifdef _WIN32
#include <windows.h>
#include <excpt.h>
#endif

namespace btq {

// Crash reporter class for handling crashes and sending telemetry
class CrashReporter {
public:
    // Initialize crash reporting system
    static void initialize(const std::string& dump_dir = "./crash_dumps",
                          const std::string& telemetry_url = "");

    // Enable/disable crash reporting
    static void enable_crash_reporting(bool enable);

    // Set user opt-out preference
    static void set_user_opt_out(bool opt_out);

    // Check if user has opted out
    static bool is_user_opt_out();

    // Check if crash reporting is enabled
    static bool is_crash_reporting_enabled();

    // Set application version for crash reports
    static void set_app_version(const std::string& version);

    // Generate a crash dump file
    static std::string generate_crash_dump(const std::string& crash_info);

    // Send a telemetry event
    static void send_telemetry_event(const std::string& event_type,
                                   const std::string& details);

    // Report error telemetry
    static void report_error_telemetry(const ErrorInfo& error, ErrorSeverity severity);

private:
    // Signal handler for catching crashes
    static void signal_handler(int signal);

#ifdef _WIN32
    // Windows exception handler
    static LONG exception_handler(PEXCEPTION_POINTERS ExceptionInfo);
#endif

    // Get stack trace for crash dumps
    static std::string get_stack_trace();
};

} // namespace btq