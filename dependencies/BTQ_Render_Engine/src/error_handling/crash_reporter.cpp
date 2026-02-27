#include "error_handling/crash_reporter.hpp"
#include "error_handling/error_reporter.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <csignal>
#include <cstring>
#include <sys/stat.h>
#include <ctime>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#else
#include <execinfo.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace btq {

// Static variables for crash reporting
static std::atomic<bool> crash_reporting_enabled_{true};
static std::atomic<bool> user_opt_out_{false};
static std::string crash_dump_directory_ = "./crash_dumps";
static std::string telemetry_endpoint_ = "";
static std::string app_version_ = "1.0.0";

void CrashReporter::initialize(const std::string& dump_dir, const std::string& telemetry_url) {
    // Set up crash dump directory
    if (!dump_dir.empty()) {
        crash_dump_directory_ = dump_dir;
    }
    
    // Create directory if it doesn't exist
    try {
        fs::create_directories(crash_dump_directory_);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create crash dump directory: " << e.what() << std::endl;
    }
    
    // Set up telemetry endpoint
    if (!telemetry_url.empty()) {
        telemetry_endpoint_ = telemetry_url;
    }
    
    // Register signal handlers for crash detection
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);
    std::signal(SIGFPE, signal_handler);
    std::signal(SIGILL, signal_handler);
    std::signal(SIGBUS, signal_handler);
    
#ifdef _WIN32
    // Windows-specific setup for minidump generation
    SetUnhandledExceptionFilter(exception_handler);
#endif
}

void CrashReporter::enable_crash_reporting(bool enable) {
    crash_reporting_enabled_.store(enable);
}

void CrashReporter::set_user_opt_out(bool opt_out) {
    user_opt_out_.store(opt_out);
}

bool CrashReporter::is_user_opt_out() {
    return user_opt_out_.load();
}

bool CrashReporter::is_crash_reporting_enabled() {
    return crash_reporting_enabled_.load() && !user_opt_out_.load();
}

void CrashReporter::set_app_version(const std::string& version) {
    app_version_ = version;
}

std::string CrashReporter::generate_crash_dump(const std::string& crash_info) {
    if (!is_crash_reporting_enabled()) {
        return "";
    }
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream filename_ss;
    filename_ss << "crash_dump_" 
                << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
                << "_" << ms.count() << ".dmp";
    
    std::string filepath = crash_dump_directory_ + "/" + filename_ss.str();
    
    try {
        std::ofstream dump_file(filepath, std::ios::out | std::ios::binary);
        if (dump_file.is_open()) {
            // Write crash information to the dump file
            dump_file << "BTQ Render Engine Crash Dump\n";
            dump_file << "===========================\n";
            dump_file << "Timestamp: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                      << "." << std::setfill('0') << std::setw(3) << ms.count() << "\n";
            dump_file << "Version: " << app_version_ << "\n";
            dump_file << "Crash Info: " << crash_info << "\n";
            
            // Add stack trace if available
            dump_file << "\nStack Trace:\n";
            dump_file << get_stack_trace() << "\n";
            
            dump_file.close();
            
            std::cout << "Crash dump saved to: " << filepath << std::endl;
            return filepath;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to write crash dump: " << e.what() << std::endl;
    }
    
    return "";
}

void CrashReporter::send_telemetry_event(const std::string& event_type, const std::string& details) {
    if (!is_crash_reporting_enabled() || telemetry_endpoint_.empty()) {
        return;
    }
    
    // In a real implementation, this would send data to a telemetry service
    // For now, we'll just log the telemetry event
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream log_entry;
    log_entry << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
              << " [TELEMETRY] Event: " << event_type 
              << ", Details: " << details
              << ", Version: " << app_version_ << std::endl;
    
    // Log to a telemetry file
    try {
        std::ofstream telemetry_log(crash_dump_directory_ + "/telemetry.log", std::ios::app);
        if (telemetry_log.is_open()) {
            telemetry_log << log_entry.str();
            telemetry_log.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to log telemetry event: " << e.what() << std::endl;
    }
}

void CrashReporter::signal_handler(int signal) {
    if (!is_crash_reporting_enabled()) {
        // Re-raise the signal to allow default behavior
        std::signal(signal, SIG_DFL);
        raise(signal);
        return;
    }
    
    std::string signal_name;
    switch (signal) {
        case SIGSEGV: signal_name = "SIGSEGV (Segmentation fault)"; break;
        case SIGABRT: signal_name = "SIGABRT (Abort)"; break;
        case SIGFPE:  signal_name = "SIGFPE (Floating point exception)"; break;
        case SIGILL:  signal_name = "SIGILL (Illegal instruction)"; break;
        case SIGBUS:  signal_name = "SIGBUS (Bus error)"; break;
        default:      signal_name = "Unknown signal (" + std::to_string(signal) + ")"; break;
    }
    
    std::string crash_info = "Signal: " + signal_name;
    std::string dump_path = generate_crash_dump(crash_info);
    
    // Send telemetry about the crash
    send_telemetry_event("application_crash", "Signal: " + signal_name);
    
    // Report the crash through the error handling system
    if (dump_path.empty()) {
        btq::report_error(BTQ_MAKE_ERROR_WITH_DETAILS(
            btq::ErrorCode::kUnknownError,
            "Application crashed",
            "Signal: " + signal_name
        ), btq::ErrorSeverity::kCritical);
    } else {
        btq::report_error(BTQ_MAKE_ERROR_WITH_DETAILS(
            btq::ErrorCode::kUnknownError,
            "Application crashed, dump saved",
            "Dump file: " + dump_path + ", Signal: " + signal_name
        ), btq::ErrorSeverity::kCritical);
    }
    
    // Re-raise the signal to allow default behavior
    std::signal(signal, SIG_DFL);
    raise(signal);
}

#ifdef _WIN32
LONG CrashReporter::exception_handler(PEXCEPTION_POINTERS ExceptionInfo) {
    if (!is_crash_reporting_enabled()) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    
    std::stringstream crash_info;
    crash_info << "Windows Exception: 0x" << std::hex << ExceptionInfo->ExceptionRecord->ExceptionCode;
    crash_info << " at address 0x" << ExceptionInfo->ExceptionRecord->ExceptionAddress;
    
    std::string dump_path = generate_crash_dump(crash_info.str());
    
    // Send telemetry about the crash
    send_telemetry_event("application_crash", crash_info.str());
    
    // Report the crash through the error handling system
    if (dump_path.empty()) {
        btq::report_error(BTQ_MAKE_ERROR_WITH_DETAILS(
            btq::ErrorCode::kUnknownError,
            "Application crashed with Windows exception",
            crash_info.str()
        ), btq::ErrorSeverity::kCritical);
    } else {
        btq::report_error(BTQ_MAKE_ERROR_WITH_DETAILS(
            btq::ErrorCode::kUnknownError,
            "Application crashed with Windows exception, dump saved",
            "Dump file: " + dump_path + ", " + crash_info.str()
        ), btq::ErrorSeverity::kCritical);
    }
    
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

std::string CrashReporter::get_stack_trace() {
    std::stringstream ss;
    
#ifdef _WIN32
    // Windows stack trace implementation
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    
    CONTEXT context;
    RtlCaptureContext(&context);
    
    DWORD image;
    STACKFRAME64 stackframe;
    ZeroMemory(&stackframe, sizeof(STACKFRAME64));
    
#ifdef _M_IX86
    image = IMAGE_FILE_MACHINE_I386;
    stackframe.AddrPC.Offset = context.Eip;
    stackframe.AddrPC.Mode = AddrModeFlat;
    stackframe.AddrFrame.Offset = context.Ebp;
    stackframe.AddrFrame.Mode = AddrModeFlat;
    stackframe.AddrStack.Offset = context.Esp;
    stackframe.AddrStack.Mode = AddrModeFlat;
#elif _M_X64
    image = IMAGE_FILE_MACHINE_AMD64;
    stackframe.AddrPC.Offset = context.Rip;
    stackframe.AddrPC.Mode = AddrModeFlat;
    stackframe.AddrFrame.Offset = context.Rsp;
    stackframe.AddrFrame.Mode = AddrModeFlat;
    stackframe.AddrStack.Offset = context.Rsp;
    stackframe.AddrStack.Mode = AddrModeFlat;
#endif
    
    SymInitialize(process, NULL, TRUE);
    DWORD64 displacement = 0;
    
    for (ULONG frame = 0; ; frame++) {
        BOOL result = StackWalk64(
            image, process, thread,
            &stackframe, &context, NULL,
            SymFunctionTableAccess64, SymGetModuleBase64, NULL);
        
        if (!result) break;
        
        char symbol_buffer[sizeof(SYMBOL_INFO) + 256];
        PSYMBOL_INFO symbol = reinterpret_cast<PSYMBOL_INFO>(symbol_buffer);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 255;
        
        if (SymFromAddr(process, stackframe.AddrPC.Offset, &displacement, symbol)) {
            ss << "#" << frame << " 0x" << std::hex << symbol->Address 
               << ": " << symbol->Name << std::dec << "\n";
        } else {
            ss << "#" << frame << " 0x" << std::hex << stackframe.AddrPC.Offset 
               << ": <unknown>" << std::dec << "\n";
        }
        
        if (stackframe.AddrReturn == 0) break;
    }
    
    SymCleanup(process);
#else
    // Linux/Unix stack trace implementation
    const int max_frames = 64;
    void* buffer[max_frames];
    int num_frames = backtrace(buffer, max_frames);
    char** symbols = backtrace_symbols(buffer, num_frames);
    
    if (symbols != nullptr) {
        for (int i = 0; i < num_frames; ++i) {
            ss << "#" << i << " " << symbols[i] << "\n";
        }
        free(symbols);
    } else {
        ss << "Unable to obtain stack trace\n";
    }
#endif
    
    return ss.str();
}

void CrashReporter::report_error_telemetry(const ErrorInfo& error, ErrorSeverity severity) {
    if (!is_crash_reporting_enabled()) {
        return;
    }
    
    std::string severity_str;
    switch (severity) {
        case ErrorSeverity::kDebug: severity_str = "debug"; break;
        case ErrorSeverity::kInfo: severity_str = "info"; break;
        case ErrorSeverity::kWarning: severity_str = "warning"; break;
        case ErrorSeverity::kError: severity_str = "error"; break;
        case ErrorSeverity::kCritical: severity_str = "critical"; break;
    }
    
    std::string event_details = "Code: " + std::to_string(static_cast<int>(error.code)) +
                               ", Message: " + error.message +
                               ", Severity: " + severity_str;
    
    if (!error.details.empty()) {
        event_details += ", Details: " + error.details;
    }
    
    send_telemetry_event("error_occurred", event_details);
}

} // namespace btq