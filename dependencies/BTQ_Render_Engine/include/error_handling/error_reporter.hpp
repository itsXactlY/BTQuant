#pragma once

#include "error_handling/result.hpp"
#include "error_handling/crash_reporter.hpp"
#include <functional>
#include <memory>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

namespace btq {

// Error severity levels
enum class ErrorSeverity {
    kDebug = 0,
    kInfo,
    kWarning,
    kError,
    kCritical
};

// Interface for error reporters
class ErrorReporter {
public:
    virtual ~ErrorReporter() = default;
    virtual void report_error(const ErrorInfo& error, ErrorSeverity severity = ErrorSeverity::kError) = 0;
    virtual void report_message(const std::string& message, ErrorSeverity severity = ErrorSeverity::kInfo) = 0;
};

// Console error reporter
class ConsoleErrorReporter : public ErrorReporter {
public:
    void report_error(const ErrorInfo& error, ErrorSeverity severity = ErrorSeverity::kError) override {
        std::string prefix;
        switch (severity) {
            case ErrorSeverity::kDebug: prefix = "[DEBUG] "; break;
            case ErrorSeverity::kInfo: prefix = "[INFO] "; break;
            case ErrorSeverity::kWarning: prefix = "[WARNING] "; break;
            case ErrorSeverity::kError: prefix = "[ERROR] "; break;
            case ErrorSeverity::kCritical: prefix = "[CRITICAL] "; break;
        }
        
        std::cerr << prefix << btq::to_string(error) << std::endl;
    }

    void report_message(const std::string& message, ErrorSeverity severity = ErrorSeverity::kInfo) override {
        std::string prefix;
        switch (severity) {
            case ErrorSeverity::kDebug: prefix = "[DEBUG] "; break;
            case ErrorSeverity::kInfo: prefix = "[INFO] "; break;
            case ErrorSeverity::kWarning: prefix = "[WARNING] "; break;
            case ErrorSeverity::kError: prefix = "[ERROR] "; break;
            case ErrorSeverity::kCritical: prefix = "[CRITICAL] "; break;
        }
        
        std::cout << prefix << message << std::endl;
    }
};

// File error reporter
class FileErrorReporter : public ErrorReporter {
private:
    std::ofstream log_file_;
    std::string filename_;
    
public:
    explicit FileErrorReporter(const std::string& filename) : filename_(filename) {
        log_file_.open(filename, std::ios::app);
        if (!log_file_.is_open()) {
            throw std::runtime_error("Could not open log file: " + filename);
        }
    }
    
    ~FileErrorReporter() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
    
    void report_error(const ErrorInfo& error, ErrorSeverity severity = ErrorSeverity::kError) override {
        if (!log_file_.is_open()) {
            return;
        }
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::string severity_str;
        switch (severity) {
            case ErrorSeverity::kDebug: severity_str = "DEBUG"; break;
            case ErrorSeverity::kInfo: severity_str = "INFO"; break;
            case ErrorSeverity::kWarning: severity_str = "WARNING"; break;
            case ErrorSeverity::kError: severity_str = "ERROR"; break;
            case ErrorSeverity::kCritical: severity_str = "CRITICAL"; break;
        }
        
        log_file_ << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        log_file_ << '.' << std::setfill('0') << std::setw(3) << ms.count();
        log_file_ << " [" << severity_str << "] " << btq::to_string(error) << std::endl;
        log_file_.flush();
    }

    void report_message(const std::string& message, ErrorSeverity severity = ErrorSeverity::kInfo) override {
        if (!log_file_.is_open()) {
            return;
        }
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::string severity_str;
        switch (severity) {
            case ErrorSeverity::kDebug: severity_str = "DEBUG"; break;
            case ErrorSeverity::kInfo: severity_str = "INFO"; break;
            case ErrorSeverity::kWarning: severity_str = "WARNING"; break;
            case ErrorSeverity::kError: severity_str = "ERROR"; break;
            case ErrorSeverity::kCritical: severity_str = "CRITICAL"; break;
        }
        
        log_file_ << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        log_file_ << '.' << std::setfill('0') << std::setw(3) << ms.count();
        log_file_ << " [" << severity_str << "] " << message << std::endl;
        log_file_.flush();
    }
};

// Composite error reporter that sends to multiple reporters
class CompositeErrorReporter : public ErrorReporter {
private:
    std::vector<std::unique_ptr<ErrorReporter>> reporters_;
    
public:
    void add_reporter(std::unique_ptr<ErrorReporter> reporter) {
        reporters_.push_back(std::move(reporter));
    }
    
    void report_error(const ErrorInfo& error, ErrorSeverity severity = ErrorSeverity::kError) override {
        for (auto& reporter : reporters_) {
            reporter->report_error(error, severity);
        }
    }

    void report_message(const std::string& message, ErrorSeverity severity = ErrorSeverity::kInfo) override {
        for (auto& reporter : reporters_) {
            reporter->report_message(message, severity);
        }
    }
};

// Global error reporter singleton
class GlobalErrorReporter {
private:
    static inline std::unique_ptr<CompositeErrorReporter> global_reporter_ = nullptr;
    
public:
    static void initialize() {
        if (!global_reporter_) {
            global_reporter_ = std::make_unique<CompositeErrorReporter>();
            
            // Add console reporter by default
            global_reporter_->add_reporter(std::make_unique<ConsoleErrorReporter>());
        }
    }
    
    static void add_reporter(std::unique_ptr<ErrorReporter> reporter) {
        if (!global_reporter_) {
            initialize();
        }
        global_reporter_->add_reporter(std::move(reporter));
    }
    
    static ErrorReporter& get() {
        if (!global_reporter_) {
            initialize();
        }
        return *global_reporter_;
    }
    
    static void shutdown() {
        global_reporter_.reset();
    }
};

// Utility functions for error reporting
inline void report_error(const ErrorInfo& error, ErrorSeverity severity = ErrorSeverity::kError) {
    GlobalErrorReporter::get().report_error(error, severity);
}

inline void report_error(ErrorCode code, const std::string& message,
                         const std::string& details = "",
                         const std::string& file = "",
                         int line = 0,
                         const std::string& func = "") {
    ErrorInfo error(code, message, details, file, line, func);
    GlobalErrorReporter::get().report_error(error, ErrorSeverity::kError);
}

inline void report_message(const std::string& message, ErrorSeverity severity = ErrorSeverity::kInfo) {
    GlobalErrorReporter::get().report_message(message, severity);
}

// RAII wrapper for automatic error cleanup
template<typename CleanupFunc>
class ErrorCleanup {
private:
    CleanupFunc cleanup_func_;
    bool active_ = true;
    
public:
    explicit ErrorCleanup(CleanupFunc func) : cleanup_func_(std::move(func)) {}
    
    ~ErrorCleanup() {
        if (active_) {
            cleanup_func_();
        }
    }
    
    void release() { active_ = false; }
};

template<typename CleanupFunc>
auto make_cleanup(CleanupFunc&& func) {
    return ErrorCleanup<std::decay_t<CleanupFunc>>(std::forward<CleanupFunc>(func));
}

} // namespace btq