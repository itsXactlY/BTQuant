#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace BTQuant {
namespace Logging {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR, CRITICAL };

struct LogMetadata {
    std::string timestamp;
    std::string level;
    std::string file;
    int line;
    std::string function;
    std::string thread_id;
    
    LogMetadata() : line(0) {}
};

class StructuredLogger {
public:
    static StructuredLogger& getInstance() {
        static StructuredLogger instance;
        return instance;
    }

    // Configuration methods
    void setLogLevel(LogLevel level) { 
        std::lock_guard<std::mutex> lock(mutex_);
        log_level_ = level; 
    }
    
    LogLevel getLogLevel() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return log_level_; 
    }
    
    void enableConsoleLogging(bool enabled) { 
        std::lock_guard<std::mutex> lock(mutex_);
        console_logging_enabled_ = enabled; 
    }
    
    void enableFileLogging(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_file_ = std::make_unique<std::ofstream>(filename, std::ios::app);
        file_logging_enabled_ = true;
    }
    
    void disableFileLogging() {
        std::lock_guard<std::mutex> lock(mutex_);
        log_file_.reset();
        file_logging_enabled_ = false;
    }
    
    void setLogFormat(bool json_format) {
        std::lock_guard<std::mutex> lock(mutex_);
        json_format_ = json_format;
    }

    // Main logging methods
    template<typename... Args>
    void log(LogLevel level, const std::string& message, const char* file = "", 
             int line = 0, const char* function = "", Args... args) {
        if (level < log_level_) {
            return;
        }

        LogMetadata metadata;
        populateMetadata(metadata, level, file, line, function);

        std::lock_guard<std::mutex> lock(mutex_);
        
        std::string formatted_log;
        if (json_format_) {
            formatted_log = formatAsJSON(level, message, metadata, args...);
        } else {
            formatted_log = formatAsText(level, message, metadata, args...);
        }

        // Output to console if enabled
        if (console_logging_enabled_) {
            std::cout << formatted_log;
        }

        // Output to file if enabled
        if (file_logging_enabled_ && log_file_ && log_file_->is_open()) {
            *log_file_ << formatted_log;
            log_file_->flush();
        }
    }

    // Convenience methods
    template<typename... Args>
    void debug(const std::string& message, const char* file = "", 
               int line = 0, const char* function = "", Args... args) {
        log(LogLevel::DEBUG, message, file, line, function, args...);
    }

    template<typename... Args>
    void info(const std::string& message, const char* file = "", 
              int line = 0, const char* function = "", Args... args) {
        log(LogLevel::INFO, message, file, line, function, args...);
    }

    template<typename... Args>
    void warning(const std::string& message, const char* file = "", 
                 int line = 0, const char* function = "", Args... args) {
        log(LogLevel::WARNING, message, file, line, function, args...);
    }

    template<typename... Args>
    void error(const std::string& message, const char* file = "", 
               int line = 0, const char* function = "", Args... args) {
        log(LogLevel::ERROR, message, file, line, function, args...);
    }

    template<typename... Args>
    void critical(const std::string& message, const char* file = "", 
                  int line = 0, const char* function = "", Args... args) {
        log(LogLevel::CRITICAL, message, file, line, function, args...);
    }

private:
    StructuredLogger() : log_level_(LogLevel::INFO), 
                         console_logging_enabled_(true), 
                         file_logging_enabled_(false),
                         json_format_(false) {}
    
    ~StructuredLogger() {
        if (log_file_ && log_file_->is_open()) {
            log_file_->close();
        }
    }

    void populateMetadata(LogMetadata& metadata, LogLevel level, 
                          const char* file, int line, const char* function) {
        // Timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::ostringstream ts_stream;
        ts_stream << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
        ts_stream << '.' << std::setfill('0') << std::setw(3) << ms.count();
        metadata.timestamp = ts_stream.str();

        // Level string
        switch (level) {
            case LogLevel::DEBUG:    metadata.level = "DEBUG";    break;
            case LogLevel::INFO:     metadata.level = "INFO";     break;
            case LogLevel::WARNING:  metadata.level = "WARNING";  break;
            case LogLevel::ERROR:    metadata.level = "ERROR";    break;
            case LogLevel::CRITICAL: metadata.level = "CRITICAL"; break;
        }

        // File and line
        metadata.file = file ? file : "";
        metadata.line = line;
        metadata.function = function ? function : "";

        // Thread ID
        std::ostringstream thread_stream;
        thread_stream << std::this_thread::get_id();
        metadata.thread_id = thread_stream.str();
    }

    template<typename... Args>
    std::string formatAsText(LogLevel level, const std::string& message,
                             const LogMetadata& metadata, Args... args) {
        std::ostringstream oss;
        oss << "[" << metadata.timestamp << "] "
            << "[" << metadata.level << "] "
            << message;

        // Add additional key-value pairs if provided
        if constexpr (sizeof...(args) > 0) {
            oss << " |";
            addKeyValuePairs(oss, args...);
        }

        oss << " (File: " << metadata.file
            << ", Line: " << metadata.line
            << ", Func: " << metadata.function
            << ", Thread: " << metadata.thread_id << ")"
            << std::endl;

        return oss.str();
    }

    template<typename... Args>
    std::string formatAsJSON(LogLevel level, const std::string& message,
                             const LogMetadata& metadata, Args... args) {
        std::ostringstream oss;
        oss << "{"
            << "\"timestamp\":\"" << metadata.timestamp << "\","
            << "\"level\":\"" << metadata.level << "\","
            << "\"message\":\"" << escapeJson(message) << "\","
            << "\"file\":\"" << metadata.file << "\","
            << "\"line\":" << metadata.line << ","
            << "\"function\":\"" << metadata.function << "\","
            << "\"thread_id\":\"" << metadata.thread_id << "\"";

        // Add additional key-value pairs if provided
        if constexpr (sizeof...(args) > 0) {
            addKeyValuePairsJSON(oss, args...);
        }

        oss << "}" << std::endl;
        return oss.str();
    }

    // Helper to add key-value pairs to text format
    template<typename T>
    void addKeyValuePairs(std::ostringstream& oss, const std::string& key, const T& value) {
        oss << " " << key << "=" << value;
    }

    template<typename T, typename... Rest>
    void addKeyValuePairs(std::ostringstream& oss, const std::string& key, const T& value, Rest... rest) {
        addKeyValuePairs(oss, key, value);
        if (sizeof...(rest) > 0) {
            addKeyValuePairs(oss, rest...);
        }
    }

    // Helper to add key-value pairs to JSON format
    template<typename T>
    void addKeyValuePairsJSON(std::ostringstream& oss, const std::string& key, const T& value) {
        oss << ",\"" << key << "\":";
        if constexpr (std::is_same_v<T, std::string>) {
            oss << "\"" << escapeJson(value) << "\"";
        } else {
            oss << value;
        }
    }

    template<typename T, typename... Rest>
    void addKeyValuePairsJSON(std::ostringstream& oss, const std::string& key, const T& value, Rest... rest) {
        oss << ",\"" << key << "\":";
        if constexpr (std::is_same_v<T, std::string>) {
            oss << "\"" << escapeJson(value) << "\"";
        } else {
            oss << value;
        }
        if (sizeof...(rest) > 0) {
            addKeyValuePairsJSON(oss, rest...);
        }
    }

    std::string escapeJson(const std::string& str) {
        std::string result;
        for (char c : str) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c; break;
            }
        }
        return result;
    }

    mutable std::mutex mutex_;
    LogLevel log_level_;
    bool console_logging_enabled_;
    bool file_logging_enabled_;
    bool json_format_;
    std::unique_ptr<std::ofstream> log_file_;
};

// Initialization function declarations
void initializeLoggingSystem();
void configureLoggingFromSettings(const std::string& configFile);

// Convenience macros
#define BTQ_LOG_DEBUG_EX(msg, ...) \
    BTQuant::Logging::StructuredLogger::getInstance().debug(msg, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)

#define BTQ_LOG_INFO_EX(msg, ...) \
    BTQuant::Logging::StructuredLogger::getInstance().info(msg, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)

#define BTQ_LOG_WARNING_EX(msg, ...) \
    BTQuant::Logging::StructuredLogger::getInstance().warning(msg, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)

#define BTQ_LOG_ERROR_EX(msg, ...) \
    BTQuant::Logging::StructuredLogger::getInstance().error(msg, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)

#define BTQ_LOG_CRITICAL_EX(msg, ...) \
    BTQuant::Logging::StructuredLogger::getInstance().critical(msg, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)

// Simple macros without extra parameters
#define BTQ_LOG_DEBUG(msg) \
    BTQuant::Logging::StructuredLogger::getInstance().debug(msg, __FILE__, __LINE__, __FUNCTION__)

#define BTQ_LOG_INFO(msg) \
    BTQuant::Logging::StructuredLogger::getInstance().info(msg, __FILE__, __LINE__, __FUNCTION__)

#define BTQ_LOG_WARNING(msg) \
    BTQuant::Logging::StructuredLogger::getInstance().warning(msg, __FILE__, __LINE__, __FUNCTION__)

#define BTQ_LOG_ERROR(msg) \
    BTQuant::Logging::StructuredLogger::getInstance().error(msg, __FILE__, __LINE__, __FUNCTION__)

#define BTQ_LOG_CRITICAL(msg) \
    BTQuant::Logging::StructuredLogger::getInstance().critical(msg, __FILE__, __LINE__, __FUNCTION__)

}  // namespace Logging
}  // namespace BTQuant