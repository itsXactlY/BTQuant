#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <fstream>

namespace BTQuant {
namespace Utils {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class DynamicLogger {
public:
    static DynamicLogger& getInstance() {
        static DynamicLogger instance;
        return instance;
    }

    void setLogLevel(LogLevel level) { log_level_ = level; }
    void enableFileLogging(const std::string& filename) {
        log_file_ = std::make_unique<std::ofstream>(filename, std::ios::app);
    }

    void log(LogLevel level, const std::string& message, const char* file = "", int line = 0) {
        if (level < log_level_) {
            return;
        }

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] ";

        switch(level) {
            case LogLevel::DEBUG: oss << "[DEBUG] "; break;
            case LogLevel::INFO: oss << "[INFO] "; break;
            case LogLevel::WARNING: oss << "[WARNING] "; break;
            case LogLevel::ERROR: oss << "[ERROR] "; break;
            case LogLevel::CRITICAL: oss << "[CRITICAL] "; break;
        }

        oss << message;

        if (file && strlen(file) > 0) {
            oss << " (File: " << file << ", Line: " << line << ")";
        }

        oss << std::endl;

        std::string log_msg = oss.str();
        std::cout << log_msg;

        if (log_file_ && log_file_->is_open()) {
            *log_file_ << log_msg;
            log_file_->flush();
        }
    }

    void debug(const std::string& msg, const char* file = "", int line = 0) { log(LogLevel::DEBUG, msg, file, line); }
    void info(const std::string& msg, const char* file = "", int line = 0) { log(LogLevel::INFO, msg, file, line); }
    void warning(const std::string& msg, const char* file = "", int line = 0) { log(LogLevel::WARNING, msg, file, line); }
    void error(const std::string& msg, const char* file = "", int line = 0) { log(LogLevel::ERROR, msg, file, line); }
    void critical(const std::string& msg, const char* file = "", int line = 0) { log(LogLevel::CRITICAL, msg, file, line); }

private:
    DynamicLogger() : log_level_(LogLevel::INFO) {}
    ~DynamicLogger() = default;
    DynamicLogger(const DynamicLogger&) = delete;
    DynamicLogger& operator=(const DynamicLogger&) = delete;

    LogLevel log_level_;
    std::unique_ptr<std::ofstream> log_file_;
};

#define BTQ_LOG_DEBUG(msg) BTQuant::Utils::DynamicLogger::getInstance().debug(msg, __FILE__, __LINE__)
#define BTQ_LOG_INFO(msg) BTQuant::Utils::DynamicLogger::getInstance().info(msg, __FILE__, __LINE__)
#define BTQ_LOG_WARNING(msg) BTQuant::Utils::DynamicLogger::getInstance().warning(msg, __FILE__, __LINE__)
#define BTQ_LOG_ERROR(msg) BTQuant::Utils::DynamicLogger::getInstance().error(msg, __FILE__, __LINE__)
#define BTQ_LOG_CRITICAL(msg) BTQuant::Utils::DynamicLogger::getInstance().critical(msg, __FILE__, __LINE__)

} // namespace Utils
} // namespace BTQuant
