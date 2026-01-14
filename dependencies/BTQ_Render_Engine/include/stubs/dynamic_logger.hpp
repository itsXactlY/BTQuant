#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>

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
    
    void log(LogLevel level, const std::string& message) {
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
        
        oss << message << std::endl;
        std::cout << oss.str();
    }
    
    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
    void critical(const std::string& msg) { log(LogLevel::CRITICAL, msg); }
    
private:
    DynamicLogger() = default;
    ~DynamicLogger() = default;
    DynamicLogger(const DynamicLogger&) = delete;
    DynamicLogger& operator=(const DynamicLogger&) = delete;
};

} // namespace Utils
} // namespace BTQuant
