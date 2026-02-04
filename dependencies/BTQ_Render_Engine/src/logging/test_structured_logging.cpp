#include "../../include/structured_logger.hpp"
#include <iostream>
#include <thread>

int main() {
    // Get the logger instance
    auto& logger = BTQuant::Logging::StructuredLogger::getInstance();
    
    // Test different log levels
    BTQ_LOG_DEBUG("This is a debug message");
    BTQ_LOG_INFO("This is an info message");
    BTQ_LOG_WARNING("This is a warning message");
    BTQ_LOG_ERROR("This is an error message");
    BTQ_LOG_CRITICAL("This is a critical message");
    
    // Test structured logging with additional parameters
    BTQ_LOG_INFO_EX("User login event", "user_id", 12345, "ip_address", "192.168.1.100", "success", true);
    BTQ_LOG_ERROR_EX("Database connection failed", "host", "db.example.com", "port", 5432, "retry_count", 3);
    
    // Change log level to see only warnings and above
    logger.setLogLevel(BTQuant::Logging::LogLevel::WARNING);
    std::cout << "\nChanged log level to WARNING - only warnings and above should appear:\n" << std::endl;
    
    BTQ_LOG_DEBUG("This debug message should not appear");
    BTQ_LOG_INFO("This info message should not appear");
    BTQ_LOG_WARNING("This warning message should appear");
    BTQ_LOG_ERROR("This error message should appear");
    
    // Test JSON format
    std::cout << "\nEnabling JSON format:\n" << std::endl;
    logger.setLogFormat(true);
    
    BTQ_LOG_INFO_EX("Application started", "version", "1.0.0", "process_id", 1234);
    BTQ_LOG_WARNING_EX("Low disk space", "available_space_mb", 1024, "threshold_mb", 2048);
    
    // Test file logging
    logger.enableFileLogging("test_application.log");
    BTQ_LOG_INFO_EX("This will be logged to file", "test_case", "file_logging", "timestamp", "now");
    
    std::cout << "\nLogging test completed. Check 'test_application.log' for file output." << std::endl;
    
    return 0;
}