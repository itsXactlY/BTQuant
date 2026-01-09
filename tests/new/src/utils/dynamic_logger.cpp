#include "utils/dynamic_logger.hpp"
#include "config/config_loader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <mutex>

namespace BTQuant::Logging {

// Console log sink implementation
void ConsoleLogSink::write(const LogMessage& message) {
    std::ostringstream ss;
    
    // Format timestamp
    ss << message.timestamp << " ";
    
    // Format level
    const char* level_str[] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};
    ss << "[" << level_str[static_cast<int>(message.level)] << "] ";
    
    // Format component
    if (!message.component.empty()) {
        ss << "[" << message.component << "] ";
    }
    
    // Format message
    ss << message.message;
    
    // Format location if available
    if (!message.file.empty()) {
        ss << " (" << message.file << ":" << message.line << ")";
    }
    
    // Output to console
    if (message.level >= LogLevel::WARNING) {
        std::cerr << ss.str() << std::endl;
    } else {
        std::cout << ss.str() << std::endl;
    }
}

void ConsoleLogSink::flush() {
    std::cout.flush();
    std::cerr.flush();
}

bool ConsoleLogSink::is_enabled(LogLevel level) const {
    return true; // Always enabled for console
}

// File log sink implementation
FileLogSink::FileLogSink(const std::string& path) : path_(path) {
    open(path);
}

FileLogSink::~FileLogSink() {
    close();
}

bool FileLogSink::open(const std::string& path) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    file_ = fopen(path.c_str(), "a");
    return file_ != nullptr;
}

void FileLogSink::close() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
}

void FileLogSink::write(const LogMessage& message) {
    if (!file_) return;
    
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    std::ostringstream ss;
    ss << message.timestamp << " ";
    
    const char* level_str[] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};
    ss << "[" << level_str[static_cast<int>(message.level)] << "] ";
    
    if (!message.component.empty()) {
        ss << "[" << message.component << "] ";
    }
    
    ss << message.message;
    
    if (!message.file.empty()) {
        ss << " (" << message.file << ":" << message.line << ")";
    }
    
    ss << "\n";
    
    fprintf(file_, "%s", ss.str().c_str());
}

void FileLogSink::flush() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (file_) {
        fflush(file_);
    }
}

bool FileLogSink::is_enabled(LogLevel level) const {
    return file_ != nullptr;
}

// Dynamic logger implementation
DynamicLogger& DynamicLogger::instance() {
    static DynamicLogger logger;
    return logger;
}

bool DynamicLogger::initialize(const std::string& config_path) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    
    if (initialized_.load()) {
        return true;
    }
    
    // Add default console sink
    add_sink(std::make_shared<ConsoleLogSink>());
    
    initialized_.store(true);
    return true;
}

void DynamicLogger::set_level(LogLevel level) {
    global_level_.store(level);
}

void DynamicLogger::set_level(const std::string& component, LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    component_levels_[component] = level;
}

void DynamicLogger::set_level_from_string(const std::string& level_str) {
    LogLevel level = LogLevel::INFO;
    if (level_str == "DEBUG") level = LogLevel::DEBUG;
    else if (level_str == "WARNING") level = LogLevel::WARNING;
    else if (level_str == "ERROR") level = LogLevel::ERROR;
    else if (level_str == "CRITICAL") level = LogLevel::CRITICAL;
    set_level(level);
}

void DynamicLogger::set_min_level(LogLevel level) {
    min_level_.store(level);
}

void DynamicLogger::add_sink(std::shared_ptr<ILogSink> sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    sinks_[sink->get_name()] = sink;
}

void DynamicLogger::remove_sink(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    sinks_.erase(name);
}

void DynamicLogger::log(LogLevel level, 
                        const std::string& component,
                        const std::string& message,
                        const std::string& file,
                        int line,
                        const std::string& function,
                        std::unordered_map<std::string, std::string> context) {
    if (!initialized_.load()) {
        // Fall back to simple output if not initialized
        std::cout << message << std::endl;
        return;
    }
    
    write_message(level, component, message, file, line, function);
}

void DynamicLogger::debug(const std::string& component, const std::string& message,
                          const std::string& file, int line, const std::string& function) {
    log(LogLevel::DEBUG, component, message, file, line, function);
}

void DynamicLogger::info(const std::string& component, const std::string& message,
                         const std::string& file, int line, const std::string& function) {
    log(LogLevel::INFO, component, message, file, line, function);
}

void DynamicLogger::warning(const std::string& component, const std::string& message,
                            const std::string& file, int line, const std::string& function) {
    log(LogLevel::WARNING, component, message, file, line, function);
}

void DynamicLogger::error(const std::string& component, const std::string& message,
                          const std::string& file, int line, const std::string& function) {
    log(LogLevel::ERROR, component, message, file, line, function);
}

void DynamicLogger::critical(const std::string& component, const std::string& message,
                             const std::string& file, int line, const std::string& function) {
    log(LogLevel::CRITICAL, component, message, file, line, function);
}

void DynamicLogger::configure(const std::unordered_map<std::string, std::string>& config) {
    // Simple configuration - can be extended
    auto it = config.find("level");
    if (it != config.end()) {
        set_level_from_string(it->second);
    }
}

void DynamicLogger::register_component(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Components are registered automatically when first logged
}

void DynamicLogger::unregister_component(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    component_levels_.erase(name);
}

uint64_t DynamicLogger::get_message_count(LogLevel level) const {
    auto it = message_counts_.find(static_cast<int>(level));
    return it != message_counts_.end() ? it->second : 0;
}

uint64_t DynamicLogger::get_total_message_count() const {
    return total_messages_.load();
}

void DynamicLogger::reset_statistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    message_counts_.clear();
    total_messages_.store(0);
}

void DynamicLogger::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [name, sink] : sinks_) {
        sink->flush();
    }
    
    sinks_.clear();
    initialized_.store(false);
}

void DynamicLogger::write_message(LogLevel level, 
                                  const std::string& component,
                                  const std::string& message,
                                  const std::string& file,
                                  int line,
                                  const std::string& function) {
    LogMessage msg;
    msg.level = level;
    msg.timestamp = get_timestamp();
    msg.component = component;
    msg.file = file;
    msg.line = line;
    msg.function = function;
    msg.message = message;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Write to all sinks
    for (auto& [name, sink] : sinks_) {
        if (sink->is_enabled(level)) {
            sink->write(msg);
        }
    }
    
    // Update statistics
    message_counts_[static_cast<int>(level)]++;
    total_messages_++;
}

std::string DynamicLogger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&time), "%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

DynamicLogger::~DynamicLogger() {
    shutdown();
}

} // namespace BTQuant::Logging
