#include "utils/shm_health_monitor.hpp"
#include "utils/dynamic_logger.hpp"
#include "hotspine_layout.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

namespace BTQuant::Monitoring {

// ============================================================================
// ShmUtils Implementation
// ============================================================================

bool ShmUtils::file_exists(const std::string& path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

bool ShmUtils::is_shm_file(const std::string& path) {
    // Check if path is in /dev/shm (POSIX shared memory)
    return path.find("/dev/shm/") == 0 && file_exists(path);
}

bool ShmUtils::check_read_permission(const std::string& path) {
    return (access(path.c_str(), R_OK) == 0);
}

bool ShmUtils::check_write_permission(const std::string& path) {
    return (access(path.c_str(), W_OK) == 0);
}

size_t ShmUtils::get_file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return 0;
    }
    return static_cast<size_t>(st.st_size);
}

uint64_t ShmUtils::get_file_mtime(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(st.st_mtim.tv_sec) * 1000000ULL + st.st_mtim.tv_nsec / 1000;
}

std::string ShmUtils::read_file_header(const std::string& path, size_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }
    
    std::string header;
    header.resize(bytes);
    file.read(&header[0], bytes);
    header.resize(static_cast<size_t>(file.gcount()));
    
    return header;
}

// ============================================================================
// Helper function to get current time in microseconds
// ============================================================================

static uint64_t get_current_time_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000ULL + tv.tv_usec;
}

// ============================================================================
// HealthReport setter implementations
// ============================================================================

void ShmHealthMonitor::HealthReport::set_shm_path(const std::string& path) {
    size_t len = std::min(path.size(), shm_path.size() - 1);
    std::memcpy(shm_path.data(), path.data(), len);
    shm_path[len] = '\0';
}

void ShmHealthMonitor::HealthReport::set_error_message(const std::string& msg) {
    size_t len = std::min(msg.size(), error_message.size() - 1);
    std::memcpy(error_message.data(), msg.data(), len);
    error_message[len] = '\0';
}

// Helper function templates to convert fixed-size arrays to strings
template<size_t N>
static std::string array_to_string(const std::array<char, N>& arr) {
    return std::string(arr.data());
}

// ============================================================================
// ShmHealthMonitor Implementation
// ============================================================================

ShmHealthMonitor::ShmHealthMonitor(const std::string& shm_path, const MonitorConfig& config)
    : shm_path_(shm_path)
    , config_(config)
    , current_report_(HealthReport()) {
    
    Logging::DynamicLogger::instance().info("SHM-MONITOR", "Initializing health monitor for: " + shm_path);
    
    // Perform initial health check
    HealthReport initial_report = perform_full_check();
    update_status(initial_report);
}

ShmHealthMonitor::~ShmHealthMonitor() {
    stop_monitoring();
}

void ShmHealthMonitor::set_config(const MonitorConfig& config) {
    config_ = config;
    Logging::DynamicLogger::instance().info("SHM-MONITOR", "Configuration updated - interval: " + 
                 std::to_string(config.check_interval_us) + "us");
}

void ShmHealthMonitor::start_monitoring() {
    if (monitoring_.exchange(true)) {
        Logging::DynamicLogger::instance().warning("SHM-MONITOR", "Monitoring already active");
        return;
    }
    
    start_time_us_ = get_current_time_us();
    Logging::DynamicLogger::instance().info("SHM-MONITOR", "Starting health monitoring - interval: " +
                 std::to_string(config_.check_interval_us) + "us");
    
    monitor_thread_ = std::thread(&ShmHealthMonitor::monitor_loop, this);
}

void ShmHealthMonitor::stop_monitoring() {
    if (!monitoring_.exchange(false)) {
        return;
    }
    
    Logging::DynamicLogger::instance().info("SHM-MONITOR", "Stopping health monitoring");
    
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void ShmHealthMonitor::monitor_loop() {
    Logging::DynamicLogger::instance().debug("SHM-MONITOR", "Monitor thread started");
    
    while (monitoring_.load()) {
        HealthReport report = perform_full_check();
        update_status(report);
        
        // Check for consecutive failures
        if (report.status == HealthStatus::FAILED || 
            report.status == HealthStatus::CRITICAL) {
            consecutive_failures_++;
            
            if (consecutive_failures_.load() >= config_.max_consecutive_failures) {
                Logging::DynamicLogger::instance().error("SHM-MONITOR", "Consecutive failures threshold reached: " +
                              std::to_string(consecutive_failures_.load()));
                // Could trigger alert here
            }
        } else {
            consecutive_failures_ = 0;
        }
        
        std::this_thread::sleep_for(std::chrono::microseconds(config_.check_interval_us));
    }
    
    Logging::DynamicLogger::instance().debug("SHM-MONITOR", "Monitor thread stopped");
}

ShmHealthMonitor::HealthReport ShmHealthMonitor::perform_full_check() {
    HealthReport report;
    report.set_shm_path(shm_path_);
    report.last_check_us = get_current_time_us();
    
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    report.uptime_us = (start_time_us_ > 0) ? 
        (static_cast<uint64_t>(tv.tv_sec) * 1000000ULL + tv.tv_usec - start_time_us_) : 0;
    
    // Check 1: File existence
    report.file_exists = check_file_exists();
    if (!report.file_exists) {
        report.status = HealthStatus::CRITICAL;
        std::string error_msg = "Shared memory file does not exist: " + shm_path_;
        report.set_error_message(error_msg);
        Logging::DynamicLogger::instance().warning("SHM-MONITOR", array_to_string(report.error_message));
        return report;
    }
    
    // Check 2: Permissions
    report.permissions_ok = check_permissions();
    if (!report.permissions_ok) {
        report.status = HealthStatus::CRITICAL;
        std::string error_msg = "Insufficient permissions for shared memory: " + shm_path_;
        report.set_error_message(error_msg);
        Logging::DynamicLogger::instance().warning("SHM-MONITOR", array_to_string(report.error_message));
        return report;
    }
    
    // Check 3: Header integrity and buffer info
    size_t capacity = 0, used = 0;
    bool header_valid = check_header_integrity(capacity, used);
    if (!header_valid) {
        report.status = HealthStatus::CRITICAL;
        report.set_error_message("Invalid or corrupted shared memory header");
        Logging::DynamicLogger::instance().warning("SHM-MONITOR", array_to_string(report.error_message));
        return report;
    }
    
    report.has_valid_header = true;
    report.buffer_capacity = capacity;
    report.buffer_used = used;
    
    // All checks passed
    report.status = HealthStatus::HEALTHY;
    report.set_error_message("");
    
    return report;
}

bool ShmHealthMonitor::check_file_exists() {
    return ShmUtils::file_exists(shm_path_);
}

bool ShmHealthMonitor::check_permissions() {
    return ShmUtils::check_read_permission(shm_path_) && 
           ShmUtils::check_write_permission(shm_path_);
}

bool ShmHealthMonitor::check_header_integrity(size_t& capacity, size_t& used) {
    // Try to open and read the shared memory header
    int fd = open(shm_path_.c_str(), O_RDWR);
    if (fd < 0) {
        Logging::DynamicLogger::instance().debug("SHM-MONITOR", "Failed to open SHM for header check: " + 
                              std::string(strerror(errno)));
        return false;
    }
    
    // Read header (first 256 bytes should be sufficient)
    char header[256];
    ssize_t bytes_read = read(fd, header, sizeof(header));
    close(fd);
    
    if (bytes_read < static_cast<ssize_t>(sizeof(uint32_t) * 3)) {
        Logging::DynamicLogger::instance().debug("SHM-MONITOR", "Incomplete header read");
        return false;
    }
    
    // Parse header (format depends on hotspine_layout.hpp)
    // Assuming first 4 bytes are magic number, next 4 bytes are version,
    // next 8 bytes are capacity, next 8 bytes are used
    uint32_t magic = *reinterpret_cast<uint32_t*>(header);
    uint32_t version = *reinterpret_cast<uint32_t*>(header + 4);
    uint64_t cap = *reinterpret_cast<uint64_t*>(header + 8);
    uint64_t usd = *reinterpret_cast<uint64_t*>(header + 16);
    
    // Validate magic number (0x42545155 = "BTQU" in little endian)
    if (magic != HotSpine::SHM_MAGIC) {
        Logging::DynamicLogger::instance().debug("SHM-MONITOR", "Invalid magic number: 0x" + 
                              std::to_string(magic) + " (expected 0x" + 
                              std::to_string(HotSpine::SHM_MAGIC) + ")");
        return false;
    }
    
    // Validate version
    const uint32_t EXPECTED_VERSION = 2;
    if (version != EXPECTED_VERSION) {
        Logging::DynamicLogger::instance().debug("SHM-MONITOR", "Invalid version: " + std::to_string(version));
        return false;
    }
    
    capacity = static_cast<size_t>(cap);
    used = static_cast<size_t>(usd);
    
    return true;
}

void ShmHealthMonitor::update_status(const HealthReport& report) {
    std::lock_guard<std::mutex> lock(report_mutex_);
    current_report_ = report;
    total_checks_++;
    
    if (report.status == HealthStatus::FAILED || 
        report.status == HealthStatus::CRITICAL) {
        failed_checks_++;
    }
}

ShmHealthMonitor::HealthReport ShmHealthMonitor::check_health() {
    return perform_full_check();
}

} // namespace BTQuant::Monitoring
