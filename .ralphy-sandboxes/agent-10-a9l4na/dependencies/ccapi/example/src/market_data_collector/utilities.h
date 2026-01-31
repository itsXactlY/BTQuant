#pragma once
#include <string>
#include <chrono>
#include <ctime>
#include <cstdio>

inline std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::tm tm = *std::localtime(&now_time);
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
    
    char ms_buffer[10];
    snprintf(ms_buffer, sizeof(ms_buffer), "%03d", static_cast<int>(now_ms.count()));
    
    return std::string(buffer) + "." + ms_buffer;
}
