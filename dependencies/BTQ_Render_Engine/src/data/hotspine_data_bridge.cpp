

#include <fcntl.h>
#include <pthread.h>  // For thread priority
#include <sched.h>    // For real-time scheduling
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <iostream>
#include <print>

#include "../../include/structured_logger.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"

namespace BTQuant {
}  // namespace BTQuant
