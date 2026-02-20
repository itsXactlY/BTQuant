#pragma once

/**
 * @file thread_affinity.hpp
 * @brief CPU Core Pinning for Ultra-Low-Latency Trading Terminal
 * 
 * This implementation provides platform-specific thread affinity control
 * to minimize context switching and cache misses for critical threads.
 * 
 * Key features:
 * - Pin market data thread to dedicated core
 * - Pin rendering thread to separate core
 * - Support for hyperthreading awareness
 * - NUMA awareness for multi-socket systems
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <processthreadsapi.h>
#else
    #include <pthread.h>
    #include <sched.h>
    #include <unistd.h>
    #include <sys/sysinfo.h>
#endif

namespace btq {
namespace threading {

/**
 * @brief CPU topology information for optimal thread placement
 */
struct CPUTopology {
    size_t num_cores{0};              // Physical cores
    size_t num_threads{0};            // Logical threads (including hyperthreading)
    size_t num_sockets{0};            // CPU sockets
    bool has_hyperthreading{false};   // Whether hyperthreading is available
    
    static CPUTopology detect() {
        CPUTopology topo;
        
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        topo.num_threads = sysinfo.dwNumberOfProcessors;
        
        // Try to get core count (Windows 7+)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformation(nullptr, &buffer_size);
        
        if (buffer_size > 0) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            GetLogicalProcessorInformation(buffer.data(), &buffer_size);
            
            for (const auto& info : buffer) {
                if (info.Relationship == RelationProcessorCore) {
                    topo.num_cores++;
                } else if (info.Relationship == RelationProcessorPackage) {
                    topo.num_sockets++;
                }
            }
        }
        
        topo.has_hyperthreading = (topo.num_threads > topo.num_cores);
        
#else
        topo.num_threads = std::thread::hardware_concurrency();
        topo.num_cores = sysconf(_SC_NPROCESSORS_ONLN);
        
        // Try to detect hyperthreading
        FILE* fp = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
        if (fp) {
            char buf[256];
            if (fgets(buf, sizeof(buf), fp)) {
                // If thread_siblings contains more than one CPU, hyperthreading is present
                std::string str(buf);
                topo.has_hyperthreading = (str.find('-') != std::string::npos || 
                                          str.find(',') != std::string::npos);
            }
            fclose(fp);
        }
        
        // Count sockets (simplified)
        topo.num_sockets = 1;  // Default assumption
#endif
        
        // Fallback if detection failed
        if (topo.num_cores == 0) {
            topo.num_cores = topo.num_threads / (topo.has_hyperthreading ? 2 : 1);
            if (topo.num_cores == 0) topo.num_cores = 1;
        }
        
        return topo;
    }
    
    std::string toString() const {
        return "CPUTopology: " + std::to_string(num_cores) + " cores, " +
               std::to_string(num_threads) + " threads, " +
               std::to_string(num_sockets) + " sockets, " +
               (has_hyperthreading ? "hyperthreading enabled" : "no hyperthreading");
    }
};

/**
 * @brief Thread affinity manager for pinning threads to specific CPU cores
 */
class ThreadAffinity {
public:
    /**
     * @brief Core assignment strategy for trading terminal threads
     */
    enum class CoreStrategy {
        DEDICATED_ISOLATED,    // Use isolated cores (requires kernel boot param)
        DEDICATED_PHYSICAL,    // Use physical cores only (avoid hyperthreading)
        DEDICATED_ANY,         // Use any available core
        SHARED_LOW_PRIORITY    // Share cores with other threads
    };
    
    /**
     * @brief Thread role for automatic core assignment
     */
    enum class ThreadRole {
        MARKET_DATA_INGESTION,  // Highest priority - network data processing
        ORDER_EXECUTION,        // High priority - order submission
        RENDERING,              // High priority - UI rendering
        INDICATOR_CALCULATION,  // Medium priority - technical analysis
        DATA_AGGREGATION,       // Medium priority - OHLCV building
        LOGGING,                // Low priority - logging and telemetry
        GENERAL                 // Default - no special affinity
    };
    
    /**
     * @brief Get singleton instance
     */
    static ThreadAffinity& getInstance() {
        static ThreadAffinity instance;
        return instance;
    }
    
    /**
     * @brief Initialize with detected topology
     */
    void initialize() {
        topology_ = CPUTopology::detect();
        core_assignments_.resize(topology_.num_threads, false);
        initialized_ = true;
    }
    
    /**
     * @brief Pin the current thread to a specific core
     * @param core_id The core ID to pin to (0-indexed)
     * @return true if successful
     */
    bool pinToCore(size_t core_id) {
        if (!initialized_) initialize();
        
        if (core_id >= topology_.num_threads) {
            return false;
        }
        
#ifdef _WIN32
        // Windows: SetThreadAffinityMask
        DWORD_PTR mask = 1ULL << core_id;
        HANDLE thread_handle = GetCurrentThread();
        DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);
        return (result != 0);
#else
        // Linux: pthread_setaffinity_np
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        
        pthread_t thread = pthread_self();
        int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        
        if (result == 0) {
            core_assignments_[core_id] = true;
            return true;
        }
        return false;
#endif
    }
    
    /**
     * @brief Pin a specific thread to a core
     * @param thread The thread to pin
     * @param core_id The core ID to pin to
     * @return true if successful
     */
    bool pinThreadToCore(std::thread& thread, size_t core_id) {
        if (!initialized_) initialize();
        
        if (core_id >= topology_.num_threads) {
            return false;
        }
        
#ifdef _WIN32
        DWORD_PTR mask = 1ULL << core_id;
        HANDLE thread_handle = reinterpret_cast<HANDLE>(thread.native_handle());
        DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);
        return (result != 0);
#else
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        
        int result = pthread_setaffinity_np(
            thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        
        if (result == 0) {
            core_assignments_[core_id] = true;
            return true;
        }
        return false;
#endif
    }
    
    /**
     * @brief Pin thread to multiple cores (for load balancing)
     * @param core_ids Vector of core IDs to allow
     * @return true if successful
     */
    bool pinToCores(const std::vector<size_t>& core_ids) {
        if (!initialized_) initialize();
        
#ifdef _WIN32
        DWORD_PTR mask = 0;
        for (size_t core_id : core_ids) {
            if (core_id < topology_.num_threads) {
                mask |= 1ULL << core_id;
            }
        }
        HANDLE thread_handle = GetCurrentThread();
        DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);
        return (result != 0);
#else
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        for (size_t core_id : core_ids) {
            if (core_id < topology_.num_threads) {
                CPU_SET(core_id, &cpuset);
                core_assignments_[core_id] = true;
            }
        }
        
        pthread_t thread = pthread_self();
        int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        return (result == 0);
#endif
    }
    
    /**
     * @brief Get recommended core for a thread role
     * @param role The thread role
     * @return Recommended core ID
     */
    size_t getRecommendedCore(ThreadRole role) {
        if (!initialized_) initialize();
        
        // Strategy: Assign cores in order of priority
        // Market Data: Core 0 (or first physical core)
        // Order Execution: Core 1
        // Rendering: Core 2
        // Indicators: Core 3
        // etc.
        
        size_t base_core = 0;
        
        switch (role) {
            case ThreadRole::MARKET_DATA_INGESTION:
                base_core = 0;
                break;
            case ThreadRole::ORDER_EXECUTION:
                base_core = 1;
                break;
            case ThreadRole::RENDERING:
                base_core = 2;
                break;
            case ThreadRole::INDICATOR_CALCULATION:
                base_core = 3;
                break;
            case ThreadRole::DATA_AGGREGATION:
                base_core = 4;
                break;
            case ThreadRole::LOGGING:
                base_core = topology_.num_threads - 1;  // Last core
                break;
            case ThreadRole::GENERAL:
            default:
                // Find first unassigned core
                for (size_t i = 0; i < topology_.num_threads; ++i) {
                    if (!core_assignments_[i]) {
                        base_core = i;
                        break;
                    }
                }
                break;
        }
        
        // If hyperthreading, prefer physical cores
        if (topology_.has_hyperthreading && base_core < topology_.num_cores) {
            // Physical cores are typically 0, 2, 4, ... or 0, 1, 2, 3 depending on topology
            // For simplicity, assume physical cores are the first num_cores
            return base_core % topology_.num_cores;
        }
        
        return base_core;
    }
    
    /**
     * @brief Pin current thread based on its role
     * @param role The thread role
     * @return true if successful
     */
    bool pinByRole(ThreadRole role) {
        size_t core = getRecommendedCore(role);
        return pinToCore(core);
    }
    
    /**
     * @brief Set thread priority
     * @param priority Priority level (0 = lowest, 99 = highest on Linux)
     * @return true if successful
     */
    bool setPriority(int priority) {
#ifdef _WIN32
        int win_priority;
        if (priority >= 90) win_priority = THREAD_PRIORITY_TIME_CRITICAL;
        else if (priority >= 70) win_priority = THREAD_PRIORITY_HIGHEST;
        else if (priority >= 50) win_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        else if (priority >= 30) win_priority = THREAD_PRIORITY_NORMAL;
        else if (priority >= 10) win_priority = THREAD_PRIORITY_BELOW_NORMAL;
        else win_priority = THREAD_PRIORITY_LOWEST;
        
        return SetThreadPriority(GetCurrentThread(), win_priority) != 0;
#else
        struct sched_param param;
        param.sched_priority = priority;
        return pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) == 0;
#endif
    }
    
    /**
     * @brief Set thread name for debugging
     * @param name The thread name
     */
    void setThreadName(const std::string& name) {
#ifdef _WIN32
        // Windows 10+: SetThreadDescription
        typedef HRESULT(WINAPI* SetThreadDescriptionFunc)(HANDLE, PCWSTR);
        static SetThreadDescriptionFunc setThreadDescription = nullptr;
        
        if (!setThreadDescription) {
            HMODULE hModule = GetModuleHandleA("kernel32.dll");
            if (hModule) {
                setThreadDescription = (SetThreadDescriptionFunc)GetProcAddress(
                    hModule, "SetThreadDescription");
            }
        }
        
        if (setThreadDescription) {
            std::wstring wname(name.begin(), name.end());
            setThreadDescription(GetCurrentThread(), wname.c_str());
        }
#else
        pthread_setname_np(pthread_self(), name.c_str());
#endif
    }
    
    /**
     * @brief Get the current thread's core
     * @return Current core ID, or SIZE_MAX if error
     */
    size_t getCurrentCore() const {
#ifdef _WIN32
        return GetCurrentProcessorNumber();
#else
        return sched_getcpu();
#endif
    }
    
    /**
     * @brief Get CPU topology
     */
    const CPUTopology& getTopology() const {
        return topology_;
    }
    
    /**
     * @brief Check if a core is assigned
     */
    bool isCoreAssigned(size_t core_id) const {
        if (core_id < core_assignments_.size()) {
            return core_assignments_[core_id];
        }
        return false;
    }
    
    /**
     * @brief Get number of available (unassigned) cores
     */
    size_t getAvailableCores() const {
        size_t count = 0;
        for (bool assigned : core_assignments_) {
            if (!assigned) ++count;
        }
        return count;
    }

private:
    ThreadAffinity() = default;
    ~ThreadAffinity() = default;
    
    // Non-copyable, non-movable
    ThreadAffinity(const ThreadAffinity&) = delete;
    ThreadAffinity& operator=(const ThreadAffinity&) = delete;
    
    CPUTopology topology_;
    std::vector<bool> core_assignments_;
    bool initialized_{false};
};

/**
 * @brief RAII helper for thread affinity
 */
class ThreadAffinityGuard {
public:
    explicit ThreadAffinityGuard(ThreadAffinity::ThreadRole role) {
        ThreadAffinity::getInstance().pinByRole(role);
    }
    
    explicit ThreadAffinityGuard(size_t core_id) {
        ThreadAffinity::getInstance().pinToCore(core_id);
    }
    
    ~ThreadAffinityGuard() {
        // Reset affinity to all cores
        std::vector<size_t> all_cores;
        auto& topo = ThreadAffinity::getInstance().getTopology();
        for (size_t i = 0; i < topo.num_threads; ++i) {
            all_cores.push_back(i);
        }
        ThreadAffinity::getInstance().pinToCores(all_cores);
    }
    
    // Non-copyable, non-movable
    ThreadAffinityGuard(const ThreadAffinityGuard&) = delete;
    ThreadAffinityGuard& operator=(const ThreadAffinityGuard&) = delete;
};

/**
 * @brief Convenience function to pin current thread to a core
 */
inline bool pinCurrentThreadToCore(size_t core_id) {
    return ThreadAffinity::getInstance().pinToCore(core_id);
}

/**
 * @brief Convenience function to pin current thread by role
 */
inline bool pinCurrentThreadByRole(ThreadAffinity::ThreadRole role) {
    return ThreadAffinity::getInstance().pinByRole(role);
}

/**
 * @brief Convenience function to set thread name
 */
inline void setCurrentThreadName(const std::string& name) {
    ThreadAffinity::getInstance().setThreadName(name);
}

} // namespace threading
} // namespace btq