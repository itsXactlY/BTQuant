#pragma once

#include <string>
#include <vector>
#include <memory>

namespace BTQuant {
namespace System {

// ============================================================================
// System Optimizer
// ============================================================================

class SystemOptimizer {
public:
    struct OptimizationResult {
        bool success;
        std::string message;
        double performance_improvement;
        std::vector<std::string> optimizations_applied;
    };

    SystemOptimizer();
    ~SystemOptimizer();

    // Optimization methods
    OptimizationResult optimize();
    OptimizationResult optimize_for_performance();
    OptimizationResult optimize_for_memory();
    OptimizationResult optimize_for_gpu();
    
    // System information
    std::string get_cpu_info() const;
    std::string get_gpu_info() const;
    std::string get_memory_info() const;
    std::string get_system_info() const;
    
    // Performance monitoring
    void set_performance_target(double target_fps);
    double get_current_fps() const;
    double get_frame_time_ms() const;
    
    // Optimization settings
    void enable_auto_optimization(bool enable);
    bool is_auto_optimization_enabled() const;

private:
    bool auto_optimization_enabled_ = false;
    double target_fps_ = 60.0;
    
    // System information
    std::string cpu_info_;
    std::string gpu_info_;
    std::vector<std::string> optimizations_applied_;
    
    // System detection
    void detect_system_capabilities();
    void apply_optimizations();
    std::vector<std::string> get_available_optimizations() const;
};

} // namespace System
} // namespace BTQuant
