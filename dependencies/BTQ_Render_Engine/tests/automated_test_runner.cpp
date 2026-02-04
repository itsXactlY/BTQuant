/*
 * Automated Test Runner for BTQ Render Engine
 * Implements CI setup for testing on multiple platforms
 * Supports Linux, Windows, and macOS environments
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <algorithm>
#include <map>
#include <functional>

// Platform detection macros
#ifdef _WIN32
    #define PLATFORM_NAME "Windows"
    #include <windows.h>
#elif __APPLE__
    #define PLATFORM_NAME "macOS"
    #include <TargetConditionals.h>
#else
    #define PLATFORM_NAME "Linux"
#endif

// Test result structure
struct TestResult {
    std::string testName;
    bool passed;
    double durationMs;
    std::string errorMessage;
    
    TestResult(const std::string& name, bool p, double dur, const std::string& err = "") 
        : testName(name), passed(p), durationMs(dur), errorMessage(err) {}
};

class AutomatedTestRunner {
private:
    std::vector<std::string> testExecutables;
    std::vector<TestResult> results;
    int totalTests = 0;
    int passedTests = 0;
    int failedTests = 0;

public:
    AutomatedTestRunner() {
        initializeTestSuite();
    }

    void initializeTestSuite() {
        // Define the test executables to run across platforms
        testExecutables = {
            "./test_frame_time_basic",
            "./test_cpu_profiler",
            "./test_memory_tracker",
            "./test_lockfree_queue",
            "./test_lockfree_structures",
            "./test_frame_pacer",
            "./test_auto_quality",
            "./test_enhanced_auto_quality",
            "./test_enhanced_quality",
            "./test_panel_culling",
            "./test_performance_integration",
            "./test_performance_regression",
            "./test_profile_settings",
            "./test_split_volume_display",
            "./test_step_profile_rendering",
            "./test_task_scheduler",
            "./test_volume_analysis_type",
            "./test_footprint_zoom",
            "./test_improved_footprint_zoom",
            "./test_incremental_updater",
            "./test_loading_states",
            "./test_imgui_optimizer",
            "./cluster_engine_tests",
            "./data_pipeline_tests",
            "./simple_exchange_aggregator_test",
            "./stress_tests",
            "./test_mock_data_generator",
            "./vwap_tests",
            "./volume_calculation_tests",
            "./test_frame_time_graph_advanced",
            "./visual_regression_tests",
            "./test_exchange_aggregator",
            "./performance_benchmarks"
        };

        // On Windows, executables have .exe extension
#ifdef _WIN32
        for (auto& exe : testExecutables) {
            if (exe.substr(0, 2) == "./") {
                exe += ".exe";
            }
        }
#endif
    }

    bool runSingleTest(const std::string& executable) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
#ifdef _WIN32
        int result = system(executable.c_str());
#else
        int result = system(executable.c_str());
#endif
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        bool passed = (result == 0);
        std::string testName = executable.substr(2); // Remove "./" prefix
        
        results.emplace_back(testName, passed, duration);
        
        if (passed) {
            passedTests++;
            std::cout << "[PASS] " << testName << " (" << duration << " ms)" << std::endl;
        } else {
            failedTests++;
            std::cout << "[FAIL] " << testName << " (" << duration << " ms)" << std::endl;
        }
        
        return passed;
    }

    void runAllTests() {
        std::cout << "BTQ Render Engine Automated Test Runner" << std::endl;
        std::cout << "Platform: " << PLATFORM_NAME << std::endl;
        std::cout << "Running " << testExecutables.size() << " tests..." << std::endl;
        std::cout << "=========================================" << std::endl;

        totalTests = testExecutables.size();
        passedTests = 0;
        failedTests = 0;
        results.clear();

        for (const auto& test : testExecutables) {
            std::cout << "Running: " << test << std::endl;
            
            // Check if test executable exists before running
            std::string checkCmd;
#ifdef _WIN32
            checkCmd = "if exist \"" + test + "\" (echo exists) else (echo missing)";
#else
            checkCmd = "test -f " + test + " && echo exists || echo missing";
#endif
            
            std::string checkResult = getCommandOutput(checkCmd);
            if (checkResult.find("missing") != std::string::npos) {
                std::cout << "[SKIP] " << test.substr(2) << " (executable not found)" << std::endl;
                continue;
            }
            
            runSingleTest(test);
        }
    }

    std::string getCommandOutput(const std::string& cmd) {
        std::string result;
#ifndef _WIN32
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return "popen failed!";
        
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);
#else
        // Windows fallback - just return "exists" for simplicity
        result = "exists";
#endif
        return result;
    }

    void generateReport() {
        std::cout << "\n=========================================" << std::endl;
        std::cout << "TEST RESULTS SUMMARY" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "Platform: " << PLATFORM_NAME << std::endl;
        std::cout << "Total Tests: " << totalTests << std::endl;
        std::cout << "Passed: " << passedTests << std::endl;
        std::cout << "Failed: " << failedTests << std::endl;
        std::cout << "Success Rate: " << (totalTests > 0 ? (100.0 * passedTests / totalTests) : 0.0) << "%" << std::endl;
        
        if (failedTests > 0) {
            std::cout << "\nFAILED TESTS:" << std::endl;
            for (const auto& result : results) {
                if (!result.passed) {
                    std::cout << "  - " << result.testName << std::endl;
                }
            }
        }
        
        std::cout << "\nDetailed Results:" << std::endl;
        for (const auto& result : results) {
            std::cout << "  " << (result.passed ? "[PASS]" : "[FAIL]") 
                      << " " << result.testName 
                      << " (" << result.durationMs << " ms)" << std::endl;
        }
    }

    int getExitCode() const {
        return failedTests > 0 ? 1 : 0;
    }
    
    bool hasFailures() const {
        return failedTests > 0;
    }
};

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;  // Suppress unused parameter warnings
    std::cout << "Starting BTQ Render Engine Automated Test Suite" << std::endl;
    
    AutomatedTestRunner runner;
    runner.runAllTests();
    runner.generateReport();
    
    std::cout << "\nTest suite completed." << std::endl;
    
    return runner.getExitCode();
}