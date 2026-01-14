#!/bin/bash

# BTQuant Comprehensive Test Runner
# 
# Runs all test suites and generates comprehensive test reports
# 
# Test Suites:
# - Comprehensive Test Suite (Unit & Integration)
# - Performance Validation Suite
# - Functional Testing Suite
# - Integration Testing Suite
# - Stress Testing Suite
# - Professional Standards Validation
# - Build System Testing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0
SKIPPED_SUITES=0

# Global test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Logging
MASTER_LOG_FILE="comprehensive_test_results.log"
echo "BTQuant Comprehensive Test Run - $(date)" > "$MASTER_LOG_FILE"
echo "========================================" >> "$MASTER_LOG_FILE"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$MASTER_LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    echo "[PASS] $1" >> "$MASTER_LOG_FILE"
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    echo "[FAIL] $1" >> "$MASTER_LOG_FILE"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    echo "[SKIP] $1" >> "$MASTER_LOG_FILE"
}

run_test_suite() {
    local suite_name="$1"
    local suite_command="$2"
    local timeout_seconds="$3"
    
    ((TOTAL_SUITES++))
    echo -e "\n${CYAN}=== Running $suite_name ===${NC}"
    echo "=== Running $suite_name ===" >> "$MASTER_LOG_FILE"
    
    local start_time=$(date +%s)
    
    if timeout "$timeout_seconds" bash -c "$suite_command" >> "$MASTER_LOG_FILE" 2>&1; then
        log_success "$suite_name completed successfully"
        ((PASSED_SUITES++))
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_failure "$suite_name timed out after ${timeout_seconds}s"
        else
            log_failure "$suite_name failed with exit code $exit_code"
        fi
        ((FAILED_SUITES++))
        return 1
    fi
}

# ============================================================================
# Build Test Suites
# ============================================================================

build_test_suites() {
    log_info "Building test suites..."
    
    # Create build directory
    BUILD_DIR="test_build_$(date +%s)"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Build all test executables
    local build_success=true
    
    # Comprehensive Test Suite
    log_info "Building comprehensive test suite..."
    if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
           ../tests/comprehensive_test_suite.cpp \
           -o comprehensive_test_suite \
           -pthread >> "$MASTER_LOG_FILE" 2>&1; then
        log_success "Comprehensive test suite built"
    else
        log_failure "Comprehensive test suite build failed"
        build_success=false
    fi
    
    # Performance Validation Suite
    log_info "Building performance validation suite..."
    if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
           ../tests/performance_validation.cpp \
           -o performance_validation \
           -pthread >> "$MASTER_LOG_FILE" 2>&1; then
        log_success "Performance validation suite built"
    else
        log_failure "Performance validation suite build failed"
        build_success=false
    fi
    
    # Functional Testing Suite
    log_info "Building functional testing suite..."
    if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
           ../tests/functional_tests.cpp \
           -o functional_tests \
           -pthread >> "$MASTER_LOG_FILE" 2>&1; then
        log_success "Functional testing suite built"
    else
        log_failure "Functional testing suite build failed"
        build_success=false
    fi
    
    # Integration Testing Suite
    log_info "Building integration testing suite..."
    if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
           ../tests/integration_tests.cpp \
           -o integration_tests \
           -pthread >> "$MASTER_LOG_FILE" 2>&1; then
        log_success "Integration testing suite built"
    else
        log_failure "Integration test suite build failed"
        build_success=false
    fi
    
    # Stress Testing Suite
    log_info "Building stress testing suite..."
    if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
           ../tests/stress_tests.cpp \
           -o stress_tests \
           -pthread >> "$MASTER_LOG_FILE" 2>&1; then
        log_success "Stress testing suite built"
    else
        log_failure "Stress testing suite build failed"
        build_success=false
    fi
    
    # Professional Standards Suite
    log_info "Building professional standards validation..."
    if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
           ../tests/professional_standards.cpp \
           -o professional_standards \
           -pthread >> "$MASTER_LOG_FILE" 2>&1; then
        log_success "Professional standards validation built"
    else
        log_failure "Professional standards validation build failed"
        build_success=false
    fi
    
    if [ "$build_success" = false ]; then
        log_failure "Some test suites failed to build"
        cd ..
        rm -rf "$BUILD_DIR"
        return 1
    fi
    
    log_success "All test suites built successfully"
    return 0
}

# ============================================================================
# Run Individual Test Suites
# ============================================================================

run_comprehensive_tests() {
    run_test_suite "Comprehensive Test Suite" "./comprehensive_test_suite" 300
}

run_performance_tests() {
    run_test_suite "Performance Validation Suite" "./performance_validation" 600
}

run_functional_tests() {
    run_test_suite "Functional Testing Suite" "./functional_tests" 300
}

run_integration_tests() {
    run_test_suite "Integration Testing Suite" "./integration_tests" 450
}

run_stress_tests() {
    run_test_suite "Stress Testing Suite" "./stress_tests" 900
}

run_professional_standards_tests() {
    run_test_suite "Professional Standards Validation" "./professional_standards" 300
}

run_build_system_tests() {
    run_test_suite "Build System Testing" "../test_build_system.sh" 1200
}

# ============================================================================
# Generate Comprehensive Report
# ============================================================================

generate_comprehensive_report() {
    log_info "Generating comprehensive test report..."
    
    # Collect results from individual test reports
    local all_reports=""
    
    # Check for individual test reports
    if [ -f "test_report.json" ]; then
        all_reports="$all_reports test_report.json"
    fi
    
    if [ -f "performance_report.json" ]; then
        all_reports="$all_reports performance_report.json"
    fi
    
    if [ -f "functional_test_report.json" ]; then
        all_reports="$all_reports functional_test_report.json"
    fi
    
    if [ -f "integration_test_report.json" ]; then
        all_reports="$all_reports integration_test_report.json"
    fi
    
    if [ -f "stress_test_report.json" ]; then
        all_reports="$all_reports stress_test_report.json"
    fi
    
    if [ -f "professional_standards_report.json" ]; then
        all_reports="$all_reports professional_standards_report.json"
    fi
    
    if [ -f "build_test_report.json" ]; then
        all_reports="$all_reports build_test_report.json"
    fi
    
    # Create comprehensive JSON report
    cat > "comprehensive_test_report.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "test_run": "BTQuant Comprehensive Test Suite",
  "total_suites": $TOTAL_SUITES,
  "passed_suites": $PASSED_SUITES,
  "failed_suites": $FAILED_SUITES,
  "skipped_suites": $SKIPPED_SUITES,
  "suite_success_rate": $(echo "scale=2; $PASSED_SUITES * 100 / $TOTAL_SUITES" | bc -l 2>/dev/null || echo "0"),
  "total_tests": $TOTAL_TESTS,
  "passed_tests": $PASSED_TESTS,
  "failed_tests": $FAILED_TESTS,
  "skipped_tests": $SKIPPED_TESTS,
  "test_success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0"),
  "individual_reports": [
EOF
    
    # Add individual report summaries
    local first=true
    for report in $all_reports; do
        if [ -f "$report" ]; then
            if [ "$first" = false ]; then
                echo "," >> "comprehensive_test_report.json"
            fi
            first=false
            
            # Extract basic info from each report
            local suite_name=$(grep -o '"test_suite": *"[^"]*"' "$report" | cut -d'"' -f4)
            local total_tests=$(grep -o '"total_tests": *[0-9]*' "$report" | grep -o '[0-9]*' | head -1)
            local passed=$(grep -o '"passed": *[0-9]*' "$report" | grep -o '[0-9]*' | head -1)
            local failed=$(grep -o '"failed": *[0-9]*' "$report" | grep -o '[0-9]*' | head -1)
            
            cat >> "comprehensive_test_report.json" << EOF
    {
      "suite": "$suite_name",
      "report_file": "$report",
      "total_tests": ${total_tests:-0},
      "passed": ${passed:-0},
      "failed": ${failed:-0},
      "success_rate": $(echo "scale=2; ${passed:-0} * 100 / ${total_tests:-1}" | bc -l 2>/dev/null || echo "0")
    }
EOF
        fi
    done
    
    cat >> "comprehensive_test_report.json" << EOF
  ],
  "overall_status": "$([ $FAILED_SUITES -eq 0 ] && echo "PASSED" || echo "FAILED")",
  "master_log": "$MASTER_LOG_FILE"
}
EOF
    
    log_success "Comprehensive test report generated: comprehensive_test_report.json"
}

# ============================================================================
# Main Test Execution
# ============================================================================

main() {
    echo -e "${PURPLE}BTQuant Comprehensive Test Runner${NC}"
    echo -e "${PURPLE}=================================${NC}"
    echo ""
    
    # Change to the BTQ_Render_Engine directory
    cd "$(dirname "$0")"
    
    # Build test suites first
    if ! build_test_suites; then
        log_failure "Failed to build test suites. Aborting."
        exit 1
    fi
    
    # Run all test suites
    run_comprehensive_tests
    run_performance_tests
    run_functional_tests
    run_integration_tests
    run_stress_tests
    run_professional_standards_tests
    run_build_system_tests
    
    # Clean up build directory
    cd ..
    rm -rf "$BUILD_DIR"
    
    # Generate comprehensive report
    generate_comprehensive_report
    
    # Final summary
    echo ""
    echo -e "${CYAN}=== Comprehensive Test Summary ===${NC}"
    echo "Test Suites Run: $TOTAL_SUITES"
    echo -e "Suites Passed: ${GREEN}$PASSED_SUITES${NC}"
    echo -e "Suites Failed: ${RED}$FAILED_SUITES${NC}"
    echo -e "Suites Skipped: ${YELLOW}$SKIPPED_SUITES${NC}"
    
    local suite_success_rate=$(echo "scale=1; $PASSED_SUITES * 100 / $TOTAL_SUITES" | bc -l 2>/dev/null || echo "0")
    echo "Suite Success Rate: $suite_success_rate%"
    
    # Overall assessment
    if [ $FAILED_SUITES -eq 0 ]; then
        echo -e "\n${GREEN}✓ COMPREHENSIVE TESTING: ALL SUITES PASSED${NC}"
        echo "The BTQuant Advanced Vulkan Dashboard meets professional standards."
        echo "Ready for production deployment."
    elif [ $FAILED_SUITES -le 2 ]; then
        echo -e "\n${YELLOW}⚠ COMPREHENSIVE TESTING: PASSED WITH WARNINGS${NC}"
        echo "The system is functional but requires attention to failed test suites."
    else
        echo -e "\n${RED}✗ COMPREHENSIVE TESTING: CRITICAL ISSUES DETECTED${NC}"
        echo "The system requires significant fixes before deployment."
        echo "Review individual test reports for detailed failure analysis."
    fi
    
    echo ""
    echo "Detailed results:"
    echo "  Master log: $MASTER_LOG_FILE"
    echo "  Comprehensive report: comprehensive_test_report.json"
    echo "  Individual suite reports: [test_report.json, performance_report.json, etc.]"
    
    # Return appropriate exit code
    [ $FAILED_SUITES -eq 0 ]
}

# Run main function
main "$@"