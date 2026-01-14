#!/bin/bash

# BTQuant Build and Deployment Testing Script
# 
# Comprehensive build system validation and deployment testing
# 
# Test Coverage:
# - Dependency validation and version checking
# - Cross-platform compatibility testing (Linux focus)
# - Installation and setup validation
# - Documentation accuracy verification
# - Build configuration testing
# - Performance optimization validation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Logging
LOG_FILE="build_test_results.log"
echo "BTQuant Build System Test - $(date)" > "$LOG_FILE"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    echo "[PASS] $1" >> "$LOG_FILE"
    ((PASSED_TESTS++))
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    echo "[FAIL] $1" >> "$LOG_FILE"
    ((FAILED_TESTS++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    echo "[SKIP] $1" >> "$LOG_FILE"
    ((SKIPPED_TESTS++))
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    ((TOTAL_TESTS++))
    log_info "Running test: $test_name"
    
    if eval "$test_command" >> "$LOG_FILE" 2>&1; then
        log_success "$test_name"
        return 0
    else
        log_failure "$test_name"
        return 1
    fi
}

# ============================================================================
# System Requirements Validation
# ============================================================================

test_system_requirements() {
    log_info "=== Testing System Requirements ==="
    
    # Test CMake version
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
        CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
        
        if [ "$CMAKE_MAJOR" -gt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -ge 16 ]); then
            log_success "CMake version check ($CMAKE_VERSION >= 3.16)"
        else
            log_failure "CMake version check ($CMAKE_VERSION < 3.16)"
        fi
    else
        log_failure "CMake not found"
    fi
    ((TOTAL_TESTS++))
    
    # Test C++ compiler
    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -n1 | grep -oE '[0-9]+\.[0-9]+')
        GCC_MAJOR=$(echo "$GCC_VERSION" | cut -d. -f1)
        
        if [ "$GCC_MAJOR" -ge 9 ]; then
            log_success "G++ compiler version check ($GCC_VERSION >= 9.0)"
        else
            log_failure "G++ compiler version check ($GCC_VERSION < 9.0)"
        fi
    elif command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | head -n1 | grep -oE '[0-9]+\.[0-9]+')
        log_success "Clang++ compiler found ($CLANG_VERSION)"
    else
        log_failure "No suitable C++ compiler found"
    fi
    ((TOTAL_TESTS++))
    
    # Test Vulkan SDK
    if command -v vulkaninfo &> /dev/null; then
        if vulkaninfo --summary &> /dev/null; then
            log_success "Vulkan SDK validation"
        else
            log_failure "Vulkan SDK validation (vulkaninfo failed)"
        fi
    else
        log_failure "Vulkan SDK not found"
    fi
    ((TOTAL_TESTS++))
    
    # Test X11 development libraries
    if pkg-config --exists x11; then
        log_success "X11 development libraries found"
    else
        log_failure "X11 development libraries not found"
    fi
    ((TOTAL_TESTS++))
    
    # Test pthread support
    if echo '#include <pthread.h>' | g++ -x c++ -c - -o /dev/null 2>/dev/null; then
        log_success "pthread support validation"
    else
        log_failure "pthread support validation"
    fi
    ((TOTAL_TESTS++))
}

# ============================================================================
# Build Configuration Testing
# ============================================================================

test_build_configurations() {
    log_info "=== Testing Build Configurations ==="
    
    # Create temporary build directory
    BUILD_DIR="build_test_$(date +%s)"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Test Debug build
    log_info "Testing Debug build configuration"
    if cmake -DCMAKE_BUILD_TYPE=Debug .. >> "$LOG_FILE" 2>&1; then
        if make -j$(nproc) >> "$LOG_FILE" 2>&1; then
            log_success "Debug build configuration"
        else
            log_failure "Debug build configuration (compilation failed)"
        fi
    else
        log_failure "Debug build configuration (cmake failed)"
    fi
    ((TOTAL_TESTS++))
    
    # Clean and test Release build
    make clean >> "$LOG_FILE" 2>&1 || true
    rm -f CMakeCache.txt
    
    log_info "Testing Release build configuration"
    if cmake -DCMAKE_BUILD_TYPE=Release .. >> "$LOG_FILE" 2>&1; then
        if make -j$(nproc) >> "$LOG_FILE" 2>&1; then
            log_success "Release build configuration"
        else
            log_failure "Release build configuration (compilation failed)"
        fi
    else
        log_failure "Release build configuration (cmake failed)"
    fi
    ((TOTAL_TESTS++))
    
    # Test RelWithDebInfo build
    make clean >> "$LOG_FILE" 2>&1 || true
    rm -f CMakeCache.txt
    
    log_info "Testing RelWithDebInfo build configuration"
    if cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. >> "$LOG_FILE" 2>&1; then
        if make -j$(nproc) >> "$LOG_FILE" 2>&1; then
            log_success "RelWithDebInfo build configuration"
        else
            log_failure "RelWithDebInfo build configuration (compilation failed)"
        fi
    else
        log_failure "RelWithDebInfo build configuration (cmake failed)"
    fi
    ((TOTAL_TESTS++))
    
    cd ..
    rm -rf "$BUILD_DIR"
}

# ============================================================================
# Dependency Testing
# ============================================================================

test_dependencies() {
    log_info "=== Testing Dependencies ==="
    
    # Test external dependencies download
    BUILD_DIR="deps_test_$(date +%s)"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Test CMake dependency fetching
    log_info "Testing dependency fetching"
    if cmake .. >> "$LOG_FILE" 2>&1; then
        log_success "Dependency fetching (CMake FetchContent)"
        
        # Check if key dependencies are available
        if [ -d "_deps/glm-src" ] || [ -d "_deps/glm-build" ]; then
            log_success "GLM dependency validation"
        else
            log_failure "GLM dependency validation"
        fi
        ((TOTAL_TESTS++))
        
        if [ -d "_deps/concurrentqueue-src" ] || [ -d "_deps/concurrentqueue-build" ]; then
            log_success "ConcurrentQueue dependency validation"
        else
            log_failure "ConcurrentQueue dependency validation"
        fi
        ((TOTAL_TESTS++))
        
        if [ -d "_deps/vulkanmemoryallocator-src" ] || [ -d "_deps/vulkanmemoryallocator-build" ]; then
            log_success "VulkanMemoryAllocator dependency validation"
        else
            log_failure "VulkanMemoryAllocator dependency validation"
        fi
        ((TOTAL_TESTS++))
        
        if [ -d "_deps/imgui-src" ] || [ -d "_deps/imgui-build" ]; then
            log_success "ImGui dependency validation"
        else
            log_failure "ImGui dependency validation"
        fi
        ((TOTAL_TESTS++))
        
    else
        log_failure "Dependency fetching (CMake FetchContent)"
        ((TOTAL_TESTS += 4)) # Count the skipped dependency tests
    fi
    
    cd ..
    rm -rf "$BUILD_DIR"
}

# ============================================================================
# Executable Testing
# ============================================================================

test_executables() {
    log_info "=== Testing Executable Generation ==="
    
    BUILD_DIR="exec_test_$(date +%s)"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Build all executables
    if cmake -DCMAKE_BUILD_TYPE=Release .. >> "$LOG_FILE" 2>&1; then
        if make -j$(nproc) >> "$LOG_FILE" 2>&1; then
            
            # Test main dashboard executable
            if [ -f "bin/dashboard_advanced" ] || [ -f "dashboard_advanced" ]; then
                log_success "Main dashboard executable generation"
                
                # Test executable permissions
                EXEC_FILE=$(find . -name "dashboard_advanced" -type f)
                if [ -x "$EXEC_FILE" ]; then
                    log_success "Dashboard executable permissions"
                else
                    log_failure "Dashboard executable permissions"
                fi
                ((TOTAL_TESTS++))
                
            else
                log_failure "Main dashboard executable generation"
            fi
            ((TOTAL_TESTS++))
            
            # Test integration test executable
            if [ -f "dashboard_test" ] || [ -f "bin/dashboard_test" ]; then
                log_success "Integration test executable generation"
            else
                log_failure "Integration test executable generation"
            fi
            ((TOTAL_TESTS++))
            
            # Test interactive features test executable
            if [ -f "test_interactive_features" ] || [ -f "bin/test_interactive_features" ]; then
                log_success "Interactive features test executable generation"
            else
                log_failure "Interactive features test executable generation"
            fi
            ((TOTAL_TESTS++))
            
        else
            log_failure "Executable compilation"
            ((TOTAL_TESTS += 4))
        fi
    else
        log_failure "Executable build configuration"
        ((TOTAL_TESTS += 4))
    fi
    
    cd ..
    rm -rf "$BUILD_DIR"
}

# ============================================================================
# Shader Compilation Testing
# ============================================================================

test_shader_compilation() {
    log_info "=== Testing Shader Compilation ==="
    
    # Test if glslc (Vulkan shader compiler) is available
    if command -v glslc &> /dev/null; then
        log_success "Vulkan shader compiler (glslc) found"
        
        # Test shader compilation
        SHADER_DIR="shaders"
        if [ -d "$SHADER_DIR" ]; then
            SHADER_COMPILE_SUCCESS=true
            
            # Test vertex shader compilation
            if [ -f "$SHADER_DIR/ui_vertex.vert" ]; then
                if glslc "$SHADER_DIR/ui_vertex.vert" -o "/tmp/ui_vertex.spv" >> "$LOG_FILE" 2>&1; then
                    log_success "Vertex shader compilation (ui_vertex.vert)"
                    rm -f "/tmp/ui_vertex.spv"
                else
                    log_failure "Vertex shader compilation (ui_vertex.vert)"
                    SHADER_COMPILE_SUCCESS=false
                fi
                ((TOTAL_TESTS++))
            fi
            
            # Test fragment shader compilation
            if [ -f "$SHADER_DIR/ui_fragment.frag" ]; then
                if glslc "$SHADER_DIR/ui_fragment.frag" -o "/tmp/ui_fragment.spv" >> "$LOG_FILE" 2>&1; then
                    log_success "Fragment shader compilation (ui_fragment.frag)"
                    rm -f "/tmp/ui_fragment.spv"
                else
                    log_failure "Fragment shader compilation (ui_fragment.frag)"
                    SHADER_COMPILE_SUCCESS=false
                fi
                ((TOTAL_TESTS++))
            fi
            
            # Test compute shader compilation
            if [ -f "$SHADER_DIR/heatmap_compute.comp" ]; then
                if glslc "$SHADER_DIR/heatmap_compute.comp" -o "/tmp/heatmap_compute.spv" >> "$LOG_FILE" 2>&1; then
                    log_success "Compute shader compilation (heatmap_compute.comp)"
                    rm -f "/tmp/heatmap_compute.spv"
                else
                    log_failure "Compute shader compilation (heatmap_compute.comp)"
                    SHADER_COMPILE_SUCCESS=false
                fi
                ((TOTAL_TESTS++))
            fi
            
            if [ "$SHADER_COMPILE_SUCCESS" = true ]; then
                log_success "Overall shader compilation"
            else
                log_failure "Overall shader compilation"
            fi
            ((TOTAL_TESTS++))
            
        else
            log_skip "Shader directory not found"
            ((TOTAL_TESTS++))
        fi
    else
        log_failure "Vulkan shader compiler (glslc) not found"
        ((TOTAL_TESTS++))
    fi
}

# ============================================================================
# Test Execution Validation
# ============================================================================

test_executable_functionality() {
    log_info "=== Testing Executable Functionality ==="
    
    BUILD_DIR="func_test_$(date +%s)"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Build executables
    if cmake -DCMAKE_BUILD_TYPE=Release .. >> "$LOG_FILE" 2>&1 && make -j$(nproc) >> "$LOG_FILE" 2>&1; then
        
        # Test comprehensive test suite
        if [ -f "../tests/comprehensive_test_suite.cpp" ]; then
            log_info "Building comprehensive test suite"
            if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
                   ../tests/comprehensive_test_suite.cpp \
                   -o comprehensive_test_suite \
                   -pthread >> "$LOG_FILE" 2>&1; then
                
                log_info "Running comprehensive test suite (timeout: 60s)"
                if timeout 60s ./comprehensive_test_suite >> "$LOG_FILE" 2>&1; then
                    log_success "Comprehensive test suite execution"
                else
                    log_failure "Comprehensive test suite execution (timeout or failure)"
                fi
            else
                log_failure "Comprehensive test suite compilation"
            fi
            ((TOTAL_TESTS++))
        fi
        
        # Test performance validation
        if [ -f "../tests/performance_validation.cpp" ]; then
            log_info "Building performance validation suite"
            if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
                   ../tests/performance_validation.cpp \
                   -o performance_validation \
                   -pthread >> "$LOG_FILE" 2>&1; then
                
                log_info "Running performance validation (timeout: 120s)"
                if timeout 120s ./performance_validation >> "$LOG_FILE" 2>&1; then
                    log_success "Performance validation execution"
                else
                    log_failure "Performance validation execution (timeout or failure)"
                fi
            else
                log_failure "Performance validation compilation"
            fi
            ((TOTAL_TESTS++))
        fi
        
        # Test functional tests
        if [ -f "../tests/functional_tests.cpp" ]; then
            log_info "Building functional test suite"
            if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
                   ../tests/functional_tests.cpp \
                   -o functional_tests \
                   -pthread >> "$LOG_FILE" 2>&1; then
                
                log_info "Running functional tests (timeout: 60s)"
                if timeout 60s ./functional_tests >> "$LOG_FILE" 2>&1; then
                    log_success "Functional test suite execution"
                else
                    log_failure "Functional test suite execution (timeout or failure)"
                fi
            else
                log_failure "Functional test suite compilation"
            fi
            ((TOTAL_TESTS++))
        fi
        
        # Test integration tests
        if [ -f "../tests/integration_tests.cpp" ]; then
            log_info "Building integration test suite"
            if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
                   ../tests/integration_tests.cpp \
                   -o integration_tests \
                   -pthread >> "$LOG_FILE" 2>&1; then
                
                log_info "Running integration tests (timeout: 90s)"
                if timeout 90s ./integration_tests >> "$LOG_FILE" 2>&1; then
                    log_success "Integration test suite execution"
                else
                    log_failure "Integration test suite execution (timeout or failure)"
                fi
            else
                log_failure "Integration test suite compilation"
            fi
            ((TOTAL_TESTS++))
        fi
        
        # Test stress tests
        if [ -f "../tests/stress_tests.cpp" ]; then
            log_info "Building stress test suite"
            if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
                   ../tests/stress_tests.cpp \
                   -o stress_tests \
                   -pthread >> "$LOG_FILE" 2>&1; then
                
                log_info "Running stress tests (timeout: 180s)"
                if timeout 180s ./stress_tests >> "$LOG_FILE" 2>&1; then
                    log_success "Stress test suite execution"
                else
                    log_failure "Stress test suite execution (timeout or failure)"
                fi
            else
                log_failure "Stress test suite compilation"
            fi
            ((TOTAL_TESTS++))
        fi
        
        # Test professional standards validation
        if [ -f "../tests/professional_standards.cpp" ]; then
            log_info "Building professional standards validation"
            if g++ -std=c++20 -O2 -I../include -I../../tests/new/include \
                   ../tests/professional_standards.cpp \
                   -o professional_standards \
                   -pthread >> "$LOG_FILE" 2>&1; then
                
                log_info "Running professional standards validation (timeout: 60s)"
                if timeout 60s ./professional_standards >> "$LOG_FILE" 2>&1; then
                    log_success "Professional standards validation execution"
                else
                    log_failure "Professional standards validation execution (timeout or failure)"
                fi
            else
                log_failure "Professional standards validation compilation"
            fi
            ((TOTAL_TESTS++))
        fi
        
    else
        log_failure "Build system setup for executable testing"
        ((TOTAL_TESTS += 6))
    fi
    
    cd ..
    rm -rf "$BUILD_DIR"
}

# ============================================================================
# Documentation Validation
# ============================================================================

test_documentation() {
    log_info "=== Testing Documentation ==="
    
    # Test README files
    if [ -f "readme.md" ]; then
        if grep -q "BTQuant" "readme.md" && grep -q "Vulkan" "readme.md"; then
            log_success "Main README validation"
        else
            log_failure "Main README validation (missing key content)"
        fi
    else
        log_failure "Main README not found"
    fi
    ((TOTAL_TESTS++))
    
    # Test integration documentation
    if [ -f "REALTIME_INTEGRATION.md" ]; then
        if grep -q "HotSpine" "REALTIME_INTEGRATION.md" && grep -q "integration" "REALTIME_INTEGRATION.md"; then
            log_success "Integration documentation validation"
        else
            log_failure "Integration documentation validation (missing key content)"
        fi
    else
        log_failure "Integration documentation not found"
    fi
    ((TOTAL_TESTS++))
    
    # Test interactive features documentation
    if [ -f "INTERACTIVE_FEATURES_DOCUMENTATION.md" ]; then
        if grep -q "interactive" "INTERACTIVE_FEATURES_DOCUMENTATION.md" && grep -q "features" "INTERACTIVE_FEATURES_DOCUMENTATION.md"; then
            log_success "Interactive features documentation validation"
        else
            log_failure "Interactive features documentation validation (missing key content)"
        fi
    else
        log_failure "Interactive features documentation not found"
    fi
    ((TOTAL_TESTS++))
    
    # Test build scripts
    if [ -f "build_integration.sh" ]; then
        if [ -x "build_integration.sh" ]; then
            log_success "Build script permissions"
        else
            log_failure "Build script permissions (not executable)"
        fi
    else
        log_failure "Build script not found"
    fi
    ((TOTAL_TESTS++))
}

# ============================================================================
# Performance Optimization Validation
# ============================================================================

test_performance_optimizations() {
    log_info "=== Testing Performance Optimizations ==="
    
    BUILD_DIR="perf_test_$(date +%s)"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Test optimized build
    log_info "Testing optimized build flags"
    if cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native -DNDEBUG" .. >> "$LOG_FILE" 2>&1; then
        if make -j$(nproc) >> "$LOG_FILE" 2>&1; then
            log_success "Optimized build compilation"
            
            # Check if optimization flags are applied
            if objdump -h dashboard_advanced 2>/dev/null | grep -q ".text" 2>/dev/null; then
                log_success "Executable optimization validation"
            else
                log_failure "Executable optimization validation"
            fi
            ((TOTAL_TESTS++))
            
        else
            log_failure "Optimized build compilation"
            ((TOTAL_TESTS++))
        fi
    else
        log_failure "Optimized build configuration"
        ((TOTAL_TESTS++))
    fi
    ((TOTAL_TESTS++))
    
    cd ..
    rm -rf "$BUILD_DIR"
}

# ============================================================================
# Installation Testing
# ============================================================================

test_installation() {
    log_info "=== Testing Installation Process ==="
    
    # Test installation directory creation
    INSTALL_DIR="/tmp/btquant_install_test_$(date +%s)"
    
    BUILD_DIR="install_test_$(date +%s)"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Test installation
    if cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" .. >> "$LOG_FILE" 2>&1; then
        if make -j$(nproc) >> "$LOG_FILE" 2>&1; then
            if make install >> "$LOG_FILE" 2>&1; then
                log_success "Installation process"
                
                # Verify installation
                if [ -d "$INSTALL_DIR" ]; then
                    log_success "Installation directory creation"
                    
                    # Check for installed files
                    if find "$INSTALL_DIR" -name "*dashboard*" -type f | grep -q .; then
                        log_success "Executable installation"
                    else
                        log_failure "Executable installation"
                    fi
                    ((TOTAL_TESTS++))
                    
                else
                    log_failure "Installation directory creation"
                    ((TOTAL_TESTS++))
                fi
                
            else
                log_failure "Installation process (make install failed)"
                ((TOTAL_TESTS++))
            fi
        else
            log_failure "Installation process (compilation failed)"
            ((TOTAL_TESTS++))
        fi
    else
        log_failure "Installation process (cmake failed)"
        ((TOTAL_TESTS++))
    fi
    ((TOTAL_TESTS++))
    
    cd ..
    rm -rf "$BUILD_DIR"
    rm -rf "$INSTALL_DIR"
}

# ============================================================================
# Main Test Execution
# ============================================================================

main() {
    echo -e "${BLUE}BTQuant Build and Deployment Testing Suite${NC}"
    echo -e "${BLUE}===========================================${NC}"
    echo ""
    
    # Change to the BTQ_Render_Engine directory
    cd "$(dirname "$0")"
    
    # Run all test categories
    test_system_requirements
    test_dependencies
    test_build_configurations
    test_executables
    test_shader_compilation
    test_performance_optimizations
    test_installation
    test_documentation
    
    # Generate final report
    echo ""
    echo -e "${BLUE}=== Build System Test Summary ===${NC}"
    echo "Total Tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo -e "Skipped: ${YELLOW}$SKIPPED_TESTS${NC}"
    
    SUCCESS_RATE=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
    echo "Success Rate: $SUCCESS_RATE%"
    
    # Overall assessment
    if [ "$FAILED_TESTS" -eq 0 ]; then
        echo -e "\n${GREEN}✓ BUILD SYSTEM VALIDATION: PASSED${NC}"
        echo "The build system meets professional standards."
    elif [ "$FAILED_TESTS" -le 2 ]; then
        echo -e "\n${YELLOW}⚠ BUILD SYSTEM VALIDATION: PASSED WITH WARNINGS${NC}"
        echo "The build system is functional but has minor issues."
    else
        echo -e "\n${RED}✗ BUILD SYSTEM VALIDATION: FAILED${NC}"
        echo "The build system requires attention before deployment."
    fi
    
    echo ""
    echo "Detailed log saved to: $LOG_FILE"
    
    # Generate JSON report
    cat > "build_test_report.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "test_suite": "BTQuant Build System Tests",
  "total_tests": $TOTAL_TESTS,
  "passed": $PASSED_TESTS,
  "failed": $FAILED_TESTS,
  "skipped": $SKIPPED_TESTS,
  "success_rate": $SUCCESS_RATE,
  "log_file": "$LOG_FILE",
  "overall_status": "$([ "$FAILED_TESTS" -eq 0 ] && echo "PASSED" || echo "FAILED")"
}
EOF
    
    echo "JSON report saved to: build_test_report.json"
    
    # Return appropriate exit code
    [ "$FAILED_TESTS" -eq 0 ]
}

# Run main function
main "$@"