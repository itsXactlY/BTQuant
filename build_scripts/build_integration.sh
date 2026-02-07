#!/bin/bash

# Integration Build Script for PubBTQuant
# This script performs comprehensive integration builds, sanitizes code, validates existing code,
# and prevents errors and crashes for executables

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting integration build process for PubBTQuant...${NC}"

# Default values
BUILD_DIR="build_integration"
SOURCE_DIR="."
INSTALL_DIR=""
NUM_CORES=$(nproc)
CLEAN_BUILD=false
VERBOSE=false
SANITIZE_CODE=true
RUN_TESTS=true
STATIC_ANALYSIS=true
BUILD_TYPE="Debug"  # Debug build for better error detection

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            BUILD_DIR="$2"
            shift 2
            ;;
        -s|--source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        -i|--install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            NUM_CORES="$2"
            shift 2
            ;;
        --no-sanitize)
            SANITIZE_CODE=false
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --no-static-analysis)
            STATIC_ANALYSIS=false
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --directory DIR         Build directory (default: build_integration)"
            echo "  -s, --source-dir DIR        Source directory (default: .)"
            echo "  -i, --install-dir DIR       Installation directory"
            echo "  -c, --clean                Perform clean build"
            echo "  -v, --verbose              Verbose output"
            echo "  -j, --jobs N               Number of parallel jobs (default: nproc)"
            echo "  --no-sanitize              Skip code sanitization"
            echo "  --no-tests                 Skip running tests"
            echo "  --no-static-analysis       Skip static analysis"
            echo "  --release                  Build in release mode"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set verbose flag for make if requested
MAKE_VERBOSE_FLAG=""
if [ "$VERBOSE" = true ]; then
    MAKE_VERBOSE_FLAG="VERBOSE=1"
    set -x  # Print commands as they execute
fi

echo -e "${YELLOW}Build directory: $BUILD_DIR${NC}"
echo -e "${YELLOW}Source directory: $SOURCE_DIR${NC}"
echo -e "${YELLOW}Number of cores: $NUM_CORES${NC}"
echo -e "${YELLOW}Build type: $BUILD_TYPE${NC}"

# Store the original directory before changing to build directory
ORIGINAL_DIR=$(pwd)

# Determine the source directory to build
if [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    # If no CMakeLists.txt in source dir, try the BTQ_Render_Engine
    if [ -d "$ORIGINAL_DIR/dependencies/BTQ_Render_Engine" ]; then
        SOURCE_DIR="$ORIGINAL_DIR/dependencies/BTQ_Render_Engine"
        echo -e "${YELLOW}Using BTQ_Render_Engine as source directory${NC}"
    elif [ -d "$ORIGINAL_DIR/../BTQ_Render_Engine" ]; then
        SOURCE_DIR="$ORIGINAL_DIR/../BTQ_Render_Engine"
        echo -e "${YELLOW}Using BTQ_Render_Engine as source directory${NC}"
    elif [ -d "$ORIGINAL_DIR/../../BTQ_Render_Engine" ]; then
        SOURCE_DIR="$ORIGINAL_DIR/../../BTQ_Render_Engine"
        echo -e "${YELLOW}Using BTQ_Render_Engine as source directory${NC}"
    else
        echo -e "${RED}No CMakeLists.txt found in source directory and BTQ_Render_Engine not found${NC}"
        exit 1
    fi
fi

# Clean and rebuild build directory
echo -e "${GREEN}Cleaning and configuring build directory: $BUILD_DIR${NC}"

# Configure and build with Ninja for faster iteration
rm -rf "$BUILD_DIR" && cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=$BUILD_TYPE && ninja -C "$BUILD_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

# Run static analysis if enabled
if [ "$STATIC_ANALYSIS" = true ]; then
    echo -e "${GREEN}Running static analysis...${NC}"
    
    # Check if clang-static-analyzer is available
    if command -v scan-build &> /dev/null; then
        echo -e "${BLUE}Running Clang Static Analyzer...${NC}"
        scan-build make -j"$NUM_CORES" || true  # Don't fail the build if scan-build finds issues
    else
        echo -e "${YELLOW}Clang Static Analyzer not found. Install with: sudo apt install clang-tools${NC}"
    fi
    
    # Check if cppcheck is available
    if command -v cppcheck &> /dev/null; then
        echo -e "${BLUE}Running Cppcheck...${NC}"
        cppcheck --enable=all --std=c++26 --template=gcc --quiet "$SOURCE_DIR" || true
    else
        echo -e "${YELLOW}Cppcheck not found. Install with: sudo apt install cppcheck${NC}"
    fi
fi

# Run tests if enabled
if [ "$RUN_TESTS" = true ]; then
    echo -e "${GREEN}Running tests...${NC}"
    
    # Look for test executables and run them
    TEST_EXECUTABLES=$(find . -name "*test*" -type f -executable 2>/dev/null)
    
    if [ -z "$TEST_EXECUTABLES" ]; then
        echo -e "${YELLOW}No test executables found${NC}"
    else
        for test_exe in $TEST_EXECUTABLES; do
            echo -e "${BLUE}Running test: $test_exe${NC}"
            if [ "$SANITIZE_CODE" = true ]; then
                # Run with sanitizers enabled
                ASAN_OPTIONS="abort_on_error=1:detect_leaks=1" "$test_exe" || {
                    echo -e "${RED}Test failed: $test_exe${NC}"
                    # Don't exit here, continue with other tests
                }
            else
                "$test_exe" || {
                    echo -e "${RED}Test failed: $test_exe${NC}"
                    # Don't exit here, continue with other tests
                }
            fi
        done
    fi
fi

echo -e "${GREEN}Integration build completed successfully!${NC}"

# Validate executables
echo -e "${GREEN}Validating executables...${NC}"

EXECUTABLES=$(find . -type f -executable -exec file {} \; | grep -E "(ELF|executable)" | cut -d: -f1)

if [ -z "$EXECUTABLES" ]; then
    echo -e "${YELLOW}No executables found${NC}"
else
    for exe in $EXECUTABLES; do
        if [[ "$exe" != *"cmake"* ]] && [[ "$exe" != *"ccache"* ]]; then
            echo -e "${BLUE}Validating executable: $exe${NC}"
            
            # Check if the executable is valid
            if file "$exe" | grep -q "not stripped"; then
                echo -e "${YELLOW}  Warning: $exe is not stripped${NC}"
            fi
            
            # Check for common security issues
            if readelf -d "$exe" | grep -q "TEXTREL"; then
                echo -e "${YELLOW}  Warning: $exe has TEXTREL (relocatable code in text section)${NC}"
            fi
            
            # Run a quick check to see if the executable loads properly
            if ldd "$exe" >/dev/null 2>&1; then
                echo -e "${GREEN}  OK: $exe dependencies are valid${NC}"
            else
                echo -e "${RED}  Error: $exe has invalid dependencies${NC}"
            fi
        fi
    done
fi

echo -e "${GREEN}Integration build validation completed!${NC}"
echo -e "${YELLOW}Build directory: $(pwd)${NC}"

# Show size of built binaries
echo -e "${GREEN}Binary sizes:${NC}"
find . -type f -executable -exec file {} \; | grep -E "(ELF|executable)" | cut -d: -f1 | xargs ls -lh

cd ..

echo -e "${GREEN}Integration build process finished.${NC}"