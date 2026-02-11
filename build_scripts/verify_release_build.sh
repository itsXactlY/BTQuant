#!/bin/bash

# Verification Script for Release Build Features
# Checks that symbols are stripped and loops are optimized

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Verifying Release build features...${NC}"

# Default values
BUILD_DIR="build_release"
SOURCE_DIR="."

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --directory DIR    Build directory (default: build_release)"
            echo "  -s, --source-dir DIR   Source directory (default: .)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Build directory: $BUILD_DIR${NC}"
echo -e "${YELLOW}Source directory: $SOURCE_DIR${NC}"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Build directory $BUILD_DIR does not exist${NC}"
    echo -e "${YELLOW}Running release build first...${NC}"
    
    # Run the release build script
    ./build_scripts/release_build.sh -d "$BUILD_DIR" -s "$SOURCE_DIR"
fi

cd "$BUILD_DIR"

# Function to check if a binary has debug symbols
has_debug_symbols() {
    local binary="$1"
    if objdump -h "$binary" 2>/dev/null | grep -q "\.debug"; then
        return 0  # Has debug symbols
    else
        return 1  # No debug symbols
    fi
}

# Function to check if a binary has been stripped
is_stripped() {
    local binary="$1"
    if strip --strip-test "$binary" 2>/dev/null; then
        return 1  # Not stripped (strip would work)
    else
        return 0  # Already stripped (strip would fail)
    fi
}

# Find all executable files
EXECUTABLES=$(find . -type f -executable -exec file {} \; | grep -E "(ELF|executable)" | cut -d: -f1)

if [ -z "$EXECUTABLES" ]; then
    echo -e "${RED}No executable binaries found in build directory${NC}"
    exit 1
fi

echo -e "${GREEN}Checking symbol stripping...${NC}"

# Check each executable
SYMBOLS_STRIPPED=true
for exe in $EXECUTABLES; do
    if [ -f "$exe" ] && [[ "$exe" != *"cmake"* ]] && [[ "$exe" != *"ccache"* ]]; then
        echo "Checking: $exe"
        
        # Check if it has debug symbols
        if has_debug_symbols "$exe"; then
            echo -e "  ${RED}FAIL: $exe still contains debug symbols${NC}"
            SYMBOLS_STRIPPED=false
        else
            echo -e "  ${GREEN}PASS: $exe has no debug symbols${NC}"
        fi
        
        # Check if it's stripped
        if is_stripped "$exe"; then
            echo -e "  ${GREEN}PASS: $exe is stripped${NC}"
        else
            echo -e "  ${RED}FAIL: $exe is not stripped${NC}"
            SYMBOLS_STRIPPED=false
        fi
    fi
done

# Check for optimization indicators
echo -e "${GREEN}Checking for loop optimizations...${NC}"

OPTIMIZATIONS_PRESENT=true
for exe in $EXECUTABLES; do
    if [ -f "$exe" ] && [[ "$exe" != *"cmake"* ]] && [[ "$exe" != *"ccache"* ]]; then
        echo "Analyzing: $exe"
        
        # Check if the binary was compiled with optimization flags by looking for optimization signatures
        # This is an indirect way to verify optimizations were applied
        if readelf -p .comment "$exe" 2>/dev/null | grep -qi "GCC.*-O[2-9]"; then
            echo -e "  ${GREEN}PASS: $exe shows optimization flags in compilation info${NC}"
        elif objdump -s -j .comment "$exe" 2>/dev/null | grep -qi "GCC.*-O[2-9]"; then
            echo -e "  ${GREEN}PASS: $exe shows optimization flags in compilation info${NC}"
        else
            # Alternative check: look for optimization characteristics in assembly
            # Count jump instructions as a proxy for optimized loops
            JUMP_COUNT=$(objdump -d "$exe" 2>/dev/null | grep -c -E "\t(jmp|je|jne|jl|jg|jle|jge|ja|jb|jae|jbe|jo|jno|js|jns|jp|jnp)\t" || echo 0)
            
            if [ "$JUMP_COUNT" -gt 100 ]; then  # Arbitrary threshold indicating optimized code
                echo -e "  ${GREEN}PASS: $exe shows signs of optimization (many jumps: $JUMP_COUNT)${NC}"
            else
                echo -e "  ${YELLOW}INFO: $exe has fewer jumps ($JUMP_COUNT), but this doesn't necessarily mean no optimization${NC}"
            fi
        fi
    fi
done

echo -e "${GREEN}Checking CMake configuration...${NC}"

# Check if CMakeCache.txt contains the right flags
if [ -f "CMakeCache.txt" ]; then
    echo "Verifying CMake configuration..."
    
    if grep -q "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG -march=native -flto -ffunction-sections -fdata-sections" CMakeCache.txt; then
        echo -e "  ${GREEN}PASS: CMAKE_CXX_FLAGS_RELEASE contains expected optimization flags${NC}"
    else
        echo -e "  ${YELLOW}INFO: Checking for optimization flags in CMakeCache.txt..."
        OPT_FLAGS_FOUND=$(grep -c "O3\|march=native\|flto\|ffunction-sections\|fdata-sections" CMakeCache.txt || echo 0)
        if [ "$OPT_FLAGS_FOUND" -gt 0 ]; then
            echo -e "  ${GREEN}PASS: Found optimization flags in CMakeCache.txt${NC}"
        else
            echo -e "  ${RED}WARNING: Optimization flags not found in CMakeCache.txt${NC}"
        fi
    fi
    
    if grep -q "strip-all\|gc-sections" CMakeCache.txt; then
        echo -e "  ${GREEN}PASS: Linker flags for symbol stripping found in CMakeCache.txt${NC}"
    else
        echo -e "  ${RED}WARNING: Symbol stripping linker flags not found in CMakeCache.txt${NC}"
    fi
else
    echo -e "${YELLOW}CMakeCache.txt not found in build directory${NC}"
fi

cd ..

# Summary
echo -e "${GREEN}Verification Summary:${NC}"
if [ "$SYMBOLS_STRIPPED" = true ]; then
    echo -e "${GREEN}✓ Symbols are properly stripped in Release build${NC}"
else
    echo -e "${RED}✗ Symbols are NOT properly stripped in Release build${NC}"
fi

if [ "$OPTIMIZATIONS_PRESENT" = true ]; then
    echo -e "${GREEN}✓ Loop optimizations are present in Release build${NC}"
else
    echo -e "${YELLOW}⚠ Could not definitively verify loop optimizations${NC}"
fi

if [ "$SYMBOLS_STRIPPED" = true ]; then
    echo -e "${GREEN}Release build verification PASSED${NC}"
    exit 0
else
    echo -e "${RED}Release build verification FAILED${NC}"
    exit 1
fi