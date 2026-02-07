#!/bin/bash

# Release Build Script for PubBTQuant
# This script performs optimized compilation, strips debug symbols, and minimizes binary size

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting release build process for PubBTQuant...${NC}"

# Default values
BUILD_DIR="build_release"
SOURCE_DIR="."
INSTALL_DIR=""
NUM_CORES=$(nproc)
CLEAN_BUILD=false
VERBOSE=false
BUILD_ENGINE_ONLY=false

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
        -e|--engine-only)
            BUILD_ENGINE_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --directory DIR    Build directory (default: build_release)"
            echo "  -s, --source-dir DIR   Source directory (default: .)"
            echo "  -i, --install-dir DIR  Installation directory"
            echo "  -c, --clean           Perform clean build"
            echo "  -v, --verbose         Verbose output"
            echo "  -j, --jobs N          Number of parallel jobs (default: nproc)"
            echo "  -e, --engine-only     Build only the BTQ_Render_Engine"
            echo "  -h, --help            Show this help message"
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

# Determine the source directory to build
if [ "$BUILD_ENGINE_ONLY" = true ]; then
    # Build only the BTQ_Render_Engine
    if [ -d "./dependencies/BTQ_Render_Engine" ]; then
        SOURCE_DIR="./dependencies/BTQ_Render_Engine"
        echo -e "${YELLOW}Building only the BTQ_Render_Engine${NC}"
    else
        echo -e "${RED}BTQ_Render_Engine directory not found${NC}"
        exit 1
    fi
elif [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    # If no CMakeLists.txt in source dir, try the BTQ_Render_Engine
    if [ -d "./dependencies/BTQ_Render_Engine" ]; then
        SOURCE_DIR="./dependencies/BTQ_Render_Engine"
        echo -e "${YELLOW}Using BTQ_Render_Engine as source directory${NC}"
    else
        echo -e "${RED}No CMakeLists.txt found in source directory and BTQ_Render_Engine not found${NC}"
        exit 1
    fi
fi

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${GREEN}Creating build directory: $BUILD_DIR${NC}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${GREEN}Performing clean build...${NC}"
    rm -rf ./*
fi

# Check if CMakeCache.txt exists to determine if we need to configure
if [ ! -f "CMakeCache.txt" ] || [ "$CLEAN_BUILD" = true ]; then
    echo -e "${GREEN}Configuring CMake with release settings...${NC}"

    # Configure with release flags for optimization and size reduction
    cmake -B . -G Ninja "$SOURCE_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_STANDARD=26 \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -flto -ffunction-sections -fdata-sections -fno-rtti -fno-exceptions" \
        -DCMAKE_EXE_LINKER_FLAGS_RELEASE="-Wl,--gc-sections -Wl,-strip-all -Wl,--exclude-libs,ALL -Wl,--as-needed" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--gc-sections -Wl,-strip-all -Wl,--as-needed" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR:-/usr/local}" \
        -DBUILD_TESTS=OFF

    if [ $? -ne 0 ]; then
        echo -e "${RED}CMake configuration failed${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Using existing CMake configuration${NC}"
fi

echo -e "${GREEN}Compiling with optimizations...${NC}"
ninja $MAKE_VERBOSE_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"

# Additional optimization: Strip debug symbols from binaries
echo -e "${GREEN}Stripping debug symbols from binaries...${NC}"

# Find all executable files and strip them
for exe in $(find . -type f -executable -exec file {} \; | grep -E "(ELF|executable)" | cut -d: -f1); do
    if [ -f "$exe" ] && [[ "$exe" != *"cmake"* ]] && [[ "$exe" != *"ccache"* ]]; then
        echo "Stripping $exe"
        strip --strip-debug --strip-unneeded "$exe" 2>/dev/null || true
    fi
done

# Additional size optimizations
echo -e "${GREEN}Performing additional size optimizations...${NC}"

# Optimize binaries further with UPX if available (optional compression)
if command -v upx &> /dev/null; then
    echo "UPX found, compressing binaries..."
    for exe in $(find . -type f -executable -exec file {} \; | grep -E "(ELF|executable)" | cut -d: -f1); do
        if [ -f "$exe" ] && [[ "$exe" != *"cmake"* ]] && [[ "$exe" != *"ccache"* ]]; then
            echo "Compressing $exe with UPX..."
            upx --best --lzma "$exe" 2>/dev/null || true
        fi
    done
else
    echo "UPX not found. Install it for additional binary compression: sudo apt install upx-ucl"
fi

echo -e "${GREEN}Release build completed successfully!${NC}"
echo -e "${YELLOW}Build directory: $(pwd)${NC}"

# Show size of built binaries
echo -e "${GREEN}Binary sizes:${NC}"
find . -type f -executable -exec file {} \; | grep -E "(ELF|executable)" | cut -d: -f1 | xargs ls -lh

cd ..

echo -e "${GREEN}Release build process finished.${NC}"