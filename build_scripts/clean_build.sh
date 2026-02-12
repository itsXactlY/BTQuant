#!/bin/bash

# Clean Build Script for PubBTQuant
# This script performs a complete clean build by removing all build artifacts and rebuilding from scratch

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting clean build process for PubBTQuant...${NC}"

# Default values
BUILD_DIR="build"
RELEASE_BUILD_DIR="build_release"
SOURCE_DIR="."
NUM_CORES=$(nproc)
VERBOSE=false
BUILD_TYPE="Release"
BUILD_ENGINE_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            BUILD_DIR="$2"
            shift 2
            ;;
        -r|--release-directory)
            RELEASE_BUILD_DIR="$2"
            shift 2
            ;;
        -s|--source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            NUM_CORES="$2"
            shift 2
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        -e|--engine-only)
            BUILD_ENGINE_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --directory DIR         Build directory (default: build)"
            echo "  -r, --release-directory DIR Build directory for release (default: build_release)"
            echo "  -s, --source-dir DIR        Source directory (default: .)"
            echo "  -v, --verbose               Verbose output"
            echo "  -j, --jobs N                Number of parallel jobs (default: nproc)"
            echo "  --debug                     Build in debug mode"
            echo "  --release                   Build in release mode (default)"
            echo "  -e, --engine-only           Build only the BTQ_Render_Engine"
            echo "  -h, --help                  Show this help message"
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
echo -e "${YELLOW}Release build directory: $RELEASE_BUILD_DIR${NC}"
echo -e "${YELLOW}Source directory: $SOURCE_DIR${NC}"
echo -e "${YELLOW}Number of cores: $NUM_CORES${NC}"
echo -e "${YELLOW}Build type: $BUILD_TYPE${NC}"

# Store the original directory before changing to build directory
ORIGINAL_DIR=$(pwd)

# Determine the source directory to build
if [ "$BUILD_ENGINE_ONLY" = true ]; then
    # Build only the BTQ_Render_Engine
    if [ -d "$ORIGINAL_DIR/dependencies/BTQ_Render_Engine" ]; then
        SOURCE_DIR="$ORIGINAL_DIR/dependencies/BTQ_Render_Engine"
        echo -e "${YELLOW}Building only the BTQ_Render_Engine${NC}"
    else
        echo -e "${RED}BTQ_Render_Engine directory not found${NC}"
        exit 1
    fi
elif [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    # If no CMakeLists.txt in source dir, try the BTQ_Render_Engine
    if [ -d "$ORIGINAL_DIR/dependencies/BTQ_Render_Engine" ]; then
        SOURCE_DIR="$ORIGINAL_DIR/dependencies/BTQ_Render_Engine"
        echo -e "${YELLOW}Using BTQ_Render_Engine as source directory${NC}"
    else
        echo -e "${RED}No CMakeLists.txt found in source directory and BTQ_Render_Engine not found${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Performing clean build...${NC}"

# Remove both build directories to ensure a completely clean state
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing existing build directory: $BUILD_DIR${NC}"
    rm -rf "$BUILD_DIR"
fi

if [ -d "$RELEASE_BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing existing release build directory: $RELEASE_BUILD_DIR${NC}"
    rm -rf "$RELEASE_BUILD_DIR"
fi

# Completely remove and recreate the build directory to ensure a clean state
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing existing build directory: $BUILD_DIR${NC}"
    rm -rf "$BUILD_DIR"
fi

echo -e "${GREEN}Creating fresh build directory: $BUILD_DIR${NC}"
mkdir -p "$BUILD_DIR"

# Change to build directory
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring CMake with $BUILD_TYPE settings...${NC}"

cmake -B . -G Ninja "$ORIGINAL_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CXX_STANDARD=26 \
    -DCMAKE_INSTALL_PREFIX="/usr/local" \
    -DBUILD_TESTS=ON

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
fi

echo -e "${GREEN}Compiling...${NC}"
ninja $MAKE_VERBOSE_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}Clean build completed successfully!${NC}"

# Show size of built binaries
echo -e "${GREEN}Binary sizes:${NC}"
find . -type f -executable -exec file {} \; | grep -E "(ELF|executable)" | cut -d: -f1 | xargs ls -lh

cd ..

echo -e "${GREEN}Clean build process finished.${NC}"