#!/bin/bash

# Clean & Build Script for PubBTQuant
# This script performs a clean build by removing existing build artifacts and rebuilding the project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting clean & build process for PubBTQuant...${NC}"

# Default values
BUILD_DIR="build"
SOURCE_DIR="."
INSTALL_DIR=""
NUM_CORES=$(nproc)
VERBOSE=false
BUILD_TYPE="Debug"
SANITIZE_CODE=false

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
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            NUM_CORES="$2"
            shift 2
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --sanitize)
            SANITIZE_CODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --directory DIR    Build directory (default: build)"
            echo "  -s, --source-dir DIR   Source directory (default: .)"
            echo "  -i, --install-dir DIR  Installation directory"
            echo "  -v, --verbose          Verbose output"
            echo "  -j, --jobs N          Number of parallel jobs (default: nproc)"
            echo "  --release             Build in release mode"
            echo "  --sanitize            Enable sanitizers for debugging"
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
echo -e "${YELLOW}Build type: $BUILD_TYPE${NC}"

# Store original directory before changing
ORIGINAL_WORKING_DIR=$(pwd)
if [[ "$SOURCE_DIR" == /* ]]; then
    # Already an absolute path
    ORIGINAL_SOURCE_DIR="$SOURCE_DIR"
else
    # Convert to absolute path
    ORIGINAL_SOURCE_DIR="$ORIGINAL_WORKING_DIR/$SOURCE_DIR"
fi

# Determine the source directory to build
if [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    echo -e "${RED}No CMakeLists.txt found in source directory${NC}"
    exit 1
fi

# Clean the build directory
echo -e "${GREEN}Cleaning build directory: $BUILD_DIR${NC}"

if [ -d "$BUILD_DIR" ]; then
    # Remove all contents of the build directory
    echo -e "${YELLOW}Removing contents of $BUILD_DIR${NC}"
    rm -rf "$BUILD_DIR"/*
    echo -e "${GREEN}Build directory cleaned${NC}"
else
    echo -e "${GREEN}Build directory does not exist, creating it...${NC}"
    mkdir -p "$BUILD_DIR"
fi

# Change to build directory
echo -e "${YELLOW}Changing to build directory: $BUILD_DIR${NC}"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring CMake...${NC}"
echo -e "${YELLOW}Current directory: $(pwd)${NC}"
echo -e "${YELLOW}Source directory for cmake: $ORIGINAL_SOURCE_DIR${NC}"

if [ "$SANITIZE_CODE" = true ]; then
    echo -e "${BLUE}Enabling sanitizers for error detection...${NC}"
    cmake "$ORIGINAL_SOURCE_DIR" \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_STANDARD=26 \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Werror" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -flto -ffunction-sections -fdata-sections" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined -Wl,--no-as-needed" \
        -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined -Wl,--no-as-needed" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR:-/usr/local}" \
        -DBUILD_TESTS=ON
else
    cmake "$ORIGINAL_SOURCE_DIR" \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_STANDARD=26 \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 -Wall -Wextra -Werror" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -flto -ffunction-sections -fdata-sections" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR:-/usr/local}" \
        -DBUILD_TESTS=ON
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
fi

echo -e "${GREEN}Building the project...${NC}"
make $MAKE_VERBOSE_FLAG -j"$NUM_CORES"

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}Clean & build completed successfully!${NC}"
echo -e "${YELLOW}Build directory: $(pwd)${NC}"

# Show size of built binaries
echo -e "${GREEN}Binary sizes:${NC}"
find . -type f -executable -exec file {} \; | grep -E "(ELF|executable)" | cut -d: -f1 | xargs ls -lh

cd ..
echo -e "${GREEN}Clean & build process finished.${NC}"