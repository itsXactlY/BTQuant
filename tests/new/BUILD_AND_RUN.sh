#!/bin/bash

# BTQuant Manipulation Detector - Build and Run Script
# Usage: ./BUILD_AND_RUN.sh [release|debug] [monitor|simple|multi]

set -e

BUILD_TYPE="${1:-release}"
TARGET="${2:-monitor}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║      BTQuant Manipulation Detector - Build Script          ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}❌ Error: CMakeLists.txt not found!${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Create build directory
# Create/clean build directory
BUILD_DIR="build"
if [ "$BUILD_TYPE" = "debug" ]; then
    BUILD_DIR="build_debug"
fi

# Clean previous build if it exists
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}🧹 Cleaning previous build...${NC}"
    rm -rf "$BUILD_DIR"
fi

echo -e "${YELLOW}📁 Build directory: $BUILD_DIR${NC}"
echo -e "${YELLOW}🔧 Build type: $BUILD_TYPE${NC}"
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo -e "${GREEN}⚙️  Configuring CMake...${NC}"
if [ "$BUILD_TYPE" = "debug" ]; then
    cmake -DCMAKE_BUILD_TYPE=Debug ..
else
    cmake -DCMAKE_BUILD_TYPE=Release ..
fi

# Build
echo -e "${GREEN}🔨 Building...${NC}"
make -j$(nproc)

echo ""
echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo ""

# Check if shared memory exists
if [ ! -e "/dev/shm/btquant_hotspine" ]; then
    echo -e "${YELLOW}⚠️  Warning: HotSpine shared memory not found at /dev/shm/btquant_hotspine${NC}"
    echo "Make sure your market data collector is running!"
    echo ""
    exit 0
fi

# Run the selected target
case "$TARGET" in
    monitor)
        echo -e "${GREEN}🚀 Starting main manipulation monitor...${NC}"
        echo ""
        ./manipulation_monitor
        ;;
    simple)
        echo -e "${GREEN}🚀 Starting simple monitor...${NC}"
        echo ""
        ./simple_monitor
        ;;
    multi)
        echo -e "${GREEN}🚀 Starting multi-exchange monitor...${NC}"
        echo ""
        ./multi_exchange_monitor
        ;;
    *)
        echo -e "${YELLOW}ℹ️  Executables built:${NC}"
        echo "  - ./manipulation_monitor"
        echo "  - ./simple_monitor"
        echo "  - ./multi_exchange_monitor"
        echo ""
        echo "Run manually or use:"
        echo "  ./BUILD_AND_RUN.sh [release|debug] [monitor|simple|multi]"
        ;;
esac