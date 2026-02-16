#!/bin/bash

# Unified Build Script for PubBTQuant
# Handles both main project and render engine dependency builds

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_DIR="build"
SOURCE_DIR="."
INSTALL_DIR=""
NUM_CORES=$(nproc)
VERBOSE=false
BUILD_TYPE="Release"
SANITIZE_CODE=false
CLEAN_BUILD=false
BUILD_RENDER_ENGINE=false
BUILD_TESTS=false
SHOW_HELP=false

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --directory DIR      Build directory (default: build)"
    echo "  -s, --source-dir DIR     Source directory (default: .)"
    echo "  -i, --install-dir DIR    Installation directory"
    echo "  -v, --verbose            Verbose output"
    echo "  -j, --jobs N             Number of parallel jobs (default: nproc)"
    echo "  --debug                  Build in debug mode"
    echo "  --release                Build in release mode (default)"
    echo "  --sanitize               Enable sanitizers for debugging"
    echo "  --clean                  Clean build directory before building"
    echo "  --render-engine          Build BTQ_Render_Engine dependency"
    echo "  --tests                  Build test executables"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Default release build"
    echo "  $0 --clean --release            # Clean release build"
    echo "  $0 --debug --sanitize           # Debug build with sanitizers"
    echo "  $0 --render-engine              # Build render engine dependency"
    echo "  $0 --clean --tests              # Clean build with tests enabled"
}

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
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --sanitize)
            SANITIZE_CODE=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --render-engine)
            BUILD_RENDER_ENGINE=true
            shift
            ;;
        --tests)
            BUILD_TESTS=true
            shift
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    print_usage
    exit 0
fi

# Store original directory
ORIGINAL_DIR=$(pwd)

# Build BTQ_Render_Engine if requested
if [ "$BUILD_RENDER_ENGINE" = true ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Building BTQ_Render_Engine Dependency                     ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    RENDER_ENGINE_DIR="$SOURCE_DIR/dependencies/BTQ_Render_Engine"
    
    if [ ! -f "$RENDER_ENGINE_DIR/build_integration.sh" ]; then
        echo -e "${RED}Error: Render engine build script not found at $RENDER_ENGINE_DIR/build_integration.sh${NC}"
        exit 1
    fi
    
    cd "$RENDER_ENGINE_DIR"
    ./build_integration.sh
    cd "$ORIGINAL_DIR"
    
    echo ""
    echo -e "${GREEN}✓ Render engine build completed${NC}"
    echo ""
fi

# Main project build
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Building PubBTQuant                                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}[CONFIG]${NC} Build configuration:"
echo -e "${BLUE}[CONFIG]${NC}   Build directory: $BUILD_DIR"
echo -e "${BLUE}[CONFIG]${NC}   Source directory: $SOURCE_DIR"
echo -e "${BLUE}[CONFIG]${NC}   Build type: $BUILD_TYPE"
echo -e "${BLUE}[CONFIG]${NC}   Parallel jobs: $NUM_CORES"
if [ "$SANITIZE_CODE" = true ]; then
    echo -e "${BLUE}[CONFIG]${NC}   Sanitizers: enabled"
fi
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${BLUE}[CONFIG]${NC}   Clean build: yes"
fi
if [ "$BUILD_TESTS" = true ]; then
    echo -e "${BLUE}[CONFIG]${NC}   Tests: enabled"
fi
echo ""

# Validate source directory and convert to absolute path
if [[ "$SOURCE_DIR" != /* ]]; then
    SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
fi

if [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    echo -e "${RED}Error: CMakeLists.txt not found in $SOURCE_DIR${NC}"
    exit 1
fi

# Handle clean build
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}[CLEAN]${NC} Removing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${BLUE}[BUILD]${NC} Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Convert build directory to absolute path
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

# Change to build directory
cd "$BUILD_DIR"

# Set verbose mode for make if requested
MAKE_VERBOSE_FLAG=""
if [ "$VERBOSE" = true ]; then
    MAKE_VERBOSE_FLAG="VERBOSE=1"
    set -x
fi

# Configure with CMake
echo -e "${BLUE}[CMAKE]${NC} Configuring project..."

if [ "$SANITIZE_CODE" = true ]; then
    echo -e "${BLUE}[CMAKE]${NC} Enabling sanitizers for error detection..."
    cmake "$SOURCE_DIR" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CXX_STANDARD=26 \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -flto -ffunction-sections -fdata-sections" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined -Wl,--no-as-needed" \
        -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined -Wl,--no-as-needed" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR:-/usr/local}" \
        -DBUILD_TESTS=$BUILD_TESTS
else
    cmake "$SOURCE_DIR" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CXX_STANDARD=26 \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 -Wall -Wextra" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -flto -ffunction-sections -fdata-sections" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR:-/usr/local}" \
        -DBUILD_TESTS=$BUILD_TESTS
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: CMake configuration failed${NC}"
    exit 1
fi

# Build the project
echo -e "${BLUE}[BUILD]${NC} Building project with $NUM_CORES jobs..."
ninja $MAKE_VERBOSE_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Build failed${NC}"
    exit 1
fi

# Return to original directory
cd "$ORIGINAL_DIR"

# Show build summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Build Completed Successfully                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}[SUMMARY]${NC} Build directory: $(cd "$BUILD_DIR" && pwd)"
echo ""

# Show executable information
echo -e "${BLUE}[BINARIES]${NC} Built executables:"
find "$BUILD_DIR" -type f -executable -name "pubbtquant*" 2>/dev/null | while read -r binary; do
    ls -lh "$binary"
done

# Show test executables if tests were built
if [ "$BUILD_TESTS" = true ]; then
    echo ""
    echo -e "${BLUE}[TESTS]${NC} Test executables:"
    find "$BUILD_DIR" -type f -executable -name "*test*" 2>/dev/null | while read -r binary; do
        ls -lh "$binary"
    done
fi

echo ""
echo -e "${GREEN}✓ Build completed successfully!${NC}"
echo ""
echo "To run the application:"
echo "  cd $BUILD_DIR && ./pubbtquant"
echo ""
