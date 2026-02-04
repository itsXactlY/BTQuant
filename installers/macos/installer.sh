#!/bin/bash

# PubBTQuant Installer Script for macOS
# This script installs the PubBTQuant trading terminal with all dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}PubBTQuant Trading Terminal Installer for macOS${NC}"
echo "=============================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please do not run this script as root${NC}"
    exit 1
fi

# Default installation directory
INSTALL_DIR="/Applications/PubBTQuant.app"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --dir DIR        Installation directory (default: /Applications/PubBTQuant.app)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Homebrew if not present
install_homebrew() {
    if ! command_exists brew; then
        echo -e "${GREEN}Installing Homebrew...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.bash_profile
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
}

# Function to install dependencies
install_dependencies() {
    echo -e "${GREEN}Installing dependencies via Homebrew...${NC}"
    
    install_homebrew
    
    # Install required dependencies
    brew install \
        cmake \
        vulkan-sdk \
        glfw \
        tbb \
        curl \
        openssl
    
    # Link Vulkan SDK if installed via Homebrew
    if [ -d "/opt/homebrew/share/vulkan" ] || [ -d "/usr/local/share/vulkan" ]; then
        echo -e "${GREEN}Vulkan SDK installed successfully${NC}"
    else
        echo -e "${YELLOW}Vulkan SDK may need manual installation. Please visit https://vulkan.lunarg.com/sdk/home#mac to download.${NC}"
    fi
    
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

# Function to download and build the application
download_and_build() {
    echo -e "${GREEN}Downloading and building PubBTQuant...${NC}"
    
    # Create temporary build directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Clone the repository (or copy from local source if available)
    if [ -d "/home/alca/projects/PubBTQuant" ]; then
        echo -e "${YELLOW}Using local source code...${NC}"
        cp -r /home/alca/projects/PubBTQuant/* .
    else
        echo -e "${RED}Local source not found. Please ensure the source code is available.${NC}"
        exit 1
    fi
    
    # Navigate to the engine directory
    cd dependencies/BTQ_Render_Engine
    
    # Create build directory and compile
    mkdir -p build
    cd build
    
    echo -e "${GREEN}Configuring with CMake...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
    
    echo -e "${GREEN}Compiling the application...${NC}"
    make -j$(sysctl -n hw.ncpu)
    
    # Return to temp directory
    cd ../..
    
    echo -e "${GREEN}Build completed successfully!${NC}"
    
    # Create application bundle structure
    APP_BUNDLE="$INSTALL_DIR"
    mkdir -p "$APP_BUNDLE"/Contents/{MacOS,Resources,Frameworks}
    
    # Copy built binaries
    cp dependencies/BTQ_Render_Engine/build/realtime_dashboard "$APP_BUNDLE/Contents/MacOS/realtime_dashboard"
    cp dependencies/BTQ_Render_Engine/build/dashboard_advanced "$APP_BUNDLE/Contents/MacOS/dashboard_advanced"
    
    # Copy resources
    cp -r dependencies/BTQ_Render_Engine/shaders "$APP_BUNDLE/Contents/Resources/"
    cp dependencies/BTQ_Render_Engine/default_layout.json "$APP_BUNDLE/Contents/Resources/"
    
    # Copy Info.plist
    cp /home/alca/projects/PubBTQuant/installers/macos/Info.plist "$APP_BUNDLE/Contents/"
    
    # Create a launcher script
    cat > "$APP_BUNDLE/Contents/MacOS/PubBTQuant" << EOF
#!/bin/bash
DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
exec "\$DIR/realtime_dashboard" "\$@"
EOF
    
    chmod +x "$APP_BUNDLE/Contents/MacOS/PubBTQuant"
    chmod +x "$APP_BUNDLE/Contents/MacOS/realtime_dashboard"
    chmod +x "$APP_BUNDLE/Contents/MacOS/dashboard_advanced"
    
    # Clean up
    cd /
    rm -rf "$TEMP_DIR"
    
    echo -e "${GREEN}Application bundle created at $APP_BUNDLE${NC}"
}

# Function to create DMG installer
create_dmg_installer() {
    echo -e "${GREEN}Creating DMG installer...${NC}"
    
    # Create a temporary directory for the DMG
    DMG_TEMP_DIR=$(mktemp -d)
    DMG_SRC_DIR="$DMG_TEMP_DIR/source"
    mkdir -p "$DMG_SRC_DIR"
    
    # Copy the app bundle to the source directory
    cp -r "$INSTALL_DIR" "$DMG_SRC_DIR/"
    
    # Create the DMG
    hdiutil create -volname "PubBTQuant" -srcfolder "$DMG_SRC_DIR" -ov -format UDZO "/tmp/PubBTQuant.dmg"
    
    echo -e "${GREEN}DMG installer created at /tmp/PubBTQuant.dmg${NC}"
    
    # Clean up
    rm -rf "$DMG_TEMP_DIR"
}

# Main installation process
main() {
    echo -e "${BLUE}Starting PubBTQuant installation...${NC}"
    echo -e "${YELLOW}Installation directory: $INSTALL_DIR${NC}"
    
    install_dependencies
    download_and_build
    create_dmg_installer
    
    echo ""
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}PubBTQuant has been installed to $INSTALL_DIR${NC}"
    echo ""
    echo -e "${BLUE}To start PubBTQuant, double-click the application in Finder,${NC}"
    echo -e "${BLUE}or run: open $INSTALL_DIR${NC}"
    echo ""
    echo -e "${YELLOW}A DMG installer has been created at /tmp/PubBTQuant.dmg${NC}"
    echo -e "${YELLOW}for distribution purposes.${NC}"
}

# Run main function
main