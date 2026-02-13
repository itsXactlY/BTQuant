#!/bin/bash

# PubBTQuant Installer Script for Linux
# This script installs the PubBTQuant trading terminal with all dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}PubBTQuant Trading Terminal Installer for Linux${NC}"
echo "=============================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please do not run this script as root${NC}"
    exit 1
fi

# Default installation directory
INSTALL_DIR="$HOME/.pubbtquant"
SYSTEM_WIDE_INSTALL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --system-wide)
            SYSTEM_WIDE_INSTALL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --dir DIR        Installation directory (default: $HOME/.pubbtquant)"
            echo "  --system-wide        Install system-wide (requires sudo)"
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

# Function to detect package manager
detect_package_manager() {
    if command_exists apt; then
        PKG_MANAGER="apt"
    elif command_exists yum; then
        PKG_MANAGER="yum"
    elif command_exists dnf; then
        PKG_MANAGER="dnf"
    elif command_exists pacman; then
        PKG_MANAGER="pacman"
    elif command_exists zypper; then
        PKG_MANAGER="zypper"
    else
        echo -e "${RED}Unsupported package manager. Please install dependencies manually.${NC}"
        exit 1
    fi
}

# Function to install dependencies based on detected package manager
install_dependencies() {
    echo -e "${GREEN}Detecting system and installing dependencies...${NC}"
    
    detect_package_manager
    echo -e "${YELLOW}Detected package manager: $PKG_MANAGER${NC}"
    
    # Determine if we need sudo based on installation type
    SUDO_CMD=""
    if [ "$SYSTEM_WIDE_INSTALL" = true ]; then
        SUDO_CMD="sudo"
    fi
    
    case $PKG_MANAGER in
        apt)
            echo -e "${GREEN}Installing dependencies via apt...${NC}"
            $SUDO_CMD apt update
            $SUDO_CMD apt install -y \
                build-essential \
                cmake \
                vulkan-sdk \
                libvulkan-dev \
                libglfw3-dev \
                libgl1-mesa-dev \
                libglu1-mesa-dev \
                libtbb-dev \
                libcurl4-openssl-dev \
                libssl-dev \
                wget \
                git \
                unzip
            ;;
        dnf|yum)
            echo -e "${GREEN}Installing dependencies via $PKG_MANAGER...${NC}"
            $SUDO_CMD $PKG_MANAGER install -y \
                gcc-c++ \
                cmake \
                vulkan-devel \
                mesa-libGL-devel \
                mesa-libGLU-devel \
                glfw-devel \
                tbb-devel \
                libcurl-devel \
                openssl-devel \
                wget \
                git \
                unzip
            ;;
        pacman)
            echo -e "${GREEN}Installing dependencies via pacman...${NC}"
            $SUDO_CMD pacman -Syu --noconfirm \
                gcc \
                cmake \
                vulkan-headers \
                vulkan-icd-loader \
                glfw-x11 \
                mesa \
                intel-tbb \
                curl \
                openssl \
                wget \
                git \
                unzip
            ;;
        zypper)
            echo -e "${GREEN}Installing dependencies via zypper...${NC}"
            $SUDO_CMD zypper refresh
            $SUDO_CMD zypper install -y \
                gcc-c++ \
                cmake \
                vulkan-devel \
                Mesa-libGL-devel \
                Mesa-libGLU-devel \
                glfw-devel \
                tbb-devel \
                libcurl-devel \
                libopenssl-devel \
                wget \
                git \
                unzip
            ;;
    esac
    
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
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    echo -e "${GREEN}Compiling the application...${NC}"
    make -j$(nproc)
    
    # Return to temp directory
    cd ../..
    
    echo -e "${GREEN}Build completed successfully!${NC}"
    
    # Create installation directory
    if [ "$SYSTEM_WIDE_INSTALL" = true ]; then
        $SUDO_CMD mkdir -p "$INSTALL_DIR"
        $SUDO_CMD chown $USER:$USER "$INSTALL_DIR"
    else
        mkdir -p "$INSTALL_DIR"
    fi
    
    # Copy built binaries and resources
    if [ "$SYSTEM_WIDE_INSTALL" = true ]; then
        $SUDO_CMD cp -r dependencies/BTQ_Render_Engine/build/BTQuantTerminal "$INSTALL_DIR/"
        $SUDO_CMD cp -r dependencies/BTQ_Render_Engine/shaders "$INSTALL_DIR/"
        $SUDO_CMD cp -r dependencies/BTQ_Render_Engine/default_layout.json "$INSTALL_DIR/"
    else
        cp -r dependencies/BTQ_Render_Engine/build/BTQuantTerminal "$INSTALL_DIR/"
        cp -r dependencies/BTQ_Render_Engine/shaders "$INSTALL_DIR/"
        cp -r dependencies/BTQ_Render_Engine/default_layout.json "$INSTALL_DIR/"
    fi
    
    # Clean up
    cd /
    rm -rf "$TEMP_DIR"
    
    echo -e "${GREEN}Application copied to $INSTALL_DIR${NC}"
}

# Function to create desktop entry
create_desktop_entry() {
    echo -e "${GREEN}Creating desktop entry...${NC}"
    
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    
    # Create desktop file
    cat > "$DESKTOP_DIR/pubbtquant.desktop" << EOF
[Desktop Entry]
Name=PubBTQuant Trading Terminal
Comment=Advanced trading terminal with real-time analytics
Exec=$INSTALL_DIR/BTQuantTerminal
Icon=$INSTALL_DIR/pubbtquant.png
Terminal=false
Type=Application
Categories=Office;Finance;
StartupNotify=true
EOF
    
    # Create icon if it doesn't exist
    if [ ! -f "$INSTALL_DIR/pubbtquant.png" ]; then
        # Create a simple placeholder icon
        mkdir -p "$INSTALL_DIR"
        touch "$INSTALL_DIR/pubbtquant.png"
    fi
    
    # Update desktop database
    if command_exists update-desktop-database; then
        update-desktop-database "$HOME/.local/share/applications"
    fi
    
    echo -e "${GREEN}Desktop entry created successfully!${NC}"
}

# Function to add to PATH
add_to_path() {
    echo -e "${GREEN}Adding to PATH...${NC}"
    
    # Add to user's shell profile
    SHELL_PROFILE=""
    if [ -n "$BASH_VERSION" ] && [ -f "$HOME/.bashrc" ]; then
        SHELL_PROFILE="$HOME/.bashrc"
    elif [ -f "$HOME/.profile" ]; then
        SHELL_PROFILE="$HOME/.profile"
    elif [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    fi
    
    if [ -n "$SHELL_PROFILE" ]; then
        # Check if already added
        if ! grep -q "export PUBBTQUANT_PATH=$INSTALL_DIR" "$SHELL_PROFILE"; then
            echo "" >> "$SHELL_PROFILE"
            echo "# PubBTQuant Trading Terminal" >> "$SHELL_PROFILE"
            echo "export PUBBTQUANT_PATH=$INSTALL_DIR" >> "$SHELL_PROFILE"
            echo 'export PATH="$PUBBTQUANT_PATH:$PATH"' >> "$SHELL_PROFILE"
            echo -e "${GREEN}Added PubBTQuant to PATH in $SHELL_PROFILE${NC}"
        fi
    fi
}

# Main installation process
main() {
    echo -e "${BLUE}Starting PubBTQuant installation...${NC}"
    echo -e "${YELLOW}Installation directory: $INSTALL_DIR${NC}"
    if [ "$SYSTEM_WIDE_INSTALL" = true ]; then
        echo -e "${YELLOW}System-wide installation enabled${NC}"
    fi
    
    install_dependencies
    download_and_build
    create_desktop_entry
    add_to_path
    
    echo ""
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}To start PubBTQuant, run:${NC}"
    echo "  $INSTALL_DIR/BTQuantTerminal"
    echo ""
    echo -e "${BLUE}Or use the desktop shortcut if available.${NC}"
    echo ""
    echo -e "${YELLOW}Note: You may need to restart your shell or source your profile to update PATH.${NC}"
}

# Run main function
main