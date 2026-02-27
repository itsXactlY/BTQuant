#!/bin/bash

# Dependency Manager for PubBTQuant
# This script handles dependency installation across all platforms

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Linux dependency checker and installer
install_linux_deps() {
    echo -e "${GREEN}Checking Linux dependencies...${NC}"
    
    # Detect package manager
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
    
    # Check for required packages
    MISSING_PKGS=()
    
    # Check for build essentials
    if ! command_exists gcc || ! command_exists g++; then
        MISSING_PKGS+=("build-essential")
    fi
    
    # Check for CMake
    if ! command_exists cmake; then
        MISSING_PKGS+=("cmake")
    fi
    
    # Check for Vulkan
    if ! command_exists vkinfo && ! command_exists vulkaninfo; then
        MISSING_PKGS+=("vulkan-sdk")
    fi
    
    # Check for OpenGL/GLFW
    if ! pkg-config --exists glfw3 2>/dev/null; then
        MISSING_PKGS+=("libglfw3-dev")
    fi
    
    # Check for TBB
    if ! pkg-config --exists tbb 2>/dev/null; then
        MISSING_PKGS+=("libtbb-dev")
    fi
    
    # Install missing packages
    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        echo -e "${YELLOW}Missing packages: ${MISSING_PKGS[*]}${NC}"
        
        case $PKG_MANAGER in
            apt)
                echo -e "${GREEN}Installing missing packages via apt...${NC}"
                sudo apt update
                sudo apt install -y "${MISSING_PKGS[@]}"
                ;;
            dnf|yum)
                echo -e "${GREEN}Installing missing packages via $PKG_MANAGER...${NC}"
                sudo $PKG_MANAGER install -y "${MISSING_PKGS[@]}"
                ;;
            pacman)
                echo -e "${GREEN}Installing missing packages via pacman...${NC}"
                sudo pacman -Sy --noconfirm "${MISSING_PKGS[@]}"
                ;;
            zypper)
                echo -e "${GREEN}Installing missing packages via zypper...${NC}"
                sudo zypper refresh
                sudo zypper install -y "${MISSING_PKGS[@]}"
                ;;
        esac
        
        echo -e "${GREEN}Dependencies installed successfully!${NC}"
    else
        echo -e "${GREEN}All required dependencies are already installed.${NC}"
    fi
}

# macOS dependency checker and installer
install_macos_deps() {
    echo -e "${GREEN}Checking macOS dependencies...${NC}"
    
    # Check for Homebrew
    if ! command_exists brew; then
        echo -e "${RED}Homebrew is required but not installed.${NC}"
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
    
    # Check for required packages
    MISSING_PKGS=()
    
    # Check for CMake
    if ! command_exists cmake; then
        MISSING_PKGS+=("cmake")
    fi
    
    # Check for Vulkan SDK
    if [ ! -d "/usr/local/share/vulkan" ] && [ ! -d "/opt/homebrew/share/vulkan" ]; then
        echo -e "${YELLOW}Vulkan SDK may need manual installation. Please visit https://vulkan.lunarg.com/sdk/home#mac to download.${NC}"
    fi
    
    # Check for GLFW
    if ! brew list glfw &>/dev/null; then
        MISSING_PKGS+=("glfw")
    fi
    
    # Check for TBB
    if ! brew list tbb &>/dev/null; then
        MISSING_PKGS+=("tbb")
    fi
    
    # Install missing packages
    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        echo -e "${YELLOW}Missing packages: ${MISSING_PKGS[*]}${NC}"
        echo -e "${GREEN}Installing missing packages via Homebrew...${NC}"
        brew install "${MISSING_PKGS[@]}"
        echo -e "${GREEN}Dependencies installed successfully!${NC}"
    else
        echo -e "${GREEN}All required dependencies are already installed.${NC}"
    fi
}

# Windows dependency checker and installer
install_windows_deps() {
    echo -e "${GREEN}Checking Windows dependencies...${NC}"
    
    # On Windows, we typically use PowerShell or external installers
    # This script would be run from within the installer
    
    # Check for Visual Studio Build Tools or Visual Studio
    if ! command_exists cl && ! command_exists msbuild; then
        echo -e "${YELLOW}Visual Studio Build Tools or Visual Studio is required.${NC}"
        echo -e "${YELLOW}Please install Visual Studio Community or Build Tools.${NC}"
    fi
    
    # Check for CMake
    if ! command_exists cmake; then
        echo -e "${YELLOW}CMake is required but not found. Installing...${NC}"
        choco install cmake -y
    fi
    
    # Check for Git
    if ! command_exists git; then
        echo -e "${YELLOW}Git is required but not found. Installing...${NC}"
        choco install git -y
    fi
    
    # Check for Vulkan SDK
    VULKAN_SDK_REG="HKLM:\\SOFTWARE\\Khronos\\Vulkan\\Drivers"
    if [ ! -d "/c/Program Files/Vulkan SDK" ]; then
        echo -e "${YELLOW}Vulkan SDK is required but not found. Please install from https://vulkan.lunarg.com/sdk/home#windows${NC}"
    fi
    
    # Check for TBB
    if ! command_exists tbb; then
        echo -e "${YELLOW}Intel TBB is required. Installing...${NC}"
        choco install intel-tbb -y
    fi
    
    echo -e "${GREEN}Dependency check completed.${NC}"
}

# Main function
main() {
    echo -e "${BLUE}PubBTQuant Dependency Manager${NC}"
    echo "============================="
    echo -e "${YELLOW}Detected OS: $OS${NC}"
    
    case $OS in
        "linux")
            install_linux_deps
            ;;
        "macos")
            install_macos_deps
            ;;
        "windows")
            install_windows_deps
            ;;
        *)
            echo -e "${RED}Unsupported operating system: $OS${NC}"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}Dependency management completed.${NC}"
}

# Run main function
main