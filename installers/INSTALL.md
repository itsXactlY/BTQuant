# PubBTQuant Installation Guide

This document describes how to install the PubBTQuant trading terminal on different platforms using the provided installer system.

## Overview

The PubBTQuant installer system provides platform-specific installation solutions:

- **Linux**: Shell script installer with support for .deb/.rpm packages via CPack
- **Windows**: NSIS-based installer with dependency checking
- **macOS**: Shell script installer with DMG creation

## Prerequisites

Before installing PubBTQuant, ensure your system meets the following requirements:

### System Requirements
- **Operating System**: 
  - Linux (Ubuntu 20.04+, Fedora 35+, Arch Linux, or equivalent)
  - Windows 10/11 (64-bit)
  - macOS 10.15+ (Catalina or later)
- **CPU**: Intel/AMD x86-64 processor with SSE4.1 support
- **RAM**: Minimum 8 GB, recommended 16 GB or more
- **GPU**: DirectX 12 / Vulkan compatible graphics card with 2 GB VRAM minimum
- **Storage**: At least 2 GB of free space

### Software Dependencies
- **Linux**: Vulkan SDK, CMake 3.15+, GCC 11+ or Clang 12+
- **Windows**: Visual Studio 2019+ or Build Tools, Vulkan SDK
- **macOS**: Xcode Command Line Tools, Vulkan SDK

## Installation Methods

### Linux Installation

#### Method 1: Using the Shell Script Installer
```bash
cd /path/to/PubBTQuant/installers/linux
chmod +x installer.sh
./installer.sh
```

The installer will:
1. Check for and install required dependencies
2. Download and build the application
3. Create desktop entries
4. Add the application to your PATH

#### Method 2: Creating Distribution Packages
For creating .deb or .rpm packages, use CPack:

```bash
cd dependencies/BTQ_Render_Engine
mkdir build && cd build
cmake ..
make
cpack -G DEB  # For Debian/Ubuntu
# or
cpack -G RPM  # For Red Hat/Fedora
```

### Windows Installation

#### Method 1: Using the NSIS Installer
1. Install NSIS (Nullsoft Scriptable Install System)
2. Compile the installer script:
   ```
   makensis installers\windows\installer.nsi
   ```
3. Run the generated `PubBTQuant-Setup.exe`

#### Method 2: Manual Installation
1. Install dependencies manually (Vulkan SDK, Visual C++ Redistributables)
2. Build the application using CMake
3. Place binaries in the desired installation directory

### macOS Installation

#### Using the Shell Script Installer
```bash
cd /path/to/PubBTQuant/installers/macos
chmod +x installer.sh
./installer.sh
```

The installer will:
1. Install dependencies via Homebrew
2. Build the application
3. Create an application bundle in `/Applications`
4. Generate a DMG for distribution

## First-Run Setup Wizard

On the first run, PubBTQuant will present a setup wizard that guides you through:

1. **Welcome**: Introduction to the application
2. **License Agreement**: Acceptance of the MIT license
3. **Data Directory**: Choose where to store market data and settings
4. **Exchange Setup**: Select and configure exchange connections
5. **Theme Selection**: Choose your preferred visual theme
6. **Completion**: Final confirmation and launch

## Configuration

After installation, you can customize PubBTQuant by:

- Editing the configuration file located at `~/.pubbtquant/config.json` (Linux/macOS) or `%APPDATA%\PubBTQuant\config.json` (Windows)
- Using the in-application settings panel
- Modifying the default layout in `default_layout.json`

## Troubleshooting

### Common Issues

**Vulkan Not Available**: Ensure your graphics drivers are up to date and Vulkan is properly installed.

**Missing Dependencies**: The installer should handle most dependencies, but on some systems you may need to install additional packages manually.

**Build Failures**: Ensure you have sufficient disk space and memory. Check that all prerequisites are met.

### Support

For additional support, please check:
- The project documentation in the `docs/` directory
- The GitHub issues page
- The community forums

## Uninstallation

### Linux
Remove the installation directory and desktop entries:
```bash
rm -rf ~/.pubbtquant
rm ~/.local/share/applications/pubbtquant.desktop
```

### Windows
Use the uninstaller from the Start Menu or Control Panel.

### macOS
Drag the application from `/Applications` to the Trash, then remove configuration files:
```bash
rm -rf ~/Library/Application\ Support/PubBTQuant
```

## Building from Source

If you prefer to build from source without using the installer:

```bash
cd dependencies/BTQ_Render_Engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # On Linux/macOS
# or
make -j%NUMBER_OF_PROCESSORS%  # On Windows with MSYS2
```

## Version Information

- **Current Version**: 1.0.0
- **Last Updated**: February 2026
- **License**: MIT

For the latest updates and releases, visit the project repository.