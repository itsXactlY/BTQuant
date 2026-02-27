# PubBTQuant Installers

This directory contains platform-specific installation packages and utilities for the PubBTQuant trading terminal.

## Directory Structure

```
installers/
├── linux/                 # Linux-specific installers
│   ├── installer.sh       # Main installation script
│   ├── CPackConfig.cmake  # CPack configuration for .deb/.rpm packages
│   └── pubbtquant.desktop.in  # Desktop entry template
├── windows/              # Windows-specific installers
│   └── installer.nsi     # NSIS installer script
├── macos/                # macOS-specific installers
│   ├── installer.sh      # Installation script
│   └── Info.plist        # Bundle information
├── dependency_manager.sh # Cross-platform dependency manager
├── setup_wizard.hpp      # First-run setup wizard (C++ header)
├── integration_example.hpp # Integration example for main app
├── test_installer.sh     # Installer functionality test script
└── INSTALL.md            # Installation guide
```

## Features

### Platform-Specific Installers
- **Linux**: Comprehensive shell script that handles dependencies, builds from source, and creates desktop entries
- **Windows**: NSIS-based installer with dependency checking and automatic downloads
- **macOS**: Shell script that creates application bundles and DMG installers

### Dependency Management
- Automatic detection and installation of required system dependencies
- Cross-platform compatibility
- Integration with native package managers (APT, Homebrew, etc.)

### First-Run Experience
- Integrated setup wizard using ImGui
- Guided configuration process
- Exchange and theme selection
- Data directory setup

### Package Creation
- CPack configuration for creating distribution packages (.deb, .rpm)
- DMG creation for macOS distribution
- Windows installer executable generation

## Usage

### For End Users
Simply run the appropriate installer for your platform:
- Linux: `./installers/linux/installer.sh`
- Windows: Compile and run `installers/windows/installer.nsi` with NSIS
- macOS: `./installers/macos/installer.sh`

### For Developers
The installer system can be extended by:
- Modifying the NSIS script for additional Windows functionality
- Updating the CPack configuration for different package options
- Enhancing the setup wizard with additional configuration steps

## Testing

Run the test suite to verify installer functionality:
```bash
./installers/test_installer.sh
```

## Building Distribution Packages

### For Linux (.deb/.rpm)
```bash
cd dependencies/BTQ_Render_Engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCPACK_PACKAGE_CONTACT="maintainer@example.com"
cpack -G DEB  # For Debian packages
cpack -G RPM  # For Red Hat packages
```

## Maintaining the Installers

When updating the application:
1. Update version numbers in installer scripts
2. Modify dependency lists if new libraries are added
3. Update the setup wizard if new configuration options are added
4. Test installers on each platform