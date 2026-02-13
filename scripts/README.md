# Clean & Build Script

This directory contains scripts for building the PubBTQuant project.

## clean_build.sh

This script performs a clean build of the PubBTQuant application by removing existing build artifacts and rebuilding the project from scratch.

### Features

- Complete cleanup of the build directory before building
- Support for both Debug and Release builds
- Optional sanitizer support for debugging
- Parallel compilation using multiple cores
- Comprehensive error handling

### Usage

```bash
# Basic usage (Debug build by default)
./scripts/clean_build.sh

# With custom build directory
./scripts/clean_build.sh -d my_build_dir

# Release build
./scripts/clean_build.sh --release

# Build with sanitizers enabled (for debugging)
./scripts/clean_build.sh --sanitize

# With custom number of parallel jobs
./scripts/clean_build.sh -j 8

# With verbose output
./scripts/clean_build.sh -v

# Build with custom source directory
./scripts/clean_build.sh -s /path/to/source

# Build with installation prefix
./scripts/clean_build.sh -i /usr/local

# Show help
./scripts/clean_build.sh -h
```

### Options

- `-d, --directory DIR`: Build directory (default: build)
- `-s, --source-dir DIR`: Source directory (default: .)
- `-i, --install-dir DIR`: Installation directory
- `-v, --verbose`: Verbose output
- `-j, --jobs N`: Number of parallel jobs (default: nproc)
- `--release`: Build in release mode
- `--sanitize`: Enable sanitizers for debugging
- `-h, --help`: Show help message

### Process

The script performs the following steps:

1. Cleans the build directory (removes all existing build artifacts)
2. Creates a fresh build directory if needed
3. Configures CMake with the specified build type and options
4. Compiles the project using multiple cores
5. Reports the size of built binaries