# Build Scripts

This directory contains scripts for building the PubBTQuant project.

## release_build.sh

This script creates an optimized release build of the PubBTQuant application with the following features:

- Optimized compilation with `-O3`, `-march=native`, and `-flto` flags
- Debug symbol stripping to reduce binary size
- Link-time optimization and section removal
- Support for building only the BTQ_Render_Engine component

### Usage

```bash
# Basic usage
./release_build.sh

# With custom build directory
./release_build.sh -d my_build_dir

# Build only the BTQ_Render_Engine
./release_build.sh -e

# Clean build (removes previous build artifacts)
./release_build.sh -c

# With verbose output
./release_build.sh -v

# Specify number of parallel jobs
./release_build.sh -j 8

# Build with custom source directory
./release_build.sh -s /path/to/source

# Build with installation prefix
./release_build.sh -i /usr/local

# Show help
./release_build.sh -h
```

### Options

- `-d, --directory DIR`: Build directory (default: build_release)
- `-s, --source-dir DIR`: Source directory (default: .)
- `-i, --install-dir DIR`: Installation directory
- `-c, --clean`: Perform clean build
- `-v, --verbose`: Verbose output
- `-j, --jobs N`: Number of parallel jobs (default: nproc)
- `-e, --engine-only`: Build only the BTQ_Render_Engine
- `-h, --help`: Show help message

### Optimization Features

The script applies several optimization techniques:

1. **Compiler Optimizations**:
   - `-O3`: Aggressive optimization level
   - `-march=native`: Optimize for the host CPU
   - `-flto`: Link-time optimization
   - `-ffunction-sections -fdata-sections`: Enable section-level garbage collection

2. **Linker Optimizations**:
   - `--gc-sections`: Remove unused sections
   - `-strip-all`: Remove all symbol and debugging information
   - `--exclude-libs,ALL`: Exclude library symbols
   - `--as-needed`: Only link libraries that are actually needed

3. **Additional Size Reduction**:
   - Strips debug symbols from binaries
   - Optional UPX compression if available

## clean_build.sh

This script performs a complete clean build by removing all build artifacts and rebuilding from scratch. It ensures a completely fresh build environment.

### Usage

```bash
# Basic usage (creates a fresh 'build' directory)
./clean_build.sh

# With custom build directory
./clean_build.sh -d my_clean_build

# Build in debug mode
./clean_build.sh --debug

# Build in release mode
./clean_build.sh --release

# With custom number of parallel jobs
./clean_build.sh -j 4

# Build only the BTQ_Render_Engine
./clean_build.sh -e

# Show help
./clean_build.sh -h
```

### Options

- `-d, --directory DIR`: Build directory (default: build)
- `-r, --release-directory DIR`: Release build directory (default: build_release)
- `-s, --source-dir DIR`: Source directory (default: .)
- `-v, --verbose`: Verbose output
- `-j, --jobs N`: Number of parallel jobs (default: nproc)
- `--debug`: Build in debug mode
- `--release`: Build in release mode (default)
- `-e, --engine-only`: Build only the BTQ_Render_Engine
- `-h, --help`: Show help message

### Features

The clean build script ensures:

1. **Complete Clean State**:
   - Removes all existing build artifacts
   - Creates a fresh build directory
   - Ensures no cached configurations interfere with the build

2. **Consistent Build Process**:
   - Follows the same configuration patterns as other build scripts
   - Maintains consistent output formatting and error handling
   - Supports the same build options and parameters