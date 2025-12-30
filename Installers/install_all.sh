#!/bin/bash

# BTQuant Complete Installation Script
# Installs everything: BTQuant, MSSQL, Fast_MSSQL Driver, CCAPI builds

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
TEMP_DIR=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        log_info "Detected distribution: $DISTRO"
    else
        log_error "Cannot detect distribution. Please install dependencies manually."
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."

    case "$DISTRO" in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                python3-dev \
                python3-venv \
                python3-pybind11 \
                unixodbc-dev \
                git \
                cmake \
                libssl-dev \
                libboost-all-dev \
                rapidjson-dev \
                curl \
                wget \
                gnupg2 \
                software-properties-common \
                lsb-release
            ;;
        fedora)
            sudo dnf groupinstall -y 'Development Tools'
            sudo dnf install -y \
                python3-devel \
                python3-pybind11 \
                unixODBC-devel \
                git \
                cmake \
                openssl-devel \
                boost-devel \
                rapidjson-devel \
                curl \
                wget \
                gnupg2
            ;;
        centos|rhel)
            sudo yum groupinstall -y 'Development Tools'
            sudo yum install -y \
                python3-devel \
                python3-pybind11 \
                unixODBC-devel \
                git \
                cmake \
                openssl-devel \
                boost-devel \
                rapidjson-devel \
                curl \
                wget \
                gnupg2
            ;;
        arch|manjaro|endeavouros|garuda)
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                base-devel \
                python \
                pybind11 \
                unixodbc \
                git \
                cmake \
                openssl \
                boost \
                rapidjson \
                curl \
                wget \
                gnupg
            ;;
        *)
            log_error "Unsupported distribution: $DISTRO"
            log_error "Please install build-essential, cmake, boost, openssl, and python3-dev manually."
            exit 1
            ;;
    esac

    log_success "System dependencies installed"
}

# Install Microsoft SQL Server
install_mssql() {
    # Check if MSSQL is already installed and running
    if systemctl is-active --quiet mssql-server 2>/dev/null; then
        log_info "Microsoft SQL Server is already installed and running. Skipping installation."
        return 0
    fi

    log_info "Installing Microsoft SQL Server..."

    case "$DISTRO" in
        ubuntu|debian)
            # Import Microsoft GPG key
            curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -

            # Add Microsoft SQL Server repository
            sudo add-apt-repository "$(curl -sSL https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/mssql-server-2022.list)"

            # Update and install
            sudo apt-get update
            sudo apt-get install -y mssql-server

            # Install ODBC driver
            curl -sSL https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
            sudo apt-get update
            sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18
            ;;
        fedora)
            sudo curl -o /etc/yum.repos.d/mssql-server.repo https://packages.microsoft.com/config/rhel/8/mssql-server-2022.repo
            sudo dnf install -y mssql-server
            sudo curl -o /etc/yum.repos.d/msprod.repo https://packages.microsoft.com/config/rhel/8/prod.repo
            sudo ACCEPT_EULA=Y dnf install -y msodbcsql18
            ;;
        centos|rhel)
            sudo curl -o /etc/yum.repos.d/mssql-server.repo https://packages.microsoft.com/config/rhel/8/mssql-server-2022.repo
            sudo yum install -y mssql-server
            sudo curl -o /etc/yum.repos.d/msprod.repo https://packages.microsoft.com/config/rhel/8/prod.repo
            sudo ACCEPT_EULA=Y yum install -y msodbcsql18
            ;;
        arch|manjaro|endeavouros|garuda)
            # Use AUR helper (assuming yay or paru is available)
            if command -v yay >/dev/null 2>&1; then
                AUR_HELPER="yay"
            elif command -v paru >/dev/null 2>&1; then
                AUR_HELPER="paru"
            else
                log_error "No AUR helper found. Please install yay or paru first."
                exit 1
            fi

            $AUR_HELPER -S --noconfirm mssql-server
            $AUR_HELPER -S --noconfirm msodbcsql
            ;;
        *)
            log_error "MSSQL installation not supported for $DISTRO"
            exit 1
            ;;
    esac

    log_success "Microsoft SQL Server installed"
}

# Configure MSSQL
configure_mssql() {
    # Check if MSSQL is already configured by testing connection
    if /opt/mssql-tools*/bin/sqlcmd -S localhost -U sa -P "q?}33YIToo:H%xue$Kr*" -C -Q "SELECT @@VERSION" >/dev/null 2>&1; then
        log_info "Microsoft SQL Server is already configured. Skipping configuration."
        return 0
    fi

    log_info "Configuring Microsoft SQL Server..."

    # Set SA password (using the same as in init_database.py)
    MSSQL_SA_PASSWORD="q?}33YIToo:H%xue$Kr*"

    case "$DISTRO" in
        ubuntu|debian|fedora|centos|rhel)
            sudo MSSQL_SA_PASSWORD="$MSSQL_SA_PASSWORD" MSSQL_PID='evaluation' /opt/mssql/bin/mssql-conf -n setup accept-eula
            ;;
        arch|manjaro|endeavouros|garuda)
            sudo systemctl enable mssql-server
            sudo systemctl start mssql-server
            sleep 10

            # Configure with sqlcmd (add TrustServerCertificate=yes)
            /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "
            ALTER LOGIN sa ENABLE;
            GO
            ALTER LOGIN sa WITH PASSWORD = '$MSSQL_SA_PASSWORD';
            GO"
            ;;
    esac

    # Wait for SQL Server to start
    log_info "Waiting for SQL Server to start..."
    sleep 30

    # Enable and start service
    sudo systemctl enable mssql-server
    sudo systemctl start mssql-server

    log_success "Microsoft SQL Server configured"
}

# Install sqlcmd tools
install_sqlcmd() {
    # Check if sqlcmd is already available
    if command -v sqlcmd >/dev/null 2>&1; then
        log_info "sqlcmd tools are already installed. Skipping."
        return 0
    fi

    log_info "Installing sqlcmd tools..."

    case "$DISTRO" in
        ubuntu|debian)
            sudo apt-get install -y mssql-tools18
            echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
            export PATH="$PATH:/opt/mssql-tools18/bin"
            ;;
        fedora|centos|rhel)
            sudo ACCEPT_EULA=Y yum install -y mssql-tools18
            echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
            export PATH="$PATH:/opt/mssql-tools18/bin"
            ;;
        arch|manjaro|endeavouros|garuda)
            $AUR_HELPER -S --noconfirm mssql-tools
            echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
            export PATH="$PATH:/opt/mssql-tools/bin"
            ;;
    esac

    log_success "sqlcmd tools installed"
}

# Clone and setup BTQuant
setup_btquant() {
    log_info "Setting up BTQuant..."

    # Clone repository directly (prototype branch)
    if [ ! -d "BTQuant" ]; then
        git clone --recurse-submodules -b prototyping https://github.com/ItsXactlY/BTQuant BTQuant
    fi
    cd BTQuant

    # Create virtual environment in home directory
    python3 -m venv "$HOME/.btq"
    source "$HOME/.btq/bin/activate"

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    # Install pybind11 in venv
    pip install pybind11

    # Install dependencies
    cd dependencies
    pip install .
    cd ..

    # Return to original directory
    cd ..

    log_success "BTQuant setup complete"
}

# Build Fast_MSSQL driver
build_fast_mssql() {
    log_info "Installing Fast_MSSQL driver..."

    source "$HOME/.btq/bin/activate"

    # Check Python version to select correct .so file
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

    # Get the correct site-packages directory
    PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

    # Copy the pre-compiled .so file to venv
    SO_FILE="BTQuant/dependencies/backtrader/feeds/mssql/fast_mssql.cpython-${PYTHON_VERSION}-x86_64-linux-gnu.so"
    if [ -f "$SO_FILE" ]; then
        cp "$SO_FILE" "$PYTHON_SITE_PACKAGES/"
        log_success "Fast_MSSQL driver installed from pre-compiled binary"
    else
        log_warning "Pre-compiled Fast_MSSQL binary not found for Python ${PYTHON_VERSION}. Attempting build..."
        # Fallback to building
        cd "BTQuant/dependencies/MsSQL"
        python -m pip install .
        cd ../..
        log_success "Fast_MSSQL driver built and installed"
    fi
}

# Initialize database
init_database() {
    log_info "Initializing database..."

    source "$HOME/.btq/bin/activate"

    # Check if database already exists
    if python3 -c "
import pyodbc
try:
    conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=BTQ_MarketData;UID=SA;PWD=q?}33YIToo:H%xue\$Kr*;TrustServerCertificate=yes;')
    conn.close()
    print('EXISTS')
except:
    print('NOT_EXISTS')
" | grep -q 'EXISTS'; then
        log_info "Database BTQ_MarketData already exists. Skipping initialization."
        return 0
    fi

    # Run the database initialization script
    python3 "BTQuant/dependencies/datacollector/init_database.py"

    log_success "Database initialized"
}

# Build CCAPI
build_ccapi() {
    log_info "Preparing CCAPI..."

    cd "BTQuant/dependencies/ccapi"

    # 🔑 THIS IS THE MISSING STEP
    log_info "Initializing CCAPI submodules..."
    git submodule update --init --recursive

    # Optional bootstrap if present
    if [ -f scripts/bootstrap.sh ]; then
        log_info "Running CCAPI bootstrap..."
        bash scripts/bootstrap.sh
    fi

    log_info "Building CCAPI example..."

    cd example
    mkdir -p build
    cd build
    rm -rf *

    cmake ..
    cmake --build .

    mkdir -p "$HOME/bin"
    cp market_data_collector "$HOME/bin/" 2>/dev/null || true
    cp hotspine "$HOME/bin/" 2>/dev/null || true

    log_success "CCAPI built successfully"
}
# Cleanup temp files
cleanup() {
    log_info "Cleaning up temporary files..."
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        log_success "Temporary files cleaned up"
    fi
}

# Main installation function
main() {
    log_info "Starting BTQuant complete installation..."

    # Set trap to cleanup on exit
    trap cleanup EXIT

    detect_distro
    install_system_deps

    # Install and configure MSSQL (skip both if already installed)
    MSSQL_INSTALLED=false
    if ! systemctl is-active --quiet mssql-server 2>/dev/null; then
        install_mssql
        MSSQL_INSTALLED=true
    else
        log_info "Microsoft SQL Server is already installed and running. Skipping installation and configuration."
    fi

    if [ "$MSSQL_INSTALLED" = true ]; then
        configure_mssql
    fi

    install_sqlcmd
    setup_btquant
    build_fast_mssql
    init_database
    build_ccapi

    log_success "🎉 BTQuant installation complete!"
    log_info ""
    log_info "To activate the virtual environment:"
    log_info "  source ~/.btq/bin/activate"
    log_info ""
    log_info "MSSQL Server is running. SA password: q?}33YIToo:H%xue$Kr*"
    log_info "Database: BTQ_MarketData on localhost"
    log_info ""
    log_info "CCAPI binaries are available in ~/bin/"
    log_info "Make sure to source your ~/.bashrc or restart your shell to update PATH"
}

# Run main function
main "$@"