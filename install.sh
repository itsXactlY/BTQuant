#!/bin/bash

# Detect Linux distribution and install necessary development packages
install_dependencies() {
    echo "Detecting Linux distribution..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        case "$DISTRO" in
            ubuntu|debian)
                echo "Detected Ubuntu/Debian. Installing build-essential and other dependencies..."
                sudo apt-get update
                sudo apt-get install -y build-essential python3-dev unixodbc-dev git
                ;;
            fedora)
                echo "Detected Fedora. Installing development tools and dependencies..."
                sudo dnf groupinstall -y 'Development Tools'
                sudo dnf install -y python3-devel unixODBC-devel git
                ;;
            centos|rhel)
                echo "Detected CentOS/RHEL. Installing development tools and dependencies..."
                sudo yum groupinstall -y 'Development Tools'
                sudo yum install -y python3-devel unixODBC-devel git
                ;;
            arch|garuda|cachyos|manjaro|endeavouros)
                echo "Detected Arch Linux. Installing base-devel and other dependencies..."
                sudo pacman -Syu --noconfirm
                sudo pacman -S --noconfirm base-devel pybind11 unixodbc git tk
                ;;
            *)
                echo "Unsupported distribution: $DISTRO. Please install build-essential or equivalent manually."
                return 1
                ;;
        esac
    else
        echo "Cannot detect distribution. Please install the necessary dependencies manually."
        return 1
    fi
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if python3 and pip are installed
if ! command_exists python3 || ! command_exists pip; then
    echo "Python3 and pip are required. Installing Python3 and pip..."
    install_dependencies || exit 1
fi

# Clone the repository
git clone --recurse-submodules https://github.com/ItsXactlY/BTQuant BTQuant || {
    echo "Failed to clone repository"
    exit 1
}
cd BTQuant

# Create and activate virtual environment
python3 -m venv ../.btq || {
    echo "Failed to create virtual environment"
    exit 1
}
source ../.btq/bin/activate

# Get Python version inside the venv
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VERSION_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
PYTHON_VERSION_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
SITE_PACKAGES="../.btq/lib/python${PYTHON_VERSION}/site-packages"

# Upgrade pip and install necessary packages
pip install --upgrade pip setuptools wheel

# Install dependencies with error handling
cd dependencies
pip install . || {
    echo "Warning: Failed to install dependencies package"
}

cd quantstats_lumi_btquant
pip install . || {
    echo "Warning: Failed to install dependencies package"
}

cd ../..
echo "Installation complete. Activate the virtual environment with:"
echo "source .btq/bin/activate"
