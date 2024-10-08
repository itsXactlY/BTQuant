#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

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
            arch)
                echo "Detected Arch Linux. Installing base-devel and other dependencies..."
                sudo pacman -Syu --noconfirm
                sudo pacman -S --noconfirm base-devel python-pybind11 unixodbc git
                ;;
            *)
                echo "Unsupported distribution: $DISTRO. Please install build-essential or equivalent manually."
                exit 1
                ;;
        esac
    else
        echo "Cannot detect distribution. Please install the necessary dependencies manually."
        exit 1
    fi
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
    install_dependencies
}

# Check if python3 and pip are installed
if ! command_exists python3 || ! command_exists pip; then
    echo "Python3 and pip are required. Installing Python3 and pip..."
fi

# Clone the repository
git clone https://github.com/itsXactlY/BTQuant/ BTQuant
cd BTQuant

# Create and activate virtual environment
python3 -m venv .btq
source .btq/bin/activate

# Upgrade pip and install necessary packages
pip install --upgrade pip setuptools wheel

# Install backtrader dependencies
cd dependencies
pip install .

# Install BTQ_Exchanges
cd BTQ_Exchanges
pip install .

# Install FastQuant
cd ../fastquant
pip install .

# Install and build fast_mssql
cd ../MsSQL
pip install .
python setup.py build_ext --inplace

cd ../..

echo "Installation complete. Activate the virtual environment with:"
echo "source .btq/bin/activate"