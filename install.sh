#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if python3 and pip are installed
if ! command_exists python3 || ! command_exists pip; then
    echo "Python3 and pip are required. Please install them and try again."
    exit 1
fi

# Create and activate virtual environment
python3 -m venv .btq
source .btq/bin/activate

# Upgrade pip and install necessary packages
pip install --upgrade pip setuptools wheel

# Clone the repository
git clone https://github.com/itsXactlY/BTQuant/ BTQuant
cd BTQuant

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
