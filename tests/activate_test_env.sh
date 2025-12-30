#!/bin/bash

# Test Environment Activation Script
# This script sets up and activates the virtual environment for testing

echo "Setting up test environment..."

# Check if virtual environment exists
if [ ! -d "$HOME/.btq" ]; then
    echo "Creating virtual environment at $HOME/.btq"
    python3 -m venv "$HOME/.btq"
fi

# Activate the virtual environment
source "$HOME/.btq/bin/activate"

# Check if dependencies are installed
if ! python -c "import backtrader" 2>/dev/null; then
    echo "Installing dependencies..."
    
    # Install the backtrader package from dependencies
    cd dependencies
    pip install -e .
    cd ..
    
    # Install development dependencies
    pip install -r requirements-dev.txt
    
    echo "Dependencies installed successfully"
else
    echo "Dependencies already installed"
fi

echo "Test environment ready"
echo "Virtual environment location: $HOME/.btq"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"