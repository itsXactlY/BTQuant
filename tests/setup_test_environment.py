#!/usr/bin/env python3

"""
Test Environment Setup Script
This script sets up the virtual environment and installs all necessary dependencies for testing
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

def create_virtual_environment():
    """Create the virtual environment if it doesn't exist"""
    venv_path = Path.home() / ".btq"
    
    if not venv_path.exists():
        print(f"Creating virtual environment at {venv_path}")
        venv.create(venv_path, with_pip=True)
        print("Virtual environment created successfully")
    else:
        print(f"Virtual environment already exists at {venv_path}")
    
    return venv_path

def install_dependencies(venv_path):
    """Install all necessary dependencies in the virtual environment"""
    
    # Determine the correct pip executable
    pip_executable = venv_path / "bin" / "pip"
    python_executable = venv_path / "bin" / "python"
    
    # Install the backtrader package from dependencies in development mode
    print("Installing backtrader package from dependencies...")
    result = subprocess.run([
        str(pip_executable), "install", "-e", "../dependencies"
    ], cwd=Path(__file__).parent, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error installing backtrader: {result.stderr}")
        return False
    
    # Install development dependencies
    print("Installing development dependencies...")
    result = subprocess.run([
        str(pip_executable), "install", "-r", "../requirements-dev.txt"
    ], cwd=Path(__file__).parent, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error installing development dependencies: {result.stderr}")
        return False
    
    print("All dependencies installed successfully")
    return True

def verify_installation(venv_path):
    """Verify that the installation was successful"""
    python_executable = venv_path / "bin" / "python"
    
    print("Verifying installation...")
    
    # Test backtrader import
    result = subprocess.run([
        str(python_executable), "-c", "import backtrader; print(f'Backtrader version: {backtrader.__version__}')"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error importing backtrader: {result.stderr}")
        return False
    
    print(result.stdout.strip())
    
    # Test pytest import
    result = subprocess.run([
        str(python_executable), "-c", "import pytest; print(f'Pytest version: {pytest.__version__}')"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error importing pytest: {result.stderr}")
        return False
    
    print(result.stdout.strip())
    
    print("Installation verification successful")
    return True

def main():
    """Main function to set up the test environment"""
    print("Setting up test environment...")
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    
    # Install dependencies
    if not install_dependencies(venv_path):
        print("Failed to install dependencies")
        return False
    
    # Verify installation
    if not verify_installation(venv_path):
        print("Failed to verify installation")
        return False
    
    print("\nTest environment setup complete!")
    print(f"Virtual environment location: {venv_path}")
    print(f"To activate the environment: source {venv_path}/bin/activate")
    print(f"To run tests: pytest")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)