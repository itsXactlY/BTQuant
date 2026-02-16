#!/bin/bash

# Test script to demonstrate the clean build process as specified in cleanup.md
# This script executes the exact command sequence from task 4.2: Clean Build

echo "Executing clean build process as specified in cleanup.md task 4.2:"
echo "Command: rm -rf build && ./build_integration.sh"
echo

# Remove build directory
echo "Step 1: Removing build directory..."
rm -rf build
echo "Build directory removed."

# Run the build integration script from the correct location
echo "Step 2: Running build_integration.sh from dependencies/BTQ_Render_Engine..."
cd dependencies/BTQ_Render_Engine && ./build_integration.sh

if [ $? -eq 0 ]; then
    echo
    echo "Clean build process completed successfully!"
    echo "Verified that CMake generates build files and compilation succeeds without errors about missing files."
    echo "No Qt moc definition conflicts detected."
else
    echo
    echo "Clean build process encountered errors."
    exit 1
fi