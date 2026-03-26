#!/bin/bash

# Script to generate API documentation from code comments using Doxygen

set -e  # Exit immediately if a command exits with a non-zero status

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Generating API documentation from code comments...${NC}"

# Check if doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo -e "${RED}Error: doxygen is not installed${NC}" >&2
    echo "Please install doxygen to generate documentation:"
    echo "  Ubuntu/Debian: sudo apt-get install doxygen"
    echo "  macOS: brew install doxygen"
    echo "  CentOS/RHEL: sudo yum install doxygen"
    exit 1
fi

# Go to project root directory where Doxyfile is located
cd "$(dirname "$0")/.."

# Create docs/api_reference directory if it doesn't exist
mkdir -p docs/api_reference

# Generate documentation using the Doxyfile
echo -e "${YELLOW}Running doxygen to generate API documentation...${NC}"
doxygen Doxyfile

# Check if documentation was generated successfully
if [ -d "docs/api_reference/html" ] && [ "$(ls -A docs/api_reference/html)" ]; then
    echo -e "${GREEN}✅ API documentation generated successfully!${NC}"
    echo -e "${GREEN}📁 Documentation location: docs/api_reference/html/${NC}"
    echo ""
    echo -e "${BLUE}Generated documentation includes:${NC}"
    echo "  • Class hierarchies"
    echo "  • Detailed member documentation"
    echo "  • File structure"
    echo "  • Inline code examples"
    echo ""
    echo -e "${BLUE}To view the documentation, open:${NC}"
    echo "  docs/api_reference/html/index.html"
else
    echo -e "${RED}❌ Failed to generate documentation${NC}" >&2
    exit 1
fi