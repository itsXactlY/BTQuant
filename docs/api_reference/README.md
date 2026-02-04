# API Documentation

Automatically generated documentation from code comments using Doxygen.

## Overview

This directory contains the automatically generated API documentation for the PubBTQuant Trading Engine. The documentation is generated from source code comments using Doxygen and includes:

- Class hierarchies
- Detailed member documentation
- File structure
- Inline code examples

## Generation Process

The API documentation is automatically generated using the following command:

```bash
./scripts/generate_docs.sh
```

This script:
1. Runs Doxygen with the configuration in the root `Doxyfile`
2. Processes all header files in `dependencies/BTQ_Render_Engine/include/`
3. Excludes ImGui/ImPlot files as they are external dependencies
4. Generates HTML documentation in `docs/api_reference/html/`

## Documentation Structure

The generated documentation includes:
- **Classes**: Complete class hierarchies with inheritance relationships
- **Functions**: Detailed function signatures, parameters, and return values
- **Variables**: Member variables with type and description
- **Enums**: Enumeration types with possible values
- **Structs**: Data structure definitions

## Updating Documentation

To regenerate the documentation after code changes:

1. Run the generation script:
   ```bash
   ./scripts/generate_docs.sh
   ```

2. The documentation will be updated in `docs/api_reference/html/`

## Doxygen Configuration

The documentation generation is controlled by the `Doxyfile` in the project root, which is configured to:
- Extract documentation from C++ header files
- Generate HTML output
- Include detailed member documentation
- Create class hierarchies and collaboration diagrams
- Process code comments in standard Doxygen format

## Viewing Documentation

Open `docs/api_reference/html/index.html` in a web browser to view the complete API documentation.