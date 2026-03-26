---
Systematic approach to understanding complex software architectures by examining code file-by-file, tracing dependencies, and building a complete mental model of how all components interact to understand the entire system's purpose, architecture, and functionality.

**When to use**: When you need to understand a large, complex codebase quickly and thoroughly, especially systems with multiple languages, layers, and sophisticated architectures.

**How to apply**: 
1. Start with documentation (README, architect) to understand purpose and scope
2. Examine directory structure to understand organization
3. Analyze key files in each layer (data collection, processing, strategy, execution)
4. Trace dependencies and data flow between components
5. Look for patterns, abstractions, and reusable designs
6. Identify integration points and interfaces between layers
7. Build mental model of data flow and control flow
8. Validate understanding by tracing specific use cases through the system

**Key techniques**:
- File-by-file examination with attention to comments and structure
- Dependency tracing (imports, includes, function calls)
- Pattern recognition (repeated designs, abstractions)
- Data flow analysis (how information moves through system)
- Control flow analysis (how decisions and execution progress)
- Interface identification (APIs, message passing, shared memory)
- Use case tracing (following specific scenarios through complete system)

**Output**: Complete understanding of system architecture, components, data flow, and design principles enabling effective modification, extension, or troubleshooting.
---