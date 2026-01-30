# Ralphy Agent Branch Iterator

This script iterates through all Git branches matching the pattern `ralphy/agent-*`.

## Features

- Identifies all local branches matching the pattern `ralphy/agent-*`
- Provides a count of total matching branches
- Offers an iteration function that can perform operations on each branch
- Handles both local and remote branches (with remote prefix removal for cleaner display)

## Usage

```bash
python list_agent_branches.py
```

This will list all branches matching the pattern and demonstrate the iteration functionality.

## Purpose

This script was created to systematically iterate through the 105 branches following the naming convention `ralphy/agent-{number}-{timestamp}-{description}`, which appear to be automated branches created for various development tasks related to volume analysis, footprint charts, and other trading interface features.