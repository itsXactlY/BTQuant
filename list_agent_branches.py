#!/usr/bin/env python3
"""
Script to iterate through all git branches with the pattern 'ralphy/agent-*'
"""

import subprocess
import re


def get_agent_branches():
    """
    Get all git branches that match the pattern 'ralphy/agent-*'

    Returns:
        list: A list of branch names matching the pattern
    """
    try:
        # Get all branch names
        result = subprocess.run(['git', 'branch', '-a'],
                                capture_output=True,
                                text=True,
                                check=True)

        # Split the output into lines and extract branch names
        branches = []
        for line in result.stdout.strip().split('\n'):
            # Remove leading spaces and asterisk (current branch indicator)
            branch = line.strip()
            if branch.startswith('*'):
                branch = branch[1:].strip()

            # Check if the branch matches the pattern
            if re.match(r'^ralphy/agent-', branch):
                branches.append(branch)

        return branches

    except subprocess.CalledProcessError:
        # Don't print error in this context as it's handled by check=True
        # But we still want to catch and return empty list
        return []
    except FileNotFoundError:
        # Git is not installed or not in PATH
        return []


def main():
    """Main function to iterate through and print agent branches"""
    print("Finding branches with pattern 'ralphy/agent-*':")
    print("=" * 50)

    agent_branches = get_agent_branches()

    if not agent_branches:
        print("No branches found matching the pattern 'ralphy/agent-*'")
        return

    print(f"Found {len(agent_branches)} branches:")
    for i, branch in enumerate(agent_branches, 1):
        print(f"{i:2d}. {branch}")

    print("=" * 50)
    print(f"Total: {len(agent_branches)} branches")


if __name__ == "__main__":
    main()
