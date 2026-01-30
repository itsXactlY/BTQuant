#!/usr/bin/env python3
"""
Script to iterate through all branches matching the pattern 'ralphy/agent-*'
"""

import subprocess
import re


def get_matching_branches(local_only=False, remote_only=False):
    """Get all branches that match the pattern 'ralphy/agent-*'"""
    # Determine which branches to fetch
    cmd = ['git', 'branch', '-a']  # -a for all (local + remote)
    if local_only:
        cmd = ['git', 'branch']  # only local branches
    elif remote_only:
        cmd = ['git', 'branch', '-r']  # only remote branches

    # Get all branches
    result = subprocess.run(cmd, capture_output=True, text=True)
    branches = result.stdout.strip().split('\n')

    # Filter branches that match the pattern
    agent_branches = []
    for branch in branches:
        # Clean up the branch name (remove leading spaces and asterisk for current branch)
        branch = branch.strip()
        if branch.startswith('*'):
            branch = branch[1:].strip()

        # Check if it matches the pattern
        if re.match(r'(?:remotes/[^/]*/)?ralphy/agent-\d+', branch):
            # Remove the remote prefix for cleaner display
            clean_branch = re.sub(r'^remotes/[^/]*/', '', branch)
            if clean_branch:  # Only add non-empty branch names
                agent_branches.append(clean_branch)

    # Remove duplicates while preserving order
    seen = set()
    unique_branches = []
    for branch in agent_branches:
        if branch not in seen:
            seen.add(branch)
            unique_branches.append(branch)

    return unique_branches


def iterate_branches(operation=None):
    """
    Iterate through all matching branches, optionally performing an operation on each
    """
    branches = get_matching_branches()

    print(f"Iterating through {len(branches)} branches matching 'ralphy/agent-*':")
    print("=" * 60)

    for i, branch in enumerate(branches, 1):
        print(f"{i:3d}. {branch}")

        if operation:
            try:
                operation(branch)
            except Exception as e:
                print(f"     Error processing {branch}: {e}")

    print("=" * 60)
    print(f"Completed iteration through {len(branches)} branches.")


def main():
    print("Branches matching pattern 'ralphy/agent-*':")
    print("=" * 50)

    branches = get_matching_branches()

    for i, branch in enumerate(branches, 1):
        print(f"{i:3d}. {branch}")

    print(f"\nTotal count: {len(branches)} branches")

    # Also demonstrate iteration with no operation (just counting)
    print("\nDemonstrating iteration functionality:")
    iterate_branches()


if __name__ == "__main__":
    main()