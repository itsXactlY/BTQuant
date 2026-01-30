#!/usr/bin/env python3
"""
Script to iterate through all git branches with the pattern 'ralphy/agent-*'
"""

import subprocess
import re


def get_agent_branches():
    """Get all branches that match the pattern 'ralphy/agent-*'"""
    try:
        # Get all branches
        result = subprocess.run(['git', 'branch', '-a'], 
                              capture_output=True, text=True, check=True)
        
        # Extract branch names and filter for ralphy/agent-* pattern
        branches = []
        for line in result.stdout.split('\n'):
            # Remove leading/trailing whitespace and asterisk for current branch
            branch = line.strip().lstrip('* ')
            if branch.startswith('ralphy/agent-'):
                branches.append(branch)
                
        return branches
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}")
        return []


def main():
    """Main function to iterate through and display agent branches"""
    print("Finding branches with pattern 'ralphy/agent-*'...")

    agent_branches = get_agent_branches()

    if not agent_branches:
        print("No branches found matching the pattern 'ralphy/agent-*'")
        return

    print(f"\nFound {len(agent_branches)} branches matching 'ralphy/agent-*':")
    print("-" * 60)

    # Get current branch
    try:
        current_branch_result = subprocess.run(['git', 'branch', '--show-current'],
                                            capture_output=True, text=True, check=True)
        current_branch = current_branch_result.stdout.strip()
    except subprocess.CalledProcessError:
        # Fallback to parsing git branch output
        try:
            all_branches_raw = subprocess.run(['git', 'branch'],
                                           capture_output=True, text=True, check=True)
            for line in all_branches_raw.stdout.split('\n'):
                if line.startswith('*'):
                    current_branch = line.lstrip('* ').strip()
                    break
            else:
                current_branch = None
        except subprocess.CalledProcessError:
            current_branch = None

    for i, branch in enumerate(agent_branches, 1):
        marker = " (current)" if branch == current_branch else ""
        print(f"{i:2d}. {branch}{marker}")


if __name__ == "__main__":
    main()