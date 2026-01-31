#!/bin/bash

# Script to commit all uncommitted changes in worktrees located in .ralphy-worktrees/agent-* directories

set -e  # Exit on any error

echo "Checking for agent worktrees..."

# Get the current directory to ensure we're in the right place
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

echo "Repository root: $REPO_ROOT"

# Find all agent worktrees in .ralphy-worktrees directory
AGENT_WORKTREES=$(find "$REPO_ROOT/.ralphy-worktrees" -mindepth 1 -maxdepth 1 -type d -name "agent-*" 2>/dev/null || true)

if [ -z "$AGENT_WORKTREES" ]; then
    echo "No agent worktrees found in $REPO_ROOT/.ralphy-worktrees"
    
    # Check if the .ralphy-worktrees directory exists
    if [ ! -d "$REPO_ROOT/.ralphy-worktrees" ]; then
        echo "Creating directory $REPO_ROOT/.ralphy-worktrees"
        mkdir -p "$REPO_ROOT/.ralphy-worktrees"
    fi
    
    echo "Looking for agent worktrees using git worktree list..."
    
    # Get all worktrees using git command and filter for agent-* patterns
    while IFS= read -r worktree_line; do
        if [[ $worktree_line =~ ^[^[:space:]]+[[:space:]]+[a-f0-9]+[[:space:]]+\[.*\]$ ]]; then
            worktree_path=$(echo "$worktree_line" | awk '{print $1}')
            worktree_branch=$(echo "$worktree_line" | sed 's/.*\[//' | sed 's/\].*//')
            
            # Check if this is an agent worktree
            if [[ "$worktree_path" == *"/.ralphy-worktrees/agent-"* ]]; then
                echo "Found agent worktree: $worktree_path (branch: $worktree_branch)"
                
                if [ -d "$worktree_path" ] && [ -d "$worktree_path/.git" ]; then
                    echo "Processing worktree: $worktree_path"
                    
                    # Check if there are uncommitted changes
                    cd "$worktree_path"
                    if ! git diff-index --quiet HEAD --; then
                        echo "Found uncommitted changes in $worktree_path"
                        
                        # Show the status
                        git status --short
                        
                        # Add all changes
                        git add .
                        
                        # Commit with a generic message
                        git commit -m "Auto-commit: Save uncommitted changes in agent worktree"
                        
                        echo "Committed changes in $worktree_path"
                    else
                        echo "No uncommitted changes in $worktree_path"
                    fi
                else
                    echo "Worktree directory does not exist or is not a git repo: $worktree_path"
                fi
            fi
        fi
    done < <(cd "$REPO_ROOT" && git worktree list | tail -n +2)
else
    # Process each agent worktree directory
    for worktree_dir in $AGENT_WORKTREES; do
        if [ -d "$worktree_dir" ] && [ -d "$worktree_dir/.git" ]; then
            echo "Processing worktree: $worktree_dir"
            
            cd "$worktree_dir"
            
            # Check if there are uncommitted changes
            if ! git diff-index --quiet HEAD --; then
                echo "Found uncommitted changes in $worktree_dir"
                
                # Show the status
                git status --short
                
                # Add all changes
                git add .
                
                # Commit with a generic message
                git commit -m "Auto-commit: Save uncommitted changes in agent worktree"
                
                echo "Committed changes in $worktree_dir"
            else
                echo "No uncommitted changes in $worktree_dir"
            fi
        else
            echo "Skipping $worktree_dir - not a valid git worktree"
        fi
    done
fi

echo "Finished processing agent worktrees."