# Merge All Agent Branches

- [x] Commit all uncommitted changes in worktrees located in .ralphy-worktrees/agent-* directories
- [x] Merge all branches matching pattern ralphy/agent-* into ralphy-base using git merge --no-edit
- [x] Delete all merged agent branches using git branch -D
- [ ] Remove worktree directories in .ralphy-worktrees/
- [ ] Run git worktree prune to clean up
