# Git Conflict Resolution & Master Merge

- [ ] Iterate through all branches: `ralphy/agent-*`
- [ ] For each branch:
    - [ ] Attempt a merge into `ralphy-base`.
    - [ ] IF conflicts occur:
        - [ ] Analyze conflicting files.
        - [ ] Favor changes that implement new features over deletions.
        - [ ] Resolve code-level conflicts by keeping both logic paths if they don't overlap.
        - [ ] Commit the resolved merge.
    - [ ] Delete the branch after successful merge and commit.
- [ ] Final task: Run a build check to ensure the engine still compiles.