---
description: "This command merges the current branch into the main branch."
---

This workflow will merge the current branch into main. 
1. First see which branch you are currently on with `git branch --show-current`. Then replace `<current-branch-name>` below with that branch name.
1. Use `git checkout main` to switch to the main branch.
2. Use `git pull origin main` to ensure the main branch is up to date.
3. Use `git merge <current-branch-name>` to merge the current branch into main.
4. Use `git push origin main` to push the updated main branch to the remote repository.
5. Use `git checkout <current-branch-name>` to switch back to the original branch.