This workflow will merge the current branch into main. 
1. First see which branch you are currently on with `git branch --show-current`. Then replace `<current-branch-name>` below with that branch name.
1. Use `git checkout main` to switch to the main branch.
2. Use `git pull origin main` to ensure the main branch is up to date.
3. Use `git merge <current-branch-name>` to merge the current branch into main.
4. Use `git push origin main` to push the updated main branch to the remote repository.
5. Use `git checkout <current-branch-name>` to switch back to the original branch.
6. Depending on the branch, you may also want to merge main back to another branch, if <current-branch-name> is 'refactor-dev-1', then we also need to sync another branch 'dev2' with main, if <current-branch-name> is 'dev2', changes should be updated to 'refactor-dev-1' instead. Follow the appropriate steps below:
7. If <current-branch-name> is 'refactor-dev-1':
   1. Use `git checkout dev2` to switch to the dev2 branch.
   2. Use `git pull origin dev2` to ensure the dev2 branch is up to date.
   3. Use `git merge main` to merge the main branch into dev2.
   4. Use `git push origin dev2` to push the updated dev2 branch to the remote repository.
   5. Use `git checkout refactor-dev-1` to switch back to the original branch.
8. If <current-branch-name> is 'dev2':
   1. Use `git checkout refactor-dev-1` to switch to the refactor-dev-1 branch.
    2. Use `git pull origin refactor-dev-1` to ensure the refactor-dev-1 branch is up to date.
    3. Use `git merge main` to merge the main branch into refactor-dev-1.
    4. Use `git push origin refactor-dev-1` to push the updated refactor-dev-1 branch to the remote repository.
    5. Use `git checkout dev2` to switch back to the original branch.
9. Finally, verify that all branches are up to date by checking out each branch and pulling the latest changes from the remote repository.
