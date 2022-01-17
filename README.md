# machine-intelligence

How to use a github repo


### 1. Clone the repository to your local machine.

You can use your terminal to navigate to the desired folder, then you can clone the repository.
This will copy the main branch of the repository to your computer.
In this repository you can later on create your own branches, start editing the files and send your changes back to Github.
But lets first clone and the rest will follow:

```
# navigate to the desired destination
cd User/Documents/University

# clone the repo
git clone https://github.com/mesjou/machine-intelligence.git

# check if everything works (this should show branch main and everything should be up to date)
git status
```

### 2. Checkout a branch to work on the code and do some changes

Now you are on the top branch.
By default this branch is called `main` or previously `master` (the master - slave wording is not popular anymore!).
It is good practice to never make changes on the `main` branch because this is the "ground truth" of the project.
If you work with other collaborateurs that all make changes to the main branch it is not clear what happens if more than one person send their changes to the main branch in github?

```
git checkout -b sheet-8-1
```

Now you have created your own branch with the name `sheet-8-1` (do not add spaces!).
This branch only exists at your local computer and nobody nows about it for now.

### 3. Make your changes and commit them to your branch

Now you can edit the files.
If you are ready with your changes or you want to "save" your changes (e.g. when you finished one step), you can commit your changes.
First add the files you changed and want to commit.
Then commit them and write a commit message.

```
# this will add all files to the commit
git add .

# this will add only one file
git add example_file.py

git commit -m"I changed something in the first excercise"
```

### 4. Push your changes to the remote repository aka Github

You can change and commit as much as you want.
If you are satisfied with your changes you can now push your changes in your local branch to the remote repository.
Remember: as for now the branch is only on your computer and nobody knows about it.


```
# try to push your changes to remote
git push

# it failed because the brnach in remote was not yet created, so run
git push --set-upstream origin sheet-8-1
```

This will create the branch sheet-8-1 in github, set your the branch to track your local branch sheet-8-1.
In the next step it pushes your changes to the remote branch (the one in the origin, the github repo).

### 5. Create Pull request (PR)
Now we have your changes in github but our main branch is still unaffected.
In order to get your changes in the main branch, we create a pull request in github.
There we can decide if we want to accept your changes and incorporate them in main or if we want you to modify it.
Also if your changes are conflicting with other peoples' changes we can esolve this problems now.
After we closed the PR we can delete the branch if we do not need it anymore.

### 6. Update your local repository
The main branch is now different than the version of the main branch you have stored on your personal computer.
In order to get the latest version of the main branch from github, first always switch to your local main branch.
Then pull the new version from github

```
# switch from branch sheet-8-1 to main
git checkout main

# get the latest version and ensure that you did no changes to your local main branch
# this will update all the remote branches and the local branch you are in (e.g. the main branch).
git pull --ff-only
```
If you want to modify something you can now start over with a creating a new branch, e.g. 2. 
