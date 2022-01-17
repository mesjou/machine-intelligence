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



