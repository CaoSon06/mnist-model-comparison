# Publish the project on GitHub

This is a standalone project directory. Do not run `git init` in the whole Downloads directory. The original notebook in Downloads is unchanged.

## Create a repository

Sign in to github.com and create an empty repository named `mnist-model-comparison` or a name of your choice. Choose public or private. Do not initialize it with a README, gitignore, or license because those files already exist locally.

In PowerShell:

```powershell
cd C:\Users\Hi\Downloads\mnist-model-comparison
git init -b main
git status
git add .
git commit -m "Initial MNIST model comparison project"
git remote add origin https://github.com/YOUR_USERNAME/mnist-model-comparison.git
git push -u origin main
```

Replace `YOUR_USERNAME` and the repository name. If Git needs a commit identity, set it in this repository:

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL_OR_NOREPLY"
```

Sign in through the browser if Git Credential Manager prompts you, or use:

```powershell
gh auth login
```

Alternatively, after committing, GitHub CLI can create and push the repository:

```powershell
gh repo create mnist-model-comparison --private --source . --remote origin --push
```

Use `--public` if you want a public repository. The project directory alone does not create or update a remote.

## After pushing

- Check that the README displays its figures and results table.
- Check Actions for the `CI` test, build, and notebook validation workflow.
- Suggested description: `Seven MNIST classifiers with reproducible training, Optuna tuning, and model-specific Explainable AI.`
- Suggested topics: `mnist`, `machine-learning`, `deep-learning`, `tensorflow`, `scikit-learn`, `explainable-ai`, `optuna`, `vision-transformer`.
- Data and model weights are excluded from Git. To share trained models, attach them to a GitHub Release with their config and metrics.

If the remote repository already has commits, clone it first and copy this project's files into that clone. The commands above assume a new, empty repository.