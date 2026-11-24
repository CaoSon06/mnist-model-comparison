# MNIST project design

## Intent and scope
Turn the user-provided AI-Final.ipynb into a GitHub-ready research project.
Keep the original scientific scope: EDA, KNN, SVM, Random Forest, MLP, CNN,
custom mini ResNet-9, a small ViT, Optuna, evaluation and Explainable AI.
The source notebook stays untouched outside this repository.

## Approach
A notebook-only repository is easy to preserve but difficult to rerun in parts.
A web application adds a separate product the user did not request.
Use a Python package with a CLI, a clean historical notebook, and a short
package-based workflow notebook. This is an implementation choice within the
user's request for a complete project, not a separately approved new feature.

## Structure and behavior
src/mnist_project contains data, model factories, training/evaluation,
explanations, tuning and CLI modules. notebooks contains the cleaned original
and the workflow. configs contains JSON defaults; tests uses unittest and tiny
offline fixtures. reports/notebook contains attributed historical results and
selected existing notebook figures, never presented as freshly reproduced.

Use the official 10,000-image test partition unchanged. Split the original
60,000 training images into 54,000 train / 6,000 validation with stratification
and seed 42. Apply the same split to all seven model families. Fit scalers only
on training data. Optuna never evaluates the test set. Limit flags deliberately
produce smaller runs and must appear in run metadata. Fix Tiny-ViT positional
embedding tracking with a registered, serializable Keras layer.

Artifacts include models, JSON config/metrics/history, predictions and plots.
Data, trained weights, environments and run outputs are Git-ignored.
No invented authors, license grant, credentials, remote URL or fresh accuracy.
Use README Vietnamese, optional English overview, architecture/methodology docs,
contribution guidance, GitHub CI and exact push instructions.

## Validation
Exercise data leakage boundaries, normalization, artifact round trips for all
seven families, finite XAI maps, IG completeness on a linear fixture, tracked ViT
embeddings, CLI train/predict/evaluate, and a one-trial tuning run. Use offline
fixtures for CI; an MNIST run requires its dataset download. Package build and
notebook schema checks must pass. Report any unexecuted expensive training.
