# MNIST Project Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a complete, tested repository based on AI-Final.ipynb.

**Architecture:** Shared data split and model factories, reusable experiment
runner and CLI, preserved research notebook plus package tutorial.

**Tech Stack:** Python 3.11–3.13, TensorFlow/Keras, scikit-learn, NumPy,
Matplotlib, Optuna, unittest, nbformat, setuptools.

**Spec:** ../specs/2026-11-24-project-design.md

## Global Constraints
- Preserve original file and clearly attribute historical metrics.
- Same stratified seed-42 partition for all models; no test data in tuning.
- No external publishing, credentials or invented project ownership.
- New project has its own directory; Downloads is not a Git repository.

## Review Focus
- Small subsets must retain sufficient class coverage for stratification.
- ViT saved weights include positional embeddings.
- SVM decision margins are not labeled probabilities.
- Repeated runs never silently overwrite existing experiment directories.
- Notebook results and new pipeline metrics have distinct provenance.

### Task 1: Data and model contracts
- [x] Write offline data/model tests and observe missing-package failure.
- [x] Implement normalize_images, split_data, load_data, build_model and load_model.
- [x] Verify disjoint splits, float32 range, shape, all model round trips.

### Task 2: Experiment workflows
- [x] Write behavioral tests for artifacts, XAI and CLI.
- [x] Implement training, evaluation, EDA, explanations, tuning, CLI and defaults.
- [x] Run suite and CLI integration on tiny offline data.

### Task 3: Repository delivery
- [x] Create clean original notebook and template-based workflow notebook.
- [x] Extract historical results/figures, write READMEs and methodology.
- [x] Add packaging, gitignore, CI, contributing and push instructions.
- [x] Validate notebooks, build wheel, review entire project and create ZIP.

## Execution record
Implement inline. User authorized local project creation; no extra permission
rounds for reversible implementation. No existing repository requires a worktree.

11 offline tests passed, including all model families and per-model default XAI.
Real-MNIST smoke runs completed for all seven models (200 train/100 test), and
all workflow code cells executed in order (CNN 2000 train/500 test).
Independent review identified sampling, target selection and duplicate inference
issues. All corrected with regression tests observed failing before fixes.
Ruling: original notebook remains a historical reference, not silently patched;
methodology fixes live in the package and are documented explicitly.
