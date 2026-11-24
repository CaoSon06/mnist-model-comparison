# Project validation

Validation date: **2026-11-24**. Environment: Windows, Python 3.13.5, TensorFlow 2.21.0 on CPU. Direct dependency versions are recorded in `requirements-tested.txt`, which is not a full transitive lockfile.

## Automated tests

Command: `python -m unittest discover -s tests -v` — **all 11 tests passed**. Tests use small synthetic offline data; accuracy is not their objective. They cover:

- uint8/float normalization, shape, range, and invalid inputs.
- Disjoint, reproducible splits preserving the official test input.
- NPZ loading and stratified subsets retaining rare classes even near full size.
- Training, saving/loading, and evaluating KNN, SVM, RF, MLP, CNN, ResNet, and ViT; model-specific XAI figures; refusal to overwrite experiments.
- ViT positional embeddings updated by training and retained through save/load.
- IG matching a directly calculable linear solution; deletion tests using the requested target even when it differs from the predicted class.
- Finite saliency, IG, Grad-CAM, and attention rollout maps of expected shapes.
- Benchmark label prediction without an extra optional score computation.
- CLI train/evaluate/predict/EDA, one Optuna trial, and training from `best_params` with batch-size override.

TensorFlow/Keras may emit upstream warnings about deprecated APIs, NumPy array conversion, or retracing when several models are created in one test process. Native Windows TensorFlow uses CPU. These warnings did not fail the tests.

## Notebooks, package, and artifacts

`python tools/validate_notebooks.py` checked notebook schemas, Python syntax, and cleared outputs. The 145 source cells in the reference notebook retained their original order. The original file outside this repository was not modified; explanatory comments and messages in the copy were translated into English. The historical notebook was not rerun end-to-end because it contains expensive tuning loops and the documented methodological limitations.

All code cells in `01_workflow.ipynb` were executed in sequence in one Python process with real MNIST: 2,000 train / 6,000 validation / 500 test images, one CNN epoch, EDA, figures, reload, prediction, and Grad-CAM. This checked code execution, not the Jupyter UI or kernel through nbclient. Delivered notebook outputs remained clear.

The CLI also completed `train --model all --epochs 1 --train-limit 200 --test-limit 100` on real MNIST with 6,000 separate validation images and generated models, metrics, plots, and `comparison.csv`. This was a smoke run; 200 training images and one epoch do not represent full-training quality.

A wheel was built and installed with `pip --target` into a separate directory, and the CLI module was checked outside the source tree. Build command:

```bash
python -m build
```

Models and data used for verification are in Git-ignored directories and were not included in the source ZIP. Local Markdown links and Git ignore rules for data, weights, and build directories were checked.

## Review and limits

An independent review found three issues that were fixed with regression tests: rare-class loss during subsampling, the wrong target in deletion tests, and ML timing that included an unnecessary extra inference. The CLI batch-size override was also fixed after a reproducing test.

Full-data training for 15 epochs, repeated-seed variation, GPU/macOS checks, and GitHub Actions on a remote repository were not performed in this validation record. README accuracy values remain historical notebook results.