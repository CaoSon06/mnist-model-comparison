# MNIST Model Comparison & Explainable AI

A Python research project derived from `AI-Final.ipynb`: seven model families,
Optuna tuning, evaluation, inference and model-specific explanations.

The [Vietnamese README](README.md) contains the full setup, command reference,
repository structure and historical results.

## Quick start

Python 3.11–3.13. Create and activate a virtual environment, then:

```bash
python -m pip install -r requirements.txt
python -m mnist_project train --model cnn --epochs 1 --train-limit 2000 --test-limit 500 --output artifacts/cnn-quick
python -m mnist_project train --model all --output artifacts/full-run
python -m mnist_project --help
```

Windows: use `.venv\Scripts\python.exe` if environment activation is unavailable.
The first run downloads MNIST into ignored `data/raw/`. For offline use, pass
`--data path/to/mnist.npz`. Models, cached data and experiment outputs are not
included in Git. Choose a new output directory for each run.

## Models and explanations

| Model | CLI name | Default explanation |
|---|---|---|
| K-nearest neighbors | `knn` | Nearest training examples |
| StandardScaler + RBF SVM | `svm` | Decision-score occlusion |
| Random Forest | `random_forest` | Global pixel importance |
| MLP | `mlp` | Integrated Gradients |
| CNN | `cnn` | Grad-CAM |
| Custom mini ResNet-9 | `resnet9` | Grad-CAM |
| Small Vision Transformer | `tinyvit` | Attention rollout |

The custom ViT is a small educational architecture, not an implementation of
the external TinyViT paper/model family. The mini ResNet follows the supplied
notebook and is not a canonical torchvision ResNet.

```bash
python -m mnist_project tune --model cnn --trials 5 --epochs 10 --output artifacts/search
python -m mnist_project train --model cnn --params artifacts/search/best_params.json --output artifacts/tuned
python -m mnist_project predict --run artifacts/tuned --image digit.png
python -m mnist_project explain --run artifacts/tuned --image digit.png --method ig --output artifacts/ig
python -m unittest discover -s tests -v
```

The new pipeline uses a shared 54,000 / 6,000 / 10,000 stratified
train/validation/official-test protocol. Historical notebook metrics are
separately attributed and **are not reproduced results** of the new pipeline.
Original MLP and ViT methodology issues are documented in
[methodology](docs/methodology.md). Attention is not a causal explanation;
single-image deletion tests are illustrative only.

Open `notebooks/01_workflow.ipynb` for the runnable package workflow.
`notebooks/AI-Final.ipynb` is a clean historical reference retaining original
code and its methodological limitations.

See [GitHub instructions](docs/github.md), [validation](docs/validation.md)
and [contributing](CONTRIBUTING.md). No distribution license has been selected.
