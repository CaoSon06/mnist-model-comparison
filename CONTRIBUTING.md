# Contributing

Create a virtual environment, install `requirements.txt`, and run:

```bash
python -m unittest discover -s tests -v
python tools/validate_notebooks.py
python -m build
```

Keep tests offline and use small synthetic fixtures. Changes to data splitting
must preserve the official test partition and train/validation disjointness.
Do not select hyperparameters based on test performance.

Use the shared model factory for both tuning and training. Any custom Keras
layer must survive a save/load round trip with matching predictions. Explain
which score an attribution method differentiates or perturbs.

Keep notebook outputs cleared before committing. Do not commit downloaded
data, weights, generated experiment folders, local environments or credentials.
Historical figures in reports/notebook are deliberately versioned artifacts
from the supplied source notebook; mark newly generated benchmarks separately.

Describe the change, the expected behavior and the validation performed in
pull requests. Note the exact data subset, seed and environment when sharing
new benchmark results.
