# Historical notebook results

The numbers and figures here come from outputs **already present** in `AI-Final.ipynb`, not from rerunning the new package. Cell indices are zero-based and refer to the source file before an introduction was added to the cleaned notebook.

`provenance.json` contains the source notebook's SHA-256 and the cells that produced each figure or metric. `historical_metrics.csv` separates macro and weighted F1; a blank cell means the value was not extracted from that output. Accuracy is recorded on a 0–1 scale.

| Figure | Source |
|---|---|
| figures/mnist_samples.png | Cell 10: 25 training images |
| figures/confusion_matrices.png | Cell 87: confusion and error analysis |
| figures/integrated_gradients.png | Cell 124: IG comparison |
| figures/gradcam.png | Cell 128: CNN/ResNet Grad-CAM |
| figures/attention_rollout.png | Cell 134: Tiny-ViT attention |

The original notebook uses a different MLP data split and overlapping train/validation data for Tiny-ViT. These figures are therefore not a unified benchmark. See [methodology](../../docs/methodology.md). Saved outputs do not prove that the full notebook ran continuously in a clean kernel, and source metadata does not replace reproducible experiments.

To extract again from the original:

```bash
python tools/prepare_notebooks.py path/to/AI-Final.ipynb
```

The script is tailored to this specific source notebook. Its cell indices are not a general converter for other notebooks. It rewrites the clean reference notebook and historical artifacts in this repository. The workflow notebook is maintained separately.