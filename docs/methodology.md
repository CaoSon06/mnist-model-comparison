# Methodology and differences from the original notebook

## Data

MNIST is read from a Keras NPZ with `x_train`, `y_train`, `x_test`, and `y_test`. Integer images are converted to float32 and divided by 255 exactly once; float images must already lie in `[0,1]`. Training uses no augmentation, matching the notebook's intent.

The official 10,000-image test partition stays unchanged. A stratified `train_test_split` with `random_state=42` takes 10% of the original 60,000 training images for validation. All seven models use 54,000 train, 6,000 validation, and 10,000 test images. ML models train on the same training partition as deep models; validation is reserved for a common protocol. SVM's StandardScaler is fit on training data only.

`--train-limit` takes a stratified subset **after** the validation split; `--test-limit` takes a subset of the official test data. Validation does not shrink when training is limited. Limits are for checks or separate research; do not compare a limited run with a full-data benchmark. Each run records its indices and sample counts.

## Models

- KNN: k=3, Euclidean distance, flattened normalized data.
- SVM: StandardScaler + RBF SVC, C=0.5, gamma=scale, with probability estimates disabled.
- Random Forest: 100 trees, max_depth=10, min_samples_leaf=5.
- MLP: the tuned MLP architecture from the notebook, rather than its initial single-layer MLP.
- CNN: two convolution blocks, 32/64 default filters, BN/ReLU/pooling, and Dense(128).
- Custom mini ResNet-9: initial convolution, two residual blocks, pooling, and dense head. It follows the notebook and is not a canonical external ResNet.
- Tiny-ViT: 4×4 patch convolution, 49 tokens, width 64, four attention heads, and two Transformer blocks. The name is educational; this is not the external TinyViT model family.

ResNet and ViT learning rate and dropout defaults retain the notebook's hard-coded values. CNN and MLP default to learning_rate=0.001, num_filters=32, and dropout=0.3. These are starting values, not independently revalidated historical best parameters.

## Tuning and evaluation

Seeded Optuna TPE maximizes validation accuracy in each trial. Tuning and training use the same model factory. After tuning, training starts anew with `best_params` on the same training partition; the official test set is used only afterward. Deep models use the requested epoch count and ReduceLROnPlateau, with no early stopping. Tuning has no scheduler, matching the initial search process. A seed aids reproducibility but does not guarantee bitwise equality across hardware or TensorFlow versions.

Accuracy and macro/weighted F1 are reported separately. Macro precision, recall, and F1 cover classes present in ground truth or predictions; classification reports and confusion matrices list all ten digits. Inference time includes input preparation and predictions or scores but excludes loading a model from disk; there is no warm-up. A benchmark makes one inference pass for labels; SVM does not call `decision_function` again inside the timed interval. Milliseconds per image depend on hardware, batch size, and runtime conditions.

## Issues identified in the original notebook

1. The original MLP combines 60,000 training and 10,000 official test images, then resplits into 56,000 train, 7,000 validation, and 7,000 test. Its evaluation set therefore differs from those of other models. The new pipeline keeps the official test set separate.
2. The final Tiny-ViT fits on `X_train_cnn` (60,000 images) while validating on `X_val_final`, which comes from those same images. The new pipeline uses a separate training partition.
3. Original positional embeddings are created from `tf.range` outside the graph and added to patch tensors. The new registered `PositionEmbedding` uses `add_weight`, so training updates and `.keras` serialization retain them. See the [Keras serialization guide](https://keras.io/guides/serialization_and_saving/).
4. Some original ResNet/ViT training hyperparameters are hard-coded instead of read directly from the completed study. The new CLI accepts `best_params.json`.
5. The original MLP report uses `len(X_test_flat)` = 10,000 to describe latency, although its actual test set has 7,000 images. Only accuracy and F1 are extracted here; that latency is excluded.

The historical notebook retains its code for traceability; fixes live in the package. Historical scores do not establish the new pipeline's accuracy.

## Explanations

Saliency, Integrated Gradients (IG), and Grad-CAM use logits before softmax. IG uses a black baseline and trapezoidal integration and preserves signed attributions in an NPY file. Overlays show magnitude only. Grad-CAM uses the final CNN activation map or final ResNet residual block. RF feature importance is global; SVM occlusion measures the decision-score change after setting a patch to zero.

Attention rollout averages heads, adds an identity matrix, normalizes each row, multiplies the two blocks, and averages queries because the model has no CLS token. It shows token interactions, not a causal or class-specific explanation. `--target` does not change attention rollout.

The IG deletion test compares removing pixels by attribution magnitude with five random deletion runs on a single selected image. It cannot support a conclusion about the full dataset. MNIST has a simple background and centered digits; results may not generalize to real photographs.