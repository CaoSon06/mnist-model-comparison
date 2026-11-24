# Dataset

Do not commit MNIST data. `python -m mnist_project train ...` downloads the dataset to `data/raw/mnist.npz`, which is excluded by `.gitignore`.

Use `--data path/to/mnist.npz` to run offline. The NPZ file must contain `x_train`, `y_train`, `x_test`, and `y_test`. Images are uint8 arrays of shape `(N,28,28)`, and labels are integers from 0 to 9. This is also the format downloaded by Keras.

Source: [Keras MNIST](https://keras.io/api/datasets/mnist/). Dataset usage terms are set by the data provider.