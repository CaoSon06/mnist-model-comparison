# Dataset

Không commit dữ liệu MNIST. `python -m mnist_project train ...` tải dataset
vào `data/raw/mnist.npz` (đã được `.gitignore` bỏ qua).

Dùng `--data đường/dẫn/mnist.npz` để chạy offline. NPZ cần bốn khóa:
`x_train`, `y_train`, `x_test`, `y_test`. Ảnh uint8 shape `(N,28,28)`, nhãn
integer 0–9. Đây cũng là định dạng file tải bởi Keras.

Nguồn: [Keras MNIST](https://keras.io/api/datasets/mnist/).
Điều kiện sử dụng dataset được quy định bởi nhà cung cấp dữ liệu.
