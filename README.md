# MNIST Model Comparison & Explainable AI

So sánh **7 mô hình nhận dạng chữ số viết tay** trên MNIST, từ machine learning
truyền thống đến mạng tích chập và Vision Transformer. Project được tổ chức lại
từ notebook `AI-Final.ipynb`, bổ sung mã nguồn tái sử dụng, CLI, kiểm thử và hướng
dẫn chạy trên máy cá nhân.

[English overview](README.en.md) · [Phương pháp](docs/methodology.md) ·
[Đưa lên GitHub](docs/github.md) · [Kết quả notebook gốc](reports/notebook/README.md)

![Ví dụ chữ số trong notebook gốc](reports/notebook/figures/mnist_samples.png)

## Nội dung

- EDA: phân bố lớp, ảnh mẫu, ảnh trung bình, phân bố pixel và độ thưa.
- KNN, SVM với StandardScaler, Random Forest.
- MLP, CNN, custom mini ResNet-9 và Tiny-ViT với 49 patch, 2 Transformer block.
- Optuna tối ưu learning rate, dropout, batch size và số filter/hidden unit.
- Accuracy, macro/weighted F1, classification report, confusion matrix, thời gian chạy.
- XAI: nearest neighbors, RF feature importance, SVM occlusion, saliency,
  Integrated Gradients, Grad-CAM, attention rollout và deletion test.
- Lưu/nạp mô hình, dự đoán ảnh riêng và xuất hình giải thích.

## Cài đặt

Python **3.11–3.13**. Môi trường kiểm tra cục bộ: Python 3.13.5, TensorFlow 2.21.0,
NumPy 2.4.1 và scikit-learn 1.8.0. CI cấu hình Python 3.11 và 3.13 trên Ubuntu;
trạng thái CI sẽ có sau khi repository được push.

Windows PowerShell, mở terminal trong thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sau đó dùng Python của môi trường này hoặc kích hoạt môi trường:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn `Activate.ps1`, có thể chạy trực tiếp
`.\.venv\Scripts\python.exe` thay cho `python` trong các lệnh bên dưới.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Chỉ cần CLI cơ bản: `python -m pip install -e .`. Thêm `.[research]` để tune,
`.[notebook]` để mở notebook. TensorFlow mới chạy CPU trên Windows native;
GPU NVIDIA dùng Linux/WSL2 theo [hướng dẫn TensorFlow](https://www.tensorflow.org/install/pip).

## Chạy nhanh

Lần đầu cần Internet để tải MNIST (~11 MB); dữ liệu được lưu tại `data/raw/`.
Lệnh chạy nhỏ dưới đây phục vụ kiểm tra luồng, không tạo benchmark đầy đủ:

```bash
python -m mnist_project train --model cnn --epochs 1 --train-limit 2000 --test-limit 500 --output artifacts/cnn-quick
```

Huấn luyện toàn bộ 7 mô hình, mặc định 15 epoch cho deep learning:

```bash
python -m mnist_project train --model all --config configs/default.json --output artifacts/full-run
```

SVM/KNN trên đầy đủ dữ liệu có thể tốn nhiều thời gian trên CPU. Mỗi model lưu
vào một thư mục con. Các lệnh từ chối ghi đè thư mục đã có nội dung; chọn tên
output mới cho mỗi thí nghiệm. Bỏ `--output` khi train để tự tạo tên theo thời gian.

```bash
python -m mnist_project eda --output artifacts/eda-run
python -m mnist_project evaluate --run artifacts/cnn-quick --output artifacts/cnn-evaluation.json
python -m mnist_project --help
```

Chạy offline với file MNIST đã tải: thêm `--data data/raw/mnist.npz` vào lệnh
`train`, `tune` hoặc `eda`. Khi chuyển experiment sang máy khác, `evaluate`
cũng nhận `--data` để cập nhật đường dẫn dataset.

## Tối ưu bằng Optuna

```bash
python -m mnist_project tune --model cnn --trials 5 --epochs 10 --output artifacts/cnn-search
python -m mnist_project train --model cnn --params artifacts/cnn-search/best_params.json --epochs 15 --output artifacts/cnn-tuned
```

`--model` của `tune`: `mlp`, `cnn`, `resnet9`, `tinyvit`. Tuning chỉ dùng train và
validation. Khi train từ kết quả tuning, dùng cùng `--seed`, `--validation-size`
và `--train-limit` nếu trước đó đã thay mặc định. CLI không tự sao chép các lựa
chọn chia dữ liệu từ file tuning.

## Dự đoán và giải thích

Ảnh đầu vào nên là chữ số sáng, được căn giữa trên nền đen, kích thước 28×28.
CLI đổi grayscale, resize giữ tỉ lệ và padding. Ảnh nền trắng dùng `--invert`.
Ảnh chụp thực tế có nền phức tạp cần tiền xử lý thêm.

```bash
python -m mnist_project predict --run artifacts/cnn-quick --image path/to/digit.png
python -m mnist_project explain --run artifacts/cnn-quick --image path/to/digit.png --method gradcam --output artifacts/cnn-gradcam
python -m mnist_project explain --run artifacts/cnn-quick --image path/to/digit.png --method ig --target 7 --steps 32 --output artifacts/cnn-ig
```

`--images-npy` nhận batch ảnh NumPy uint8 `[0,255]` hoặc float `[0,1]` thay cho
`--image`. `explain` dùng ảnh đầu tiên của batch. `--method auto` chọn phương
pháp phù hợp:

| Model | Giá trị `--model` | XAI mặc định |
|---|---|---|
| KNN | `knn` | 3 láng giềng gần nhất |
| SVM | `svm` | Occlusion theo decision score |
| Random Forest | `random_forest` | Global pixel importance |
| MLP | `mlp` | Integrated Gradients |
| CNN | `cnn` | Grad-CAM |
| Custom mini ResNet-9 | `resnet9` | Grad-CAM |
| Tiny-ViT | `tinyvit` | Attention rollout |

Deep models còn hỗ trợ `ig`, `saliency`; `attention` dành cho Tiny-ViT,
`gradcam` dành cho CNN/ResNet. IG lưu cả deletion test minh họa trên một ảnh.
Attention không phải giải thích nhân quả; RF importance là giải thích toàn cục;
SVM decision score không phải xác suất.

![Grad-CAM từ notebook gốc](reports/notebook/figures/gradcam.png)

## Cấu trúc

```text
mnist-model-comparison/
├── .github/workflows/ci.yml
├── configs/default.json
├── data/README.md
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── github.md
│   └── validation.md
├── notebooks/
│   ├── AI-Final.ipynb        # Bản lưu nghiên cứu gốc, đã xóa output
│   └── 01_workflow.ipynb     # Hướng dẫn chạy package
├── reports/notebook/        # Kết quả và hình trích từ output gốc
├── src/mnist_project/
│   ├── data.py
│   ├── models.py
│   ├── vit.py
│   ├── experiments.py
│   ├── tuning.py
│   ├── explain.py
│   ├── eda.py
│   └── cli.py
├── tests/test_project.py
├── tools/prepare_notebooks.py
├── artifacts/               # Dataset/model/output lớn không commit
├── pyproject.toml
├── requirements.txt
├── requirements-tested.txt
├── CONTRIBUTING.md
├── README.en.md
└── README.md
```

Mỗi experiment lưu `model.keras` hoặc `model.joblib`, `config.json`,
`split_indices.npz`, `metrics.json`, `history.json`, `predictions.npy`,
`confusion_matrix.png` và `comparison.csv`. Deep models có thêm
`learning_curves.png`. Config ghi seed, số mẫu, đường dẫn dữ liệu và phiên bản
thư viện. Chỉ nạp artifact do bạn tạo hoặc từ nguồn bạn tin cậy.

## Kết quả tham khảo

Các số dưới đây **được trích từ output đã lưu trong notebook gốc**, không phải
kết quả chạy lại sau khi tách project:

| Mô hình | Accuracy gốc | Ghi chú |
|---|---:|---|
| KNN | 97.05% | Official test, 10.000 ảnh |
| SVM | 95.96% | Official test, 10.000 ảnh |
| Random Forest | 94.77% | Official test, 10.000 ảnh |
| Tuned MLP | 97.57% | Test 7.000 ảnh sau khi gộp và chia lại dữ liệu |
| CNN | 99.43% | Official test, 10.000 ảnh |
| Custom ResNet-9 | 99.48% | Official test, 10.000 ảnh |
| Tiny-ViT | 97.83% | Validation bị trùng với dữ liệu huấn luyện gốc |

Pipeline mới dùng **54.000 train / 6.000 validation / 10.000 test** thống nhất,
khắc phục việc trùng validation của Tiny-ViT và lưu positional embedding đúng
trong model. Vì thay đổi giao thức và một số mặc định, không kỳ vọng khớp hoàn
toàn các số cũ. Chi tiết tại [methodology](docs/methodology.md).

## Notebook và kiểm thử

```bash
python -m jupyter lab
python -m unittest discover -s tests -v
python tools/validate_notebooks.py
python -m build
```

Mở `notebooks/01_workflow.ipynb` cho workflow mới. `AI-Final.ipynb` giữ nguyên
logic nghiên cứu lịch sử, đã xóa output và execution count; đọc ghi chú đầu
notebook trước khi chạy vì nó vẫn có các hạn chế chia dữ liệu của bản gốc.

Test dùng fixture nhỏ ngoại tuyến, thực sự train một epoch, lưu/nạp đủ 7 model,
kiểm tra chia dữ liệu, CLI và XAI. Đây là kiểm tra tính vận hành, không chứng
minh chất lượng dự đoán trên MNIST. Xem [báo cáo kiểm tra](docs/validation.md).

## GitHub và giấy phép

Làm theo [docs/github.md](docs/github.md) để tạo repository và push. Project
chưa chọn giấy phép phân phối; chủ sở hữu có thể bổ sung `LICENSE` phù hợp.
Không giả định giấy phép của dataset hoặc notebook thuộc bên thứ ba.
