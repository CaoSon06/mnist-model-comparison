# Kiểm tra project

Ngày kiểm tra: **24/11/2026**. Môi trường: Windows, Python 3.13.5, TensorFlow
2.21.0 trên CPU. Các phiên bản trực tiếp được ghi tại `requirements-tested.txt`;
đó không phải lockfile đầy đủ của mọi dependency bắc cầu.

## Kiểm thử tự động

Lệnh: `python -m unittest discover -s tests -v` — **11 bài kiểm thử đều qua**.
Test dùng dữ liệu nhân tạo nhỏ, ngoại tuyến, không dùng accuracy làm mục tiêu:

- Normalize uint8/float, shape, range, input không hợp lệ.
- Split disjoint và tái lập, giữ nguyên official-test input.
- NPZ loading; tập con vẫn có lớp hiếm và nhận kích thước gần toàn bộ tập.
- Train, lưu/nạp và đánh giá đủ KNN/SVM/RF/MLP/CNN/ResNet/ViT; xuất hình XAI
  theo phương pháp mặc định của từng loại và từ chối ghi đè experiment.
- ViT positional embedding được cập nhật khi train và giữ nguyên sau save/load.
- IG khớp lời giải tuyến tính có thể tính trực tiếp; deletion test dùng đúng
  target kể cả khi target khác lớp dự đoán.
- Saliency, IG, Grad-CAM và attention rollout có shape đúng, giá trị hữu hạn.
- Benchmark có nhánh lấy nhãn không chạy thêm phần xuất scores tùy chọn.
- CLI train/evaluate/predict/EDA, một trial Optuna và train từ best_params
  với batch size được override.

TensorFlow/Keras có cảnh báo upstream về deprecated API, NumPy array conversion
và retracing khi tạo nhiều model trong một test process. Windows CPU không có
GPU TensorFlow. Những cảnh báo này không làm test thất bại.

## Notebook, package và artifact

`python tools/validate_notebooks.py` xác nhận schema, cú pháp Python và output
đã xóa của cả hai notebook. 145 cell nguồn trong notebook tham chiếu giữ nguyên
nội dung/thứ tự; file gốc không bị sửa. Bản tham chiếu lịch sử không được chạy
lại toàn bộ vì chứa nhiều vòng tuning và các hạn chế phương pháp đã ghi rõ.

Các code cell của `01_workflow.ipynb` đã chạy lần lượt từ đầu đến cuối trong
một Python process với MNIST thật: 2.000 train / 6.000 validation / 500 test,
CNN một epoch, EDA, xuất hình, nạp lại, predict và Grad-CAM. Đây là thực thi
tuần tự code cell; chưa chạy kiểm tra giao diện Jupyter hoặc kernel bằng nbclient.
Notebook được giao vẫn sạch output.

CLI `train --model all --epochs 1 --train-limit 200 --test-limit 100` cũng
đã chạy xong cả bảy mô hình trên MNIST thật với 6.000 validation ảnh giữ riêng,
xuất đủ model/metrics/plots và comparison.csv. Đây là smoke run; 200 training
ảnh và một epoch không đại diện cho chất lượng mô hình sau huấn luyện đầy đủ.

Wheel đã build, cài vào thư mục kiểm tra riêng bằng `pip --target` và kiểm tra
module CLI từ ngoài source tree. Lệnh build:

```bash
python -m pip --disable-pip-version-check --no-cache-dir wheel . --no-deps --no-build-isolation --wheel-dir artifacts/package-check
```

Các model và dữ liệu dùng khi xác minh nằm trong thư mục được Git bỏ qua;
không được đóng kèm ZIP mã nguồn. Kiểm tra link Markdown cục bộ và Git ignore
cho dataset, weights, build directory đã được thực hiện.

## Rà soát và giới hạn

Một lượt review độc lập phát hiện ba lỗi đã sửa và có kiểm thử hồi quy: tập con
làm mất lớp hiếm, deletion test dùng sai target, và thời gian ML tính cả lượt
inference bổ sung không cần thiết. CLI batch-size override cũng đã được sửa
sau khi test tái hiện lỗi.

Chưa huấn luyện đủ 15 epoch trên full-data, chưa lặp nhiều seed để báo cáo
độ biến thiên, chưa kiểm tra trên GPU/macOS, và GitHub Actions chưa chạy trên
remote. Các accuracy lịch sử trong README vẫn chỉ là số từ notebook nguồn.
