# Kết quả lịch sử từ notebook

Các số và hình trong thư mục này được lấy từ output **đã có sẵn** trong
`AI-Final.ipynb`, không phải chạy lại package mới. Cell index dùng số bắt đầu
từ 0 trong file nguồn trước khi thêm lời dẫn vào bản notebook sạch.

`provenance.json` chứa SHA-256 notebook nguồn và cell xuất hình/metric.
`historical_metrics.csv` tách macro F1 và weighted F1; ô trống nghĩa là không
trích giá trị đó từ output tương ứng. Accuracy lưu ở thang 0–1.

| Hình | Nguồn |
|---|---|
| figures/mnist_samples.png | Cell 10: 25 ảnh training |
| figures/confusion_matrices.png | Cell 87: phân tích confusion/error |
| figures/integrated_gradients.png | Cell 124: IG comparison |
| figures/gradcam.png | Cell 128: CNN/ResNet Grad-CAM |
| figures/attention_rollout.png | Cell 134: Tiny-ViT attention |

Notebook gốc có cách chia dữ liệu MLP riêng và Tiny-ViT validation overlap;
không xem bảng này là benchmark thống nhất. Xem [methodology](../../docs/methodology.md).
Source outputs không chứng minh toàn bộ notebook đã được chạy liên tục trong
một kernel sạch. Metadata nguồn không thay thế việc tái lập thực nghiệm.

Tái trích từ bản gốc:

```bash
python tools/prepare_notebooks.py path/to/AI-Final.ipynb
```

Script dành cho đúng notebook nguồn này; các cell index không phải bộ chuyển
đổi tổng quát cho notebook bất kỳ. Nó ghi lại notebook sạch và historical
artifacts trong repository hiện tại. Workflow notebook được duy trì riêng.
