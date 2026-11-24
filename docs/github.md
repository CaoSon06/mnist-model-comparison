# Đưa project lên GitHub

Thư mục này là một project riêng; không chạy `git init` trong toàn bộ Downloads.
File notebook gốc ở thư mục Downloads không bị sửa.

## Tạo repository

Trên github.com, đăng nhập và tạo repository tên `mnist-model-comparison` hoặc
tên bạn muốn. Chọn public/private theo nhu cầu. Để repository trống: không
tạo README, gitignore hoặc license ở bước này vì project đã có file cục bộ.

Trong PowerShell:

```powershell
cd C:\Users\Hi\Downloads\mnist-model-comparison
git init -b main
git status
git add .
git commit -m "Initial MNIST model comparison project"
git remote add origin https://github.com/YOUR_USERNAME/mnist-model-comparison.git
git push -u origin main
```

Thay `YOUR_USERNAME` và tên repository bằng thông tin thật. Nếu Git cần danh
tính cho commit, đặt **trong repository này** bằng thông tin GitHub của bạn:

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL_OR_NOREPLY"
```

Đăng nhập bằng trình duyệt khi Git Credential Manager yêu cầu, hoặc dùng CLI:

```powershell
gh auth login
```

Nếu muốn GitHub CLI tạo repository thay cho giao diện web, chạy sau commit:

```powershell
gh repo create mnist-model-comparison --public --source . --remote origin --push
```

Chọn `--private` thay `--public` nếu cần. Đây là lệnh tùy chọn để bạn chủ động
thực thi; project được chuẩn bị cục bộ, chưa tạo remote hay push tự động.

## Sau khi push

- Kiểm tra README hiển thị đủ hình minh họa và bảng kết quả.
- Vào Actions xem workflow `CI` chạy test/build/notebook validation.
- Gợi ý description: `Seven MNIST classifiers with reproducible training,
  Optuna tuning, and model-specific Explainable AI.`
- Gợi ý topics: `mnist`, `machine-learning`, `deep-learning`, `tensorflow`,
  `scikit-learn`, `explainable-ai`, `optuna`, `vision-transformer`.
- Dataset và weights không được Git theo dõi. Nếu muốn chia sẻ mô hình đã
  train, có thể đính kèm vào GitHub Release cùng config và metrics.

Nếu repository remote đã có commit, clone repository đó trước rồi copy các
file project vào; hướng dẫn trên dành cho một repository mới và trống.
