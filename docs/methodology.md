# Phương pháp và khác biệt với notebook gốc

## Dữ liệu

MNIST được đọc từ file Keras NPZ với bốn khóa `x_train`, `y_train`, `x_test`,
`y_test`. Ảnh integer được đổi float32 và chia 255 đúng một lần; ảnh float phải
nằm trong [0,1]. Không augmentation trong luồng train, phù hợp chủ ý notebook.

Giữ nguyên 10.000 ảnh official test. `train_test_split` lấy 10% của 60.000 ảnh
training làm validation, `stratify=y_train`, `random_state=42`. Mọi mô hình dùng
cùng 54.000/6.000/10.000 mẫu. ML dùng train partition như DL; validation được
giữ lại để bảo đảm cùng giao thức. StandardScaler trong SVM fit trên train.

`--train-limit` chọn tập con có phân tầng **sau** bước chia validation;
`--test-limit` lấy tập con official test. Validation không tự giảm khi giới hạn
train. Chỉ dùng giới hạn để kiểm tra hoặc nghiên cứu riêng, không so sánh một
run giới hạn với benchmark full-data. Các index và số mẫu được lưu với run.

## Mô hình

- KNN: k=3, Euclidean distance, dữ liệu flatten đã normalize.
- SVM: StandardScaler + RBF SVC, C=0.5, gamma=scale; không bật probability.
- Random Forest: 100 cây, max_depth=10, min_samples_leaf=5.
- MLP: Dense(num_filters) + BN + ReLU, Dense(128) + BN + ReLU + dropout.
  Đây là kiến trúc tuned MLP trong notebook, không phải pure MLP một lớp ban đầu.
- CNN: hai conv block, 32/64 filter mặc định, BN/ReLU/pooling, Dense(128).
- Custom mini ResNet-9: initial conv, hai residual block, pooling và dense head;
  giữ cấu trúc của notebook, không tuyên bố đây là một ResNet chuẩn khác.
- Tiny-ViT: patch convolution 4×4, 49 token, chiều 64, 4 attention head,
  key_dim=64, hai block, feed-forward 128/64, global average pooling.
  Đây là tên gọi giáo dục trong notebook, không phải model TinyViT của một
  thư viện hoặc bài báo bên ngoài.

Learning rate/default dropout ResNet và ViT giữ các số hard-code từ notebook.
CNN và MLP mặc định learning_rate=0.001, num_filters=32, dropout=0.3;
chúng là điểm khởi đầu, không phải best_params cũ đã được xác minh lại.

## Tối ưu và đánh giá

Optuna TPE có seed cố định, chọn max validation accuracy trong mỗi trial.
Factory dùng chung cho tuning và train để kiến trúc nhất quán. Sau tuning,
train lại từ đầu với best_params trên cùng train partition; official test chỉ
dùng sau huấn luyện. Giữ nguyên số epoch yêu cầu, dùng ReduceLROnPlateau cho
train deep models, không early stopping. Tuning không có scheduler, phù hợp
luồng tìm kiếm ban đầu. Seed hỗ trợ tái lập nhưng không bảo đảm giống bit giữa
thiết bị hoặc phiên bản TensorFlow.

Accuracy, macro/weighted F1 được xuất riêng. Macro precision/recall/F1 tính
trên các lớp có trong ground truth hoặc dự đoán; classification report và
confusion matrix luôn liệt kê đủ 10 chữ số. Thời gian inference gồm chuẩn bị
input và lấy predictions/scores, không gồm nạp model từ đĩa; không có warm-up.
Benchmark chỉ tính một lượt suy luận để lấy nhãn; SVM không chạy thêm
decision_function trong khoảng đo. Thông số ms/image phụ thuộc phần cứng,
batch và điều kiện chạy.

## Các vấn đề nhận diện trong bản gốc

1. MLP gộp 60.000 train + 10.000 official test, rồi chia thành 56.000 train,
   7.000 validation, 7.000 test. Cách này không dùng cùng tập đánh giá với
   các mô hình còn lại; pipeline mới giữ official test tách biệt.
2. Tiny-ViT cuối cùng fit trên `X_train_cnn` (60.000 ảnh) nhưng dùng
   `X_val_final` (lấy từ 60.000 ảnh này) để validation. Đây là chồng lặp
   train/validation; pipeline mới dùng đúng train partition đã tách.
3. Positional embedding trong code gốc tạo từ `tf.range` ngoài graph rồi cộng
   vào patch tensor. Pipeline mới dùng registered `PositionEmbedding` với
   `add_weight`, bảo đảm tham số được train và đi theo file `.keras`.
   Cách lưu custom layer dựa theo [Keras serialization guide](https://keras.io/guides/serialization_and_saving/).
4. Một số hyperparameter ResNet/ViT trong bước train gốc là số hard-code,
   không đọc trực tiếp study vừa chạy; CLI mới nhận file `best_params.json`.
5. Báo cáo MLP gốc dùng `len(X_test_flat)`=10.000 để mô tả latency dù test
   MLP thực tế 7.000. Project chỉ trích accuracy/F1, không sử dụng latency đó.

Notebook lịch sử giữ nguyên code để truy nguyên; sửa đổi nằm trong package.
Không dùng bảng kết quả cũ để khẳng định pipeline mới đạt accuracy tương tự.

## Explainable AI

Saliency, IG và Grad-CAM dùng logits trước softmax. IG dùng black baseline,
trapezoidal integration và giữ attribution có dấu trong file NPY. Overlay chỉ
hiển thị độ lớn, không thay thế attribution gốc. Grad-CAM dùng activation map
cuối của CNN hoặc residual block cuối ResNet. RF importance là toàn cục.
SVM occlusion đo chênh lệch decision score sau khi đặt patch về 0.

Attention rollout lấy trung bình head, cộng identity, chuẩn hóa theo hàng,
nhân qua hai block và lấy trung bình theo query do model không có CLS token.
Đây là tương tác token, không phải lời giải thích có tính nhân quả hay dành
riêng cho một lớp. `--target` không thay đổi attention rollout.

IG deletion test so sánh xóa pixel theo độ lớn attribution với 5 lần xóa ngẫu
nhiên, trên duy nhất ảnh được chọn. Không diễn giải nó thành kết luận cho toàn
dataset. MNIST có nền đơn giản và chữ số căn giữa; kết quả có thể không áp
dụng cho ảnh thực tế.
