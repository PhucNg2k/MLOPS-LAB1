# 🚀 PyTorch Lightning với Optuna và MLflow

## 👤 Thông tin sinh viên

| Họ và tên           | MSSV      |
|---------------------|-----------|
| Nguyễn Thượng Phúc  | 22521134  |

---

## 🔧 Giới thiệu chung về Pipeline

Dự án này xây dựng một **pipeline huấn luyện mô hình học sâu** sử dụng:
- **Optuna** để tự động tối ưu siêu tham số.
- **MLflow** để tracking thí nghiệm (logs, checkpoint, hyperparams,...).
- **PyTorch Lightning** để đơn giản hóa training loop và tổ chức code rõ ràng.

Pipeline hoạt động hoàn toàn tự động:
1. Train mạng neural network với bộ dataset FashionMnist
2. **Tạo study Optuna** để chạy nhiều trial huấn luyện với siêu tham số khác nhau, tối ưu theo validation accuracy.
3. **MLflow** ghi lại mọi thông tin của từng trial: model, val/test accuracy, checkpoint,...
4. Tự động dừng sớm trial kém hiệu quả với `EarlyStopping` và `PruningCallback`.
5. Tự động lưu và tải lại checkpoint có validation accuracy tốt nhất.

> 🔥 **Điểm mới / sáng tạo**:  
> - Kết hợp đầy đủ cả 3 công cụ hiện đại: PytorchLightning + Optuna + MLflow.
> - Cấu trúc lại pipeline để dễ mở rộng, dễ quản lý logs, mô hình.
> - Có thể chạy chỉ với 1 dòng lệnh (`python pl_optuna.py -p`), mọi thứ còn lại được tự động hóa.

---

## 🧠 Công nghệ sử dụng và đặc điểm nổi bật

| Công nghệ          | Vai trò                                                       |
|-------------------|---------------------------------------------------------------|
| **PyTorch Lightning** | Tổ chức training loop sạch, hỗ trợ callback, logger tự động |
| **Optuna**         | Tối ưu hóa siêu tham số với pruning                          |
| **MLflow**         | Ghi log thí nghiệm, checkpoint, model, metric                |
| **FashionMNIST**   | Dataset minh họa (ảnh thời trang 28x28)                      |

---

## Hướng dẫn cài đặt

1. Clone dự án từ GitHub:
   ```bash
   git clone https://github.com/PhucNg2k/MLOPS-LAB1.git
   cd MLOPS-LAB1
2. Cài đặt các thư viện cần thiết:
   ```bash
    pip install -r requirements.txt
## Hướng dẫn sử dụng
**Bước 1: Khởi động MLflow Server**

Mở một terminal và chạy lệnh sau để khởi động MLflow server:

    mlflow server --host localhost --port 8080

Giao diện MLflow sẽ được hiển thị tại địa chỉ http://localhost:8080.

**Bước 2: Chạy script huấn luyện**

Mở một terminal khác và chạy script huấn luyện với tính năng pruning:

    python pl_optuna.py -p

## Kết quả

Kết quả từng trial sẽ hiện ở terminal (accuracy, params,...).

MLflow UI sẽ lưu trữ:

+ Hyperparameters, metrics

+ Checkpoint model tốt nhất

+ Logs, artifacts

Tên run sẽ được đặt theo định danh dễ đọc (VD: Trial:0_14/04/2025_18:32)

## Lưu ý
Đảm bảo rằng mlflow server đã được khởi động trước khi chạy script huấn luyện.

Có thể chỉnh sửa file pl_optuna.py để tùy chỉnh thí nghiệm theo nhu cầu.

## 🎥 Video Demo

[![Demo Video](https://img.youtube.com/vi/mela8dFpKq0/0.jpg)](https://www.youtube.com/watch?v=mela8dFpKq0)