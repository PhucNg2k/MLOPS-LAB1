# 🚀 PyTorch Lightning với Optuna và MLflow

## 👤 Thông tin sinh viên

| Họ và tên          | MSSV      |
|------------------- |-----------|
| Nguyễn Thượng Phúc | 22521134 |


## Giới thiệu
Project minh họa cách sử dụng **Optuna** để tối ưu hóa siêu tham số, **MLflow** để theo dõi thí nghiệm, và **PyTorch Lightning** để huấn luyện mô hình học sâu. Ví dụ sử dụng bộ dữ liệu FashionMNIST để tích hợp các công cụ này.

## Công nghệ sử dụng
- **Optuna**: Tối ưu hóa siêu tham số.
- **MLflow**: Theo dõi thí nghiệm và trực quan hóa kết quả.
- **PyTorch Lightning**: Đơn giản hóa quá trình huấn luyện mô hình PyTorch.

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

Các siêu tham số tốt nhất và độ chính xác trên tập validation sẽ được hiển thị trên terminal.

Chi tiết thí nghiệm, bao gồm các metric, tham số và artifact, sẽ được lưu trên giao diện MLflow.

## Lưu ý
Đảm bảo rằng mlflow server đã được khởi động trước khi chạy script huấn luyện.

Có thể chỉnh sửa file pl_optuna.py để tùy chỉnh thí nghiệm theo nhu cầu.