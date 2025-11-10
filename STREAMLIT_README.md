# FallSense - Hướng dẫn sử dụng Streamlit

## 🚀 Chạy ứng dụng Streamlit

Sau khi đã cài đặt tất cả dependencies, bạn có thể chạy ứng dụng Streamlit bằng lệnh:

```bash
cd FallSense
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt web tại địa chỉ: `http://localhost:8501`

## 📋 Tính năng

### 1. Chế độ Camera 📹
- Sử dụng camera trực tiếp để phát hiện ngã trong thời gian thực
- Tự động ghi video khi phát hiện ngã (nếu bật)
- Hiển thị keypoints (skeleton) nếu bật

### 2. Chế độ Video File 📁
- Upload và xử lý file video
- Phát hiện ngã trong toàn bộ video
- Hiển thị tiến trình xử lý

### 3. Chế độ Hình ảnh 🖼️
- Upload và phân tích hình ảnh
- Phát hiện ngã trong hình ảnh tĩnh

## ⚙️ Cài đặt

### Sidebar - Cài đặt

1. **Chọn chế độ**: Camera, Video File, hoặc Image
2. **Hiển thị keypoints**: Bật/tắt hiển thị skeleton
3. **Lật ngang camera**: Lật ngang hình ảnh từ camera
4. **Tự động ghi khi phát hiện ngã**: 
   - Bật tính năng này để tự động ghi video khi phát hiện ngã
   - Nhập đường dẫn thư mục để lưu video

## 📁 Cấu trúc thư mục

```
FallSense/
├── app.py                    # Ứng dụng Streamlit chính
├── Main_Gui.py              # Ứng dụng PyQt5 (gốc)
├── weights/
│   └── fall_detection_person.pt  # Model weights
├── src/
│   └── Fall_detection.py    # Module phát hiện ngã
└── requirements.txt         # Dependencies
```

## 🔧 Troubleshooting

### Lỗi không tìm thấy model weights
- Đảm bảo file `weights/fall_detection_person.pt` tồn tại
- Nếu chưa có, model sẽ tự động được tải từ Hugging Face khi chạy lần đầu

### Lỗi camera không hoạt động
- Kiểm tra quyền truy cập camera trong trình duyệt
- Đảm bảo không có ứng dụng khác đang sử dụng camera

### Lỗi import module
- Đảm bảo đã cài đặt tất cả dependencies: `pip install -r requirements.txt`
- Kiểm tra Python version (khuyến nghị Python 3.9+)

## 📝 Ghi chú

- Ứng dụng Streamlit dễ sử dụng và phát triển hơn so với PyQt5
- Có thể deploy lên Streamlit Cloud để chia sẻ dễ dàng
- Model weights được cache sau lần tải đầu tiên

## 🌐 Deploy lên Streamlit Cloud

1. Push code lên GitHub
2. Đăng ký tài khoản tại [streamlit.io](https://streamlit.io)
3. Kết nối repository và deploy
4. Model weights sẽ tự động được tải từ Hugging Face

