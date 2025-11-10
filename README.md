# 🚨 FallSense - Cảnh báo trượt ngã và đột quỵ

**FallSense** là ứng dụng AI mã nguồn mở với giao diện web hiện đại, sử dụng mô hình YOLOv7 (PyTorch) đã được fine-tune để phát hiện ngã và đột quỵ trong thời gian thực. Được thiết kế để giám sát bệnh nhân cao tuổi và những người có nguy cơ, FallSense cung cấp cảnh báo tức thì, ghi video tự động và phân tích dữ liệu để giúp người chăm sóc và chuyên gia y tế phản ứng nhanh chóng và hiệu quả.

---

## ✨ Tính năng

### 1. Phát hiện ngã và đột quỵ thời gian thực
- Sử dụng mô hình YOLOv7 state-of-the-art, được fine-tune trên dữ liệu CCTV thực tế
- Nhanh, chính xác và đáng tin cậy - hoạt động với camera trực tiếp hoặc file video

### 2. Ghi video tự động
- Tự động ghi và lưu video khi phát hiện ngã hoặc đột quỵ
- Video được lưu cục bộ để xem lại, phân tích y tế hoặc retrain model

### 3. Phân tích và xem lại
- Tải và phân tích các sự kiện đã ghi
- Hữu ích cho cả chuyên gia y tế và nhà nghiên cứu AI

### 4. Giao diện người dùng thân thiện
- **Streamlit**: Giao diện web hiện đại, dễ sử dụng và phát triển
- **PyQt5**: Giao diện desktop truyền thống (tùy chọn)
- Tùy chọn hiển thị keypoints (skeleton)
- Cài đặt tùy chỉnh cho ghi video, lưu trữ và lật camera

---

## 🚀 Cài đặt & Thiết lập

### 1. **Chuẩn bị môi trường**

```bash
# Cài đặt Miniconda (nếu chưa có)
# Tạo môi trường ảo mới
conda create -n fallsense python=3.9
conda activate fallsense
```

### 2. **Clone Repository**

```bash
git clone https://github.com/vatallus/canh-bao-truot-nga-va-dot-quy.git
cd canh-bao-truot-nga-va-dot-quy
```

### 3. **Cài đặt Dependencies**

```bash
pip install -r requirements.txt
```

### 4. **Tải Model Weights**

Model weights sẽ tự động được tải từ Hugging Face khi chạy ứng dụng lần đầu, hoặc bạn có thể tải thủ công:

```bash
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ngotphong/FallSense', filename='fall_detection_person.pt', local_dir='weights')"
```

### 5. **Chạy Ứng dụng**

#### Chế độ Streamlit (Khuyến nghị - Dễ sử dụng)

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại: `http://localhost:8501`

#### Chế độ PyQt5 (Desktop)

```bash
python Main_Gui.py
```

---

## 📸 Hướng dẫn sử dụng

### Streamlit Interface

1. **Chọn chế độ**:
   - 📹 **Camera**: Phát hiện ngã từ camera trực tiếp
   - 📁 **Video File**: Upload và xử lý file video
   - 🖼️ **Image**: Phân tích hình ảnh tĩnh

2. **Cài đặt**:
   - Bật/tắt hiển thị keypoints (skeleton)
   - Bật/tắt lật ngang camera
   - Bật tự động ghi video khi phát hiện ngã
   - Chọn thư mục lưu video

3. **Sử dụng**:
   - Ứng dụng sẽ tự động phát hiện ngã trong thời gian thực
   - Khi phát hiện ngã, sẽ hiển thị cảnh báo màu đỏ
   - Nếu bật tự động ghi, video sẽ được lưu vào thư mục đã chọn

---

## 📂 Cấu trúc Dự án

```
FallSense/
├── app.py                    # Ứng dụng Streamlit (Giao diện web)
├── Main_Gui.py               # Ứng dụng PyQt5 (Giao diện desktop)
├── weights/
│   └── fall_detection_person.pt  # Model weights (tự động tải)
├── src/
│   ├── Fall_detection.py    # Module phát hiện ngã
│   ├── Main.py               # Logic xử lý chính
│   └── config.py             # Cấu hình
├── GUI/                      # Tài nguyên giao diện PyQt5
├── models/                   # YOLOv7 model architecture
├── utils/                    # Utilities và helper functions
├── .streamlit/               # Cấu hình Streamlit
│   └── config.toml
├── requirements.txt          # Dependencies
├── README.md                 # File này
└── STREAMLIT_README.md       # Hướng dẫn chi tiết Streamlit
```

---

## 🛠️ Yêu cầu Hệ thống

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **OpenCV**: 4.6+
- **Streamlit**: 1.50+ (cho giao diện web)
- **PyQt5**: 5.15+ (cho giao diện desktop, tùy chọn)

### GPU (Tùy chọn)
- CUDA-compatible GPU để tăng tốc độ xử lý
- Ứng dụng vẫn hoạt động tốt trên CPU

---

## 📝 Tính năng Chi tiết

### Phát hiện Ngã
- Phân tích keypoints của cơ thể (skeleton)
- Phát hiện dựa trên vị trí vai, chân và tỷ lệ chiều cao/rộng
- Độ chính xác cao với dữ liệu thực tế

### Ghi Video Tự động
- Tự động bắt đầu ghi khi phát hiện ngã
- Tự động dừng sau khi không còn phát hiện ngã
- Lưu video với timestamp

### Hiển thị Keypoints
- Vẽ skeleton (xương) của người được phát hiện
- Giúp hiểu rõ hơn về cách model phát hiện ngã

---

## 🤝 Đóng góp

Đóng góp rất được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📄 License

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🙏 Lời cảm ơn

- [YOLOv7 by WongKinYiu](https://github.com/WongKinYiu/yolov7)
- [Hugging Face Model Hosting](https://huggingface.co/ngotphong/FallSense)
- PyTorch, OpenCV, Streamlit, PyQt5 và cộng đồng mã nguồn mở

---

## 📬 Liên hệ

Để đặt câu hỏi, hỗ trợ hoặc hợp tác, vui lòng:
- Mở issue trên GitHub
- Liên hệ qua email: vatallus@users.noreply.github.com

---

## 🌟 Stars

Nếu dự án này hữu ích, hãy cho một ⭐ trên GitHub!

---

**Lưu ý**: Model weights (~161MB) sẽ tự động được tải từ Hugging Face khi chạy ứng dụng lần đầu. Đảm bảo có kết nối internet ổn định.
