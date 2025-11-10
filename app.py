"""
FallSense - Streamlit Application
Ứng dụng phát hiện ngã và đột quỵ sử dụng YOLOv7 với giao diện Streamlit
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
import tempfile

# Import các module từ dự án FallSense
from src.Fall_detection import FallDetector

# Cấu hình trang
st.set_page_config(
    page_title="FallSense - Phát hiện ngã và đột quỵ",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo session state
if 'fall_detector' not in st.session_state:
    st.session_state.fall_detector = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'video_writer' not in st.session_state:
    st.session_state.video_writer = None
if 'save_folder' not in st.session_state:
    st.session_state.save_folder = None
if 'show_keypoints' not in st.session_state:
    st.session_state.show_keypoints = False
if 'flip_horizontal' not in st.session_state:
    st.session_state.flip_horizontal = False
if 'last_fall_time' not in st.session_state:
    st.session_state.last_fall_time = None

# Tiêu đề chính
st.title("🚨 FallSense - Phát hiện ngã và đột quỵ")
st.markdown("---")

# Sidebar - Cài đặt
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    # Chọn chế độ
    mode = st.radio(
        "Chọn chế độ:",
        ["📹 Camera", "📁 Video File", "🖼️ Image"],
        index=0
    )
    
    # Tùy chọn hiển thị keypoints
    st.session_state.show_keypoints = st.checkbox(
        "Hiển thị keypoints (skeleton)",
        value=st.session_state.show_keypoints
    )
    
    # Tùy chọn lật ngang
    st.session_state.flip_horizontal = st.checkbox(
        "Lật ngang camera",
        value=st.session_state.flip_horizontal
    )
    
    # Chọn thư mục lưu
    auto_record = st.checkbox("Tự động ghi khi phát hiện ngã", value=False)
    
    if auto_record:
        save_folder_input = st.text_input(
            "Thư mục lưu video:",
            value=st.session_state.save_folder or "./recordings",
            help="Nhập đường dẫn thư mục để lưu video khi phát hiện ngã"
        )
        if save_folder_input:
            st.session_state.save_folder = save_folder_input
            os.makedirs(st.session_state.save_folder, exist_ok=True)
    
    st.markdown("---")
    st.header("ℹ️ Thông tin")
    st.info("""
    **FallSense** sử dụng mô hình YOLOv7 để phát hiện:
    - Ngã (Fall detection)
    - Đột quỵ (Stroke detection)
    
    Ứng dụng có thể hoạt động với:
    - Camera trực tiếp
    - Video file
    - Hình ảnh
    """)

# Khởi tạo model
@st.cache_resource
def load_model():
    """Tải mô hình YOLOv7 một lần và cache"""
    model_path = "weights/fall_detection_person.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model weights tại: {model_path}")
        st.stop()
    
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    detector = FallDetector(
        model_path,
        device,
        show_keypoints=st.session_state.show_keypoints
    )
    return detector

# Tải model
if st.session_state.fall_detector is None:
    with st.spinner("Đang tải mô hình YOLOv7..."):
        try:
            st.session_state.fall_detector = load_model()
            st.session_state.fall_detector.show_keypoints = st.session_state.show_keypoints
            st.success("✅ Mô hình đã được tải thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
            st.stop()

# Cập nhật show_keypoints cho detector
if st.session_state.fall_detector:
    st.session_state.fall_detector.show_keypoints = st.session_state.show_keypoints

# Hàm xử lý frame
def process_frame(frame, orig_img=None):
    """Xử lý một frame và trả về kết quả"""
    if st.session_state.flip_horizontal:
        frame = cv2.flip(frame, 1)
    
    if orig_img is None:
        orig_img = frame.copy()
    
    # Tạo padded image để hiển thị
    height, width = frame.shape[:2]
    padded_img = frame.copy()
    
    # Chạy inference
    img_result, is_fall = st.session_state.fall_detector.inference_and_draw_on_display(
        orig_img, padded_img, 1.0, 0, 0, width, height
    )
    
    return img_result, is_fall

# Hàm bắt đầu ghi video
def start_recording(frame):
    """Bắt đầu ghi video khi phát hiện ngã"""
    if not st.session_state.recording and st.session_state.save_folder:
        st.session_state.recording = True
        filename = time.strftime("recording_%Y_%m_%d_%H_%M_%S.mp4")
        save_path = os.path.join(st.session_state.save_folder, filename)
        height, width = frame.shape[:2]
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        st.session_state.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        return True
    return False

# Hàm dừng ghi video
def stop_recording():
    """Dừng ghi video"""
    if st.session_state.recording and st.session_state.video_writer:
        st.session_state.video_writer.release()
        st.session_state.video_writer = None
        st.session_state.recording = False
        return True
    return False

# Xử lý theo chế độ
if mode == "📹 Camera":
    st.subheader("📹 Chế độ Camera")
    
    # Sử dụng st.camera_input cho camera đơn giản hơn
    camera_input = st.camera_input("Bật camera để bắt đầu")
    
    if camera_input is not None:
        try:
            # Chuyển đổi từ PIL Image sang numpy array
            # st.camera_input trả về PIL Image hoặc BytesIO
            if isinstance(camera_input, Image.Image):
                # Chuyển PIL Image sang numpy array (RGB format)
                img_array = np.array(camera_input.convert('RGB'))
            else:
                # Nếu là BytesIO, đọc lại bằng PIL
                camera_input.seek(0)  # Reset về đầu file
                img_pil = Image.open(camera_input)
                img_array = np.array(img_pil.convert('RGB'))
            
            # Kiểm tra định dạng và chuyển đổi
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Đảm bảo là uint8
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                
                # Chuyển đổi RGB sang BGR cho OpenCV
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                st.error(f"❌ Lỗi: Không thể xử lý hình ảnh từ camera. Shape: {img_array.shape if 'img_array' in locals() else 'N/A'}")
                st.stop()
            
            # Xử lý frame
            img_result, is_fall = process_frame(img_bgr)
            
            # Hiển thị kết quả
            col1, col2 = st.columns([2, 1])
            
            with col1:
                img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                st.image(img_result_rgb, caption="Kết quả phát hiện", use_container_width=True)
            
            with col2:
                # Hiển thị trạng thái
                if is_fall:
                    st.error("🚨 **PHÁT HIỆN NGÃ!**")
                    st.balloons()  # Hiệu ứng khi phát hiện ngã
                else:
                    st.success("✅ **Bình thường**")
                
                # Xử lý ghi video tự động
                if auto_record and st.session_state.save_folder:
                    if is_fall:
                        if not st.session_state.recording:
                            start_recording(img_bgr)
                            st.session_state.last_fall_time = time.time()
                        if st.session_state.video_writer:
                            st.session_state.video_writer.write(img_bgr)
                    else:
                        # Dừng ghi sau 2 giây không phát hiện ngã
                        if st.session_state.recording:
                            if st.session_state.last_fall_time:
                                if time.time() - st.session_state.last_fall_time > 2:
                                    stop_recording()
                    
                    # Hiển thị trạng thái ghi
                    if st.session_state.recording:
                        st.warning("🔴 Đang ghi video...")
                    else:
                        st.info("⏸️ Không ghi")
        except Exception as e:
            st.error(f"❌ Lỗi xử lý camera: {str(e)}")
            st.exception(e)

elif mode == "📁 Video File":
    st.subheader("📁 Chế độ Video File")
    
    uploaded_file = st.file_uploader(
        "Chọn file video",
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_file is not None:
        # Lưu file tạm
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Mở video
        video_capture = cv2.VideoCapture(tfile.name)
        
        if not video_capture.isOpened():
            st.error("❌ Không thể mở file video.")
        else:
            st.success("✅ Video đã được tải!")
            
            # Lấy thông tin video
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Thanh tiến trình
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Hiển thị video
            video_placeholder = st.empty()
            
            frame_count = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                # Xử lý frame
                img_result, is_fall = process_frame(frame)
                
                # Hiển thị
                img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                video_placeholder.image(img_result_rgb, channels="RGB", use_container_width=True)
                
                # Cập nhật tiến trình
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                
                # Hiển thị trạng thái
                if is_fall:
                    status_text.error(f"🚨 **PHÁT HIỆN NGÃ!** - Frame {frame_count}/{total_frames}")
                else:
                    status_text.success(f"✅ **Bình thường** - Frame {frame_count}/{total_frames}")
                
                time.sleep(1.0 / fps)  # Giữ tốc độ video gốc
            
            video_capture.release()
            os.unlink(tfile.name)
            st.success("✅ Đã xử lý xong video!")

elif mode == "🖼️ Image":
    st.subheader("🖼️ Chế độ Hình ảnh")
    
    uploaded_file = st.file_uploader(
        "Chọn hình ảnh",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Đọc hình ảnh
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Chuyển đổi RGB sang BGR cho OpenCV
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Xử lý
        img_result, is_fall = process_frame(img_bgr)
        
        # Hiển thị kết quả
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Hình ảnh gốc", use_container_width=True)
        
        with col2:
            img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
            st.image(img_result_rgb, caption="Kết quả phát hiện", use_container_width=True)
        
        # Hiển thị trạng thái
        if is_fall:
            st.error("🚨 **PHÁT HIỆN NGÃ!**")
        else:
            st.success("✅ **Bình thường**")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>FallSense - Phát hiện ngã và đột quỵ sử dụng YOLOv7</p>
        <p>Powered by PyTorch & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

