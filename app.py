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
import threading
from datetime import datetime

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
if 'fall_count' not in st.session_state:
    st.session_state.fall_count = 0
if 'fall_history' not in st.session_state:
    st.session_state.fall_history = []
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Chưa bắt đầu"

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
    st.header("📊 Thống kê")
    st.metric("Số lần phát hiện ngã", st.session_state.fall_count)
    
    if st.session_state.fall_history:
        st.subheader("Lịch sử phát hiện")
        for i, fall_time in enumerate(reversed(st.session_state.fall_history[-5:]), 1):
            st.text(f"{i}. {fall_time}")
    
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
            st.exception(e)
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
    
    # Lấy kích thước frame
    height, width = frame.shape[:2]
    
    # Tạo padded image để hiển thị (giữ nguyên kích thước)
    padded_img = frame.copy()
    
    # Chạy inference với scale = 1.0, pad = 0 vì không resize
    img_result, is_fall = st.session_state.fall_detector.inference_and_draw_on_display(
        orig_img, padded_img, 1.0, 0, 0, width, height
    )
    
    # Thêm text cảnh báo nếu phát hiện ngã
    if is_fall:
        # Vẽ text cảnh báo lớn
        text = "PHAT HIEN NGA!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        color = (0, 0, 255)  # Đỏ
        
        # Tính toán vị trí text (giữa màn hình)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = (width - text_width) // 2
        y = (height + text_height) // 2
        
        # Vẽ background cho text
        cv2.rectangle(img_result, 
                     (x - 10, y - text_height - 10), 
                     (x + text_width + 10, y + baseline + 10), 
                     (0, 0, 0), -1)
        
        # Vẽ text
        cv2.putText(img_result, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Vẽ border đỏ
        cv2.rectangle(img_result, (0, 0), (width-1, height-1), (0, 0, 255), 10)
    
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
    
    # Nút điều khiển camera
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("▶️ Bắt đầu Camera", type="primary"):
            if st.session_state.video_capture is None:
                st.session_state.video_capture = cv2.VideoCapture(0)
                if st.session_state.video_capture.isOpened():
                    st.session_state.camera_active = True
                    st.session_state.current_status = "Đang chạy"
                    st.success("✅ Camera đã được khởi động!")
                else:
                    st.error("❌ Không thể mở camera. Vui lòng kiểm tra kết nối.")
            else:
                st.session_state.camera_active = True
                st.session_state.current_status = "Đang chạy"
    
    with col_btn2:
        if st.button("⏹️ Dừng Camera"):
            st.session_state.camera_active = False
            if st.session_state.video_capture:
                st.session_state.video_capture.release()
                st.session_state.video_capture = None
            stop_recording()
            st.session_state.current_status = "Đã dừng"
            st.info("⏹️ Camera đã được dừng.")
    
    with col_btn3:
        if st.button("🔄 Reset"):
            st.session_state.fall_count = 0
            st.session_state.fall_history = []
            st.session_state.current_status = "Đã reset"
            st.success("🔄 Đã reset thống kê!")
    
    # Hiển thị trạng thái
    status_col1, status_col2 = st.columns([3, 1])
    
    with status_col1:
        status_placeholder = st.empty()
    
    with status_col2:
        metric_placeholder = st.empty()
    
    # Hiển thị video
    video_placeholder = st.empty()
    
    # Sử dụng st.camera_input cho camera ổn định hơn
    if st.session_state.camera_active:
        camera_input = st.camera_input("Camera đang hoạt động", key="camera_stream")
        
        if camera_input is not None:
            try:
                # Chuyển đổi từ PIL Image sang numpy array
                if isinstance(camera_input, Image.Image):
                    img_array = np.array(camera_input.convert('RGB'))
                else:
                    camera_input.seek(0)
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
                    
                    # Xử lý frame
                    img_result, is_fall = process_frame(img_bgr)
                    
                    # Cập nhật thống kê nếu phát hiện ngã
                    if is_fall:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # Chỉ thêm vào lịch sử nếu chưa có trong 1 giây gần nhất
                        if not st.session_state.fall_history or \
                           (datetime.now() - datetime.strptime(st.session_state.fall_history[-1], "%Y-%m-%d %H:%M:%S")).total_seconds() > 1:
                            st.session_state.fall_count += 1
                            st.session_state.fall_history.append(current_time)
                            st.session_state.last_fall_time = time.time()
                    
                    # Hiển thị kết quả
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                        st.image(img_result_rgb, caption="Kết quả phát hiện", use_container_width=True)
                    
                    with col2:
                        # Hiển thị trạng thái với cảnh báo rõ ràng (không nhấp nháy)
                        if is_fall:
                            status_placeholder.markdown(
                                f"""
                                <div style='text-align: center; padding: 20px; background-color: #ff4444; border-radius: 10px; border: 5px solid #ff0000;'>
                                    <h1 style='color: white; font-size: 48px; margin: 0;'>🚨 CẢNH BÁO!</h1>
                                    <h2 style='color: white; font-size: 32px; margin: 10px 0;'>PHÁT HIỆN NGÃ</h2>
                                    <p style='color: white; font-size: 18px; margin: 5px 0;'>Thời gian: {datetime.now().strftime("%H:%M:%S")}</p>
                                    <p style='color: white; font-size: 16px; margin: 5px 0; font-weight: bold;'>VUI LÒNG KIỂM TRA NGAY!</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            metric_placeholder.metric("Trạng thái", "🚨 NGÃ", delta="Cảnh báo", delta_color="inverse")
                            
                            # Hiệu ứng chỉ chạy một lần
                            if 'last_fall_display' not in st.session_state or st.session_state.last_fall_display != current_time:
                                st.balloons()
                                st.session_state.last_fall_display = current_time
                        else:
                            status_placeholder.markdown(
                                f"""
                                <div style='text-align: center; padding: 20px; background-color: #44ff44; border-radius: 10px;'>
                                    <h2 style='color: white; font-size: 32px; margin: 0;'>✅ Bình thường</h2>
                                    <p style='color: white; font-size: 16px; margin: 5px 0;'>Thời gian: {datetime.now().strftime("%H:%M:%S")}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            metric_placeholder.metric("Trạng thái", "✅ Bình thường", delta=None)
                        
                        # Xử lý ghi video tự động
                        if auto_record and st.session_state.save_folder:
                            if is_fall:
                                if not st.session_state.recording:
                                    start_recording(img_bgr)
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
                                st.sidebar.warning("🔴 Đang ghi video...")
                            else:
                                st.sidebar.info("⏸️ Không ghi")
                else:
                    st.error(f"❌ Lỗi: Không thể xử lý hình ảnh từ camera. Shape: {img_array.shape if 'img_array' in locals() else 'N/A'}")
            except Exception as e:
                st.error(f"❌ Lỗi xử lý camera: {str(e)}")
                st.exception(e)
    else:
        # Hiển thị placeholder khi camera chưa bật
        st.info("👆 Nhấn 'Bắt đầu Camera' để bắt đầu phát hiện ngã")
        video_placeholder.image("GUI/images/backgroud-placeholder.png" if os.path.exists("GUI/images/backgroud-placeholder.png") else None, 
                               caption="Camera chưa được khởi động", use_container_width=True)

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
            fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Thanh tiến trình
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Hiển thị video
            video_placeholder = st.empty()
            
            frame_count = 0
            fall_detected_in_video = False
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                # Xử lý frame
                try:
                    img_result, is_fall = process_frame(frame)
                    
                    if is_fall:
                        fall_detected_in_video = True
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if not st.session_state.fall_history or \
                           (datetime.now() - datetime.strptime(st.session_state.fall_history[-1], "%Y-%m-%d %H:%M:%S")).total_seconds() > 1:
                            st.session_state.fall_count += 1
                            st.session_state.fall_history.append(current_time)
                    
                    # Hiển thị
                    img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(img_result_rgb, channels="RGB", use_container_width=True)
                    
                    # Cập nhật tiến trình
                    frame_count += 1
                    progress = frame_count / total_frames if total_frames > 0 else 0
                    progress_bar.progress(progress)
                    
                    # Hiển thị trạng thái
                    if is_fall:
                        status_text.markdown(
                            f"""
                            <div style='text-align: center; padding: 15px; background-color: #ff4444; border-radius: 10px; border: 3px solid #ff0000;'>
                                <h2 style='color: white; font-size: 24px; margin: 0;'>🚨 PHÁT HIỆN NGÃ!</h2>
                                <p style='color: white; font-size: 14px; margin: 5px 0;'>Frame {frame_count}/{total_frames} - Thời gian: {datetime.now().strftime('%H:%M:%S')}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        status_text.markdown(
                            f"""
                            <div style='text-align: center; padding: 15px; background-color: #44ff44; border-radius: 10px;'>
                                <h2 style='color: white; font-size: 24px; margin: 0;'>✅ Bình thường</h2>
                                <p style='color: white; font-size: 14px; margin: 5px 0;'>Frame {frame_count}/{total_frames}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    time.sleep(1.0 / fps)  # Giữ tốc độ video gốc
                except Exception as e:
                    st.error(f"❌ Lỗi xử lý frame: {str(e)}")
                    break
            
            video_capture.release()
            os.unlink(tfile.name)
            
            if fall_detected_in_video:
                st.error("🚨 **Đã phát hiện ngã trong video!**")
            else:
                st.success("✅ **Không phát hiện ngã trong video**")

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
        try:
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
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 20px; background-color: #ff4444; border-radius: 10px; border: 5px solid #ff0000;'>
                        <h1 style='color: white; font-size: 48px; margin: 0; animation: blink 1s infinite;'>🚨 CẢNH BÁO!</h1>
                        <h2 style='color: white; font-size: 32px; margin: 10px 0;'>PHÁT HIỆN NGÃ</h2>
                        <p style='color: white; font-size: 18px; margin: 5px 0;'>Thời gian: {datetime.now().strftime("%H:%M:%S")}</p>
                    </div>
                    <style>
                        @keyframes blink {{
                            0%, 100% {{ opacity: 1; }}
                            50% {{ opacity: 0.5; }}
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.balloons()
            else:
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 20px; background-color: #44ff44; border-radius: 10px;'>
                        <h2 style='color: white; font-size: 32px; margin: 0;'>✅ Bình thường</h2>
                        <p style='color: white; font-size: 16px; margin: 5px 0;'>Không phát hiện ngã trong hình ảnh</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"❌ Lỗi xử lý hình ảnh: {str(e)}")
            st.exception(e)

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
