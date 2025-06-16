import streamlit as st
import numpy as np
import cv2
from modules import (
    grayscale, sampling_quantization, resize_interpolate,
    rotate_flip_affine, histogram_equalization, edge_detection,
    enhancement, filtering, features, segmentation,
    morphology, analysis
)

# 讓整個頁面左右寬度可以拉滿，兩張圖顯示不會擠在一起
st.set_page_config(
    page_title="數位影像處理應用程式",
    layout="wide"  # 允許使用 full-width
)

st.title("數位影像處理應用程式")

# 側邊欄：上傳圖片、選擇操作
uploaded_file = st.sidebar.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
operations = {
    '灰階 (Grayscale)': grayscale.apply_grayscale,
    '取樣與量化 (Sampling & Quantization)': sampling_quantization.quantize,
    '縮放與插值 (Resize & Interpolate)': resize_interpolate.resize_image,
    '旋轉/翻轉/仿射 (Rotate/Flip/Affine)': rotate_flip_affine.process,
    '直方圖均衡 (Histogram Equalization)': histogram_equalization.equalize_hist,
    '邊緣偵測 (Sobel/Canny)': edge_detection.detect_edges,
    '負片 (Negative)': enhancement.negative_image,
    'Gamma 矯正 (Gamma)': enhancement.adjust_gamma,
    '亮度調整 (Beta)': enhancement.adjust_beta,
    '濾波 (Mean/Gaussian/Median/Bilateral)': filtering.apply_filter,
    '影像梯度 (Image Gradient)': features.image_gradient,
    '霍夫線條/圓偵測 (Hough)': features.hough_detect,
    '分割 (Threshold & Segmentation)': segmentation.segment,
    '形態學 (Morphology)': morphology.morphology_ops,
    '標籤與分析 (Label & Analysis)': analysis.analyze_objects
}
choice = st.sidebar.selectbox("選擇操作", list(operations.keys()))

if uploaded_file:
    # 讀取為 OpenCV 圖片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 呼叫對應的處理函式
    result = operations[choice](st, img)

    # 左右兩欄顯示：左邊原圖、右邊處理後圖
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原圖")
        st.image(img, channels="BGR", use_container_width=True)
    with col2:
        st.subheader("處理後")
        st.image(result, channels="BGR", use_container_width=True)
else:
    st.info("🔸 先從左側上傳一張圖片，然後選擇想要的影像處理操作。")
