import cv2
import numpy as np

def detect_edges(st, img):
    method = st.sidebar.selectbox("邊緣偵測方法", ["Sobel 邊緣偵測","Canny 邊緣偵測"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 如果選 Sobel，就先放一個滑桿讓使用者指定卷積核大小 k，範圍 1 到 31，預設 3，且每次跳 2，確保核大小為奇數（Sobel 要用奇數核）
    if method == "Sobel":
        k = st.sidebar.slider("核大小", 1, 31, 3, step=2)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.magnitude(dx, dy)
        mag = np.uint8(np.clip(mag, 0, 255))
        return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    else:
        t1 = st.sidebar.slider("低閾值", 0, 255, 100)
        t2 = st.sidebar.slider("高閾值", 0, 255, 200)
        edges = cv2.Canny(gray, t1, t2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
