import cv2
import numpy as np

def detect_edges(st, img):
    method = st.sidebar.selectbox("Method", ["Sobel","Canny"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "Sobel":
        k = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.magnitude(dx, dy)
        mag = np.uint8(np.clip(mag, 0, 255))
        return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    else:
        t1 = st.sidebar.slider("Threshold1", 0, 255, 100)
        t2 = st.sidebar.slider("Threshold2", 0, 255, 200)
        edges = cv2.Canny(gray, t1, t2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)