import cv2
import numpy as np

# 影像梯度 Magnitude
def image_gradient(st, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 滑桿：讓使用者指定 Sobel 核大小 k，範圍 1–31，步長 2（確保奇數核）
    k = st.sidebar.slider("Sobel Kernel Size", 1, 31, 3, step=2)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.magnitude(dx, dy)
    mag = np.uint8(np.clip(mag, 0, 255))
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

# 霍夫直線／圓偵測
def hough_detect(st, img):
    mode = st.sidebar.selectbox("Hough Mode", ["Lines","Circles"])
    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "Lines":
        th = st.sidebar.slider("Threshold", 1, 200, 50)
        lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=th, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                cv2.line(out, (x1,y1), (x2,y2), (0,255,0), 2)
    # 偵測圓形
    else:
        dp = st.sidebar.slider("dp", 1.0, 3.0, 1.2)
        md = st.sidebar.slider("Min Dist", 10, 100, 20)
        p1 = st.sidebar.slider("Param1", 10, 300, 100)
        p2 = st.sidebar.slider("Param2", 10, 100, 30)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, md, param1=p1, param2=p2)
        if circles is not None:
            for x,y,r in np.uint16(np.around(circles[0])):
                cv2.circle(out, (x,y), r, (255,0,0), 2)
    return out