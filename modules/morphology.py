import cv2
import numpy as np
# 從 scikit-image 拿來做二值影像的骨架化（skeleton）與細化（thinning）演算法
from skimage.morphology import skeletonize, thin

def morphology_ops(st, img):
    op = st.sidebar.selectbox("Operation", [
        "Erode","Dilate","Open","Close","Gradient",
        "Top Hat","Black Hat","Skeletonize","Thinning"
    ])
    k = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
    iters = st.sidebar.slider("Iterations", 1, 10, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 侵蝕（Erode）
    if op == "Erode":
        res = cv2.erode(gray, kernel, iterations=iters)
    # 膨脹（Dilate）
    elif op == "Dilate":
        res = cv2.dilate(gray, kernel, iterations=iters)
    # 開運算（Open）
    elif op == "Open":
        res = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iters)
    # 關運算（Close）
    elif op == "Close":
        res = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iters)
    # 影像梯度（Gradient）
    elif op == "Gradient":
        res = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=iters)
    # 顶帽（Top Hat）
    elif op == "Top Hat":
        res = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=iters)
    # 黑帽（Black Hat）
    elif op == "Black Hat":
        res = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=iters)
    # 骨架化（Skeletonize）或細化（Thinning）
    elif op == "Skeletonize":
        th = st.sidebar.slider("Threshold", 0, 255, 127)
        bw = gray > th
        skel = skeletonize(bw)
        res = (skel * 255).astype(np.uint8)
    # 細化（Thinning）
    else:
        th = st.sidebar.slider("Threshold", 0, 255, 127)
        bw = gray > th
        thin_img = thin(bw)
        res = (thin_img * 255).astype(np.uint8)
    return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)