import cv2
import numpy as np
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
    if op == "Erode":
        res = cv2.erode(gray, kernel, iterations=iters)
    elif op == "Dilate":
        res = cv2.dilate(gray, kernel, iterations=iters)
    elif op == "Open":
        res = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iters)
    elif op == "Close":
        res = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iters)
    elif op == "Gradient":
        res = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=iters)
    elif op == "Top Hat":
        res = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=iters)
    elif op == "Black Hat":
        res = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=iters)
    elif op == "Skeletonize":
        th = st.sidebar.slider("Threshold", 0, 255, 127)
        bw = gray > th
        skel = skeletonize(bw)
        res = (skel * 255).astype(np.uint8)
    else:
        th = st.sidebar.slider("Threshold", 0, 255, 127)
        bw = gray > th
        thin_img = thin(bw)
        res = (thin_img * 255).astype(np.uint8)
    return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)