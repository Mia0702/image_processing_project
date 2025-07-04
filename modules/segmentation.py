import cv2
import numpy as np

def segment(st, img):
    # 全域閾值、平均法自適應閾值、高斯法自適應閾值、分水嶺（Watershed）與 GrabCut
    method = st.sidebar.selectbox("分割方法", [
    "全域閾值","自適應閾值(平均法)","自適應閾值(高斯法)","分水嶺","GrabCut"
    ])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "Global Threshold":
        th = st.sidebar.slider("門檻值", 0, 255, 127)
        _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif method == "Adaptive Mean":
        bs = st.sidebar.slider("區塊大小", 3, 51, 11, step=2)
        C = st.sidebar.slider("C 值", 0, 20, 2)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, bs, C)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif method == "Adaptive Gaussian":
        bs = st.sidebar.slider("Block Size", 3, 51, 11, step=2)
        C = st.sidebar.slider("C", 0, 20, 2)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, bs, C)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif method == "Watershed":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.7*dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [0,0,255]
        return img
    else:
        mask = np.zeros(img.shape[:2], np.uint8)
        x = st.sidebar.slider("矩形 X", 0, img.shape[1], 10)
        y = st.sidebar.slider("矩形 Y", 0, img.shape[0], 10)
        w = st.sidebar.slider("矩形寬度", 10, img.shape[1], img.shape[1]//2)
        h = st.sidebar.slider("矩形高度", 10, img.shape[0], img.shape[0]//2)
        rect = (x, y, w, h)
        bgd = np.zeros((1,65), dtype=np.float64)
        fgd = np.zeros((1,65), dtype=np.float64)
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        return img * mask2[:,:,np.newaxis]
