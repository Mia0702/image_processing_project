import cv2
import numpy as np

def segment(st, img):
    method = st.sidebar.selectbox("Segmentation", [
        "Global Threshold","Adaptive Mean","Adaptive Gaussian","Watershed","GrabCut"
    ])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "Global Threshold":
        th = st.sidebar.slider("Threshold", 0, 255, 127)
        _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif method == "Adaptive Mean":
        bs = st.sidebar.slider("Block Size", 3, 51, 11, step=2)
        C = st.sidebar.slider("C", 0, 20, 2)
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
        x = st.sidebar.slider("Rect x", 0, img.shape[1], 10)
        y = st.sidebar.slider("Rect y", 0, img.shape[0], 10)
        w = st.sidebar.slider("Rect w", 10, img.shape[1], img.shape[1]//2)
        h = st.sidebar.slider("Rect h", 10, img.shape[0], img.shape[0]//2)
        rect = (x, y, w, h)
        bgd = np.zeros((1,65), dtype=np.float64)
        fgd = np.zeros((1,65), dtype=np.float64)
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        return img * mask2[:,:,np.newaxis]