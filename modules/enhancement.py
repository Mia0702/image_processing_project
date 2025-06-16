import cv2
import numpy as np

def negative_image(st, img):
    return cv2.bitwise_not(img)

def adjust_gamma(st, img):
    gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0)
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)], dtype="uint8")
    return cv2.LUT(img, table)

def adjust_beta(st, img):
    beta = st.sidebar.slider("Brightness Offset", -100, 100, 0)
    return cv2.convertScaleAbs(img, alpha=1, beta=beta)