import cv2
import numpy as np

def negative_image(st, img):
    # 對影像的每個像素值做位元反轉（bitwise NOT），相當於 new = 255 – old
    return cv2.bitwise_not(img)

def adjust_gamma(st, img):
    # slider("Gamma", 0.1, 5.0, 1.0)：在側邊欄放一個滑桿，範圍 0.1–5.0，預設 1.0。該值控制「亮度曲線」的彎曲程度
    gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0)
    # inv = 1.0 / gamma：實際應用的指數是 γ 的倒數
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)], dtype="uint8")
    return cv2.LUT(img, table)

def adjust_beta(st, img):
    beta = st.sidebar.slider("Brightness Offset", -100, 100, 0)
    # alpha=1（不改變對比度），beta 則直接加在每個像素上，負值會變暗、正值會變亮
    return cv2.convertScaleAbs(img, alpha=1, beta=beta)