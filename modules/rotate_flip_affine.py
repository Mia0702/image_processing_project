import cv2
import numpy as np

def process(st, img):
    """Rotate / Flip / Affine transform"""
    (h, w) = img.shape[:2]
    mode = st.sidebar.radio("Mode", ["Rotate", "Flip", "Affine"])

    if mode == "Rotate":
        # 在側邊欄放一個滑桿（slider），標題是 Angle（角度），範圍從 0 到 360，預設值為 0
        # cv2.getRotationMatrix2D 會回傳一個旋轉矩陣，第一個參數是旋轉中心，第二個參數是角度，第三個參數是縮放比例
        angle = st.sidebar.slider("Angle", 0, 360, 0)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    elif mode == "Flip":
        # 在側邊欄放一個下拉選單（selectbox），標題是 Flip（翻轉），選項包括：
        # Horizontal：水平翻轉
        choice = st.sidebar.selectbox("Flip", ["Horizontal", "Vertical", "Both"])
        code = {"Horizontal":1, "Vertical":0, "Both":-1}[choice]
        return cv2.flip(img, code)

    else:  # Affine shear
        pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])
        shift = st.sidebar.slider("Shear", -0.5, 0.5, 0.0)
        pts2 = np.float32([[0,0], [w-1, shift*w], [shift*h, h-1]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(img, M, (w, h))
