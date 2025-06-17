import cv2
import numpy as np

def process(st, img):
    # 在側邊欄（sidebar）放一組單選按鈕（radio），標題是 “Mode”，選項有三種：'Rotate'：旋轉、'Flip'：翻轉、'Affine'：仿射（此例做簡單剪切）
    mode = st.sidebar.radio("Mode", ['Rotate','Flip','Affine'])
    if mode=='Rotate':
        angle = st.sidebar.slider("Angle", 0, 360, 0)
        (h,w) = img.shape[:2]
        # 產生一個 2×3 的旋轉矩陣 M
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w,h))
        return rotated
    elif mode=='Flip':
        # 放一個下拉選單（selectbox），標題 “Flip”，選項有三種翻轉方式：水平、垂直、同時
        f = st.sidebar.selectbox("Flip", ['Horizontal','Vertical','Both'])
        code = {'Horizontal':1,'Vertical':0,'Both':-1}[f]
        return cv2.flip(img, code)
    else:
        # Simple shear affine
        pts1 = np.float32([[0,0],[w-1,0],[0,h-1]])
        shift = st.sidebar.slider("Shear", -0.5,0.5,0.0)
        pts2 = np.float32([[0,0],[w-1,0+shift*w],[0,h-1]])
        M = cv2.getAffineTransform(pts1,pts2)
        return cv2.warpAffine(img, M, (w,h))