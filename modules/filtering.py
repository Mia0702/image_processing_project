import cv2

def apply_filter(st, img):
    ftype = st.sidebar.selectbox("濾波方法", ["平均濾波","高斯濾波","中值濾波","雙邊濾波"])
    # 均值濾波
    if ftype == "Mean":
        k = st.sidebar.slider("核大小", 1, 31, 3, step=2)
        return cv2.blur(img, (k, k))
    # 高斯濾波
    elif ftype == "Gaussian":
        k = st.sidebar.slider("核大小", 1, 31, 3, step=2)
        sigma = st.sidebar.slider("Sigma值", 0.1, 10.0, 1.0)
        return cv2.GaussianBlur(img, (k, k), sigma)
    # 中值濾波
    elif ftype == "Median":
        k = st.sidebar.slider("核大小", 1, 31, 3, step=2)
        return cv2.medianBlur(img, k)
    # 雙邊濾波
    else:
        d = st.sidebar.slider("Diameter", 1, 31, 5, step=2)
        sc = st.sidebar.slider("色彩σ", 1, 150, 75)
        ss = st.sidebar.slider("空間σ", 1, 150, 75)
        return cv2.bilateralFilter(img, d, sc, ss)
