import cv2

def apply_filter(st, img):
    ftype = st.sidebar.selectbox("Filter", ["Mean","Gaussian","Median","Bilateral"])
    if ftype == "Mean":
        k = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
        return cv2.blur(img, (k, k))
    elif ftype == "Gaussian":
        k = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
        sigma = st.sidebar.slider("Sigma", 0.1, 10.0, 1.0)
        return cv2.GaussianBlur(img, (k, k), sigma)
    elif ftype == "Median":
        k = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
        return cv2.medianBlur(img, k)
    else:
        d = st.sidebar.slider("Diameter", 1, 31, 5, step=2)
        sc = st.sidebar.slider("Sigma Color", 1, 150, 75)
        ss = st.sidebar.slider("Sigma Space", 1, 150, 75)
        return cv2.bilateralFilter(img, d, sc, ss)