import cv2

def resize_image(st, img):
    w = st.sidebar.slider("Width", 50, img.shape[1], img.shape[1])
    h = st.sidebar.slider("Height", 50, img.shape[0], img.shape[0])
    interp = st.sidebar.selectbox(
        "Interpolation", ['INTER_NEAREST','INTER_LINEAR','INTER_AREA','INTER_CUBIC','INTER_LANCZOS4']
    )
    interp_flag = getattr(cv2, interp)
    return cv2.resize(img, (w, h), interpolation=interp_flag)