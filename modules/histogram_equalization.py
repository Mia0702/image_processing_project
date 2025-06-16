import cv2

def equalize_hist(st, img):
    mode = st.sidebar.selectbox("Equalization Mode", ["Global","CLAHE"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "Global":
        eq = cv2.equalizeHist(gray)
    else:
        clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0)
        tile = st.sidebar.slider("Tile Grid Size", 1, 8, 8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)