import cv2
import numpy as np

def quantize(st, img):
    levels = st.sidebar.slider("Quantization Levels", 2, 256, 16)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Uniform quantization
    factor = 256 // levels
    quant = (gray // factor) * factor
    return cv2.cvtColor(quant.astype(np.uint8), cv2.COLOR_GRAY2BGR)