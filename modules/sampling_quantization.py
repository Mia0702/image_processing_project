import cv2
import numpy as np

def quantize(st, img):
    # 在側邊欄建立一個滑桿，標籤是 “Quantization Levels”（量化階數），可選範圍從 2 到 256，預設 16
    levels = st.sidebar.slider("Quantization Levels", 2, 256, 16)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Uniform quantization
    factor = 256 // levels
    quant = (gray // factor) * factor
    return cv2.cvtColor(quant.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# 使用者透過滑桿決定「要把灰階分成幾個階」，可視化地調整量化粗細
# 畫面變化是「顏色分段變得更明顯」，階數越少（levels小），影像越「階梯狀」、越抽象；階數越多，越接近原圖