import cv2
import numpy as np

def apply_grayscale(st, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 為了保持主程式對「三通道影像」的一致處理流程，再把灰階圖轉回三通道
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)