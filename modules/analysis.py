import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.morphology import skeletonize, thin

def analyze_objects(st, img):
    mode = st.sidebar.selectbox("分析模式", [
        "標籤與量測",
        "Harris 角點",
        "Shi–Tomasi 角點",
        "ORB 關鍵點",
        "膚色偵測",
        "人臉偵測"
    ])
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #標籤與量測
    if mode == "標籤與量測":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        data = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            comp = (perim**2) / (4 * np.pi * area) if area > 0 else 0
            roundness = (4 * np.pi * area) / (perim**2) if perim > 0 else 0
            data.append({
                "面積": area,
                "周長": perim,
                "緊密度": comp,
                "圓度": roundness
            })
        df = pd.DataFrame(data)
        st.dataframe(df)
        return output
    #Harris 角點偵測
    elif mode == "Harris 角點":
        block = st.sidebar.slider("區塊大小 (blockSize)", 2, 10, 2)
        ksize = st.sidebar.slider("Sob el 核大小 (ksize)", 1, 7, 3)
        k = st.sidebar.slider("自由參數 k", 0.01, 0.1, 0.04)
        dst = cv2.cornerHarris(gray, block, ksize, k)
        dst = cv2.dilate(dst, None)
        output[dst > 0.01 * dst.max()] = [0, 0, 255]
        return output
    #Shi–Tomasi 好特徵偵測
    elif mode == "Shi–Tomasi 角點":
        maxC = st.sidebar.slider("最大角點數量", 1, 100, 50)
        q = st.sidebar.slider("品質水準", 0.01, 0.1, 0.04)
        md = st.sidebar.slider("最小距離", 1, 50, 10)
        corners = cv2.goodFeaturesToTrack(gray, maxC, q, md)
        if corners is not None:
            pts = corners.reshape(-1, 2).astype(int)
            for x, y in pts:
                cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
        return output
    #ORB 關鍵點偵測
    elif mode == "ORB 關鍵點":
        orb = cv2.ORB_create()
        kp = orb.detect(gray, None)
        return cv2.drawKeypoints(img, kp, None, (0, 255, 0), flags=0)
    #膚色偵測
    elif mode == "膚色偵測":
        space = st.sidebar.selectbox("顏色空間", ["HSV", "YCrCb"])
        if space == "HSV":
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 48, 80], dtype="uint8")
            upper = np.array([20, 255, 255], dtype="uint8")
        else:
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            lower = np.array([0, 133, 77], dtype="uint8")
            upper = np.array([255, 173, 127], dtype="uint8")
        mask = cv2.inRange(conv, lower, upper)
        return cv2.bitwise_and(img, img, mask=mask)
    #人臉偵測
    elif mode == "人臉偵測":
        sf = st.sidebar.slider("縮放因子 (scaleFactor)", 1.01, 2.0, 1.1)
        mn = st.sidebar.slider("最小鄰居數 (minNeighbors)", 1, 10, 5)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn)
        for x, y, w, h in faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return output

    # 若無任何模式符合，回傳原圖
    return output
