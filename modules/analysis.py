import cv2
import numpy as np
import pandas as pd

def analyze_objects(st, img):
    mode = st.sidebar.selectbox("Analysis", [
        "Labeling & Measurements","Harris","Shi-Tomasi",
        "ORB Keypoints","Skin Detection","Face Detection"
    ])
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "Labeling & Measurements":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        data = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            comp = (perim**2)/(4*np.pi*area) if area>0 else 0
            roundness = (4*np.pi*area)/(perim**2) if perim>0 else 0
            data.append({"Area": area, "Perimeter": perim,
                         "Compactness": comp, "Roundness": roundness})
        df = pd.DataFrame(data)
        st.dataframe(df)
        return output
    elif mode == "Harris":
        block = st.sidebar.slider("Block Size", 2, 10, 2)
        ksize = st.sidebar.slider("KS Size", 1, 7, 3)
        k = st.sidebar.slider("k", 0.01, 0.1, 0.04)
        dst = cv2.cornerHarris(gray, block, ksize, k)
        dst = cv2.dilate(dst, None)
        output[dst > 0.01 * dst.max()] = [0, 0, 255]
        return output
    elif mode == "Shi-Tomasi":
        maxC = st.sidebar.slider("Max Corners", 1, 100, 50)
        q = st.sidebar.slider("Quality Level", 0.01, 0.1, 0.04)
        md = st.sidebar.slider("Min Distance", 1, 50, 10)
        corners = cv2.goodFeaturesToTrack(gray, maxC, q, md)
        if corners is not None:
            for x,y in np.int0(corners):
                cv2.circle(output, (x,y), 5, (0,255,0), -1)
        return output
    elif mode == "ORB Keypoints":
        orb = cv2.ORB_create()
        kp = orb.detect(gray, None)
        return cv2.drawKeypoints(img, kp, None, (0,255,0), flags=0)
    elif mode == "Skin Detection":
        space = st.sidebar.selectbox("Color Space", ["HSV","YCrCb"])
        if space == "HSV":
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0,48,80], dtype="uint8")
            upper = np.array([20,255,255], dtype="uint8")
        else:
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            lower = np.array([0,133,77], dtype="uint8")
            upper = np.array([255,173,127], dtype="uint8")
        mask = cv2.inRange(conv, lower, upper)
        return cv2.bitwise_and(img, img, mask=mask)
    else:
        sf = st.sidebar.slider("Scale Factor", 1.01, 2.0, 1.1)
        mn = st.sidebar.slider("Min Neighbors", 1, 10, 5)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, sf, mn)
        for x,y,w,h in faces:
            cv2.rectangle(output, (x,y), (x+w,y+h), (255,0,0), 2)
        return output