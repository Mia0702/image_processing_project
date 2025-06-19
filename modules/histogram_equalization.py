import cv2

def equalize_hist(st, img):
    # "Global"：全域直方圖均衡、"CLAHE"：限制對比度的自適應直方圖均衡
    mode = st.sidebar.selectbox("均衡模式", ["全域均衡","CLAHE 自適應均衡"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "Global":
        eq = cv2.equalizeHist(gray)
    else:
        # 建立一個滑桿，範圍 1.0–10.0，預設 2.0，用來控制「限幅因子」。它決定局部直方圖中每個灰階桶（bin）最大能增強多少，防止過度對比拉扯
        clip = st.sidebar.slider("限幅因子", 1.0, 10.0, 2.0)
        # 建立另一個滑桿，範圍 1–8，預設 8，用來指定把整張圖分成多少格（tile）做局部均衡
        tile = st.sidebar.slider("區塊大小", 1, 8, 8)
        # 使用 cv2.createCLAHE() 建立一個 CLAHE 物件，並用 apply() 方法對灰階圖進行均衡化
        # clipLimit 是限制對比度的參數，tileGridSize 是每個 tile 的大小
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        eq = clahe.apply(gray)
    # 最後把處理後的灰階影像 eq 再轉回三通道 BGR（好讓主程式統一用彩色格式顯示），並回傳這張圖
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
