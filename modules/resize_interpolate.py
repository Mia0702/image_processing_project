import cv2

def resize_image(st, img):
    w = st.sidebar.slider("Width", 50, img.shape[1], img.shape[1])
    h = st.sidebar.slider("Height", 50, img.shape[0], img.shape[0])
    # 在側邊欄放一個下拉選單（selectbox），標題是 Interpolation（插值方法），選項包括：
    # INTER_NEAREST：最近鄰插值，速度最快但畫質最差
    # INTER_LINEAR：線性插值，平衡速度與品質（OpenCV 預設）
    # INTER_AREA：區域插值，適合縮小影像時保留細節
    # INTER_CUBIC：立方插值，比線性更平滑，但比較慢
    # INTER_LANCZOS4：Lanczos（8×8 區域）插值，品質最佳但最慢
    interp = st.sidebar.selectbox(
    "插值方法", ['最近鄰','線性插值','區域插值','三次插值','Lanczos 插值']
    )
    # getattr(cv2, interp) 會動態從 cv2 模組取得該插值方法的常數值（整數）
    interp_flag = getattr(cv2, interp)
    return cv2.resize(img, (w, h), interpolation=interp_flag)
