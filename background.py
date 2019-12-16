import numpy as np 
import cv2

cap = cv2.VideoCapture(0)
# history = 20
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
# fgbg.setHistory(history)

while(1): 
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask) 
    
        # 对原始帧进行膨胀去噪
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    # 获取所有检测框
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #     # 获取矩形框边界坐标
    #     x, y, w, h = cv2.boundingRect(c)
    #     # 计算矩形框的面积
    #     area = cv2.contourArea(c)
    #     if 4000 < area < 6000:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow('frame2',dilated) 
    cv2.imshow("detection", frame)
    k = cv2.waitKey(30) & 0xff 
    if k == 27: 
        break
cap.release() 
cv2.destroyAllWindows()