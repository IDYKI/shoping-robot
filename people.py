
import cv2 

cap = cv2.VideoCapture(0)
# src = cv2.imread('C:/Users/Admin/Desktop/people.jpg')

# # hog特征描述
hog = cv2.HOGDescriptor()    
# 创建SVM检测器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while True:
    ret, src = cap.read()
    cv2.imshow("input", src)
    
    

    # 检测行人
    (rects, weights) = hog.detectMultiScale(src,
                                            winStride=(8, 8),
                                            padding=(32, 32),
                                            scale=1.2,
                                            useMeanshiftGrouping=False)
    for (x, y, w, h) in rects:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("hog-people", src)
    k = cv2.waitKey(30) & 0xff 
    if k == 27: 
        break
    
cap.release() 
cv2.destroyAllWindows()