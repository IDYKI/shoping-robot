import cv2
import numpy as np
import xml.dom.minidom as xmldom

# i = 80
camera1 = cv2.VideoCapture(0)
# image = cv2.imread("./"+ "/cup-posi/"+str(i)+'.jpg')

# 加载训练好的模型
svm = cv2.ml.SVM_load('./svm_data.dat')
# 创建hog特征描述子函数
hog = cv2.HOGDescriptor()
print(1)
while True:
    ret, image = camera1.read()
    image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    # 原图太大，降低原图分辨率
    test_img = image
    # test_img = cv2.resize(image, (0, 0), fx=1, fy=1)
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # 获取大小
    h, w = test_img.shape[:2]

       
    # 为了筛选框，记录框坐标总和以及框的个数，为了最后求出所有候选框的均值框
    sum_x = 0
    sum_y = 0
    count = 0

    # 为了加快计算，窗口滑动的步长为4，一个cell是8个像素
    for row in range(64, h-64, 4):
        for col in range(32, w-32, 4):
            win_roi = gray[row-64:row+64,col-32:col+32]
            hog_desc = hog.compute(win_roi, winStride=(8, 8), padding=(0, 0))
            one_fv = np.zeros([len(hog_desc)], dtype=np.float32)
            for i in range(len(hog_desc)):
                one_fv[i] = hog_desc[i][0]
            one_fv = one_fv.reshape(-1, len(hog_desc))
            # 预测
            result = svm.predict(one_fv)[1]
            # 统计正样本 
            if result[0][0] > 0:
                sum_x += (col-32)
                sum_y += (row-64)
                count += 1
                # 画出所有框
                cv2.rectangle(test_img, (col-32, row-64), (col+32, row+64), (0, 233, 255), 1, 8, 0)
    if count:
        # 求取均值框
        x = sum_x // count
        y = sum_y // count
        # 画出均值框
        cv2.rectangle(test_img, (x, y), (x+64, y+128), (0, 0, 255), 2, 8, 0)


    cv2.imshow('Vision',image)
    cv2.imshow('Vision2',test_img)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyAllWindows
