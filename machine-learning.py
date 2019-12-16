import cv2
import numpy as np
import matplotlib.pyplot as plt

# data rand1 woman rand2 man
rand1 = np.array([[155,48],[159,50],
                  [164,53],[168,56]
                  ,[172,60]])
rand2 = np.array([[152,53],[156,55],
                  [160,56],[172,64],
                  [176,65]])

print(rand1)
#lable
lable = np.array([[0],[0],[0],[0],[0],
                  [1],[1],[1],[1],[1]])
#data
data = np.vstack((rand1,rand2)) #合并到一起
data = np.array(data,dtype='float32')

#svm 最基本的要求:所有的数据都要有label<标签>
#[155,48] ->[0] 女生 [152,53] -> [1] 男生
#有标签的训练，叫做监督学习
#监督学习：在每一次训练结束后告诉对还是错
# 0 负样本 1 正样本(数据)

#训练
svm = cv2.ml.SVM_create()
# ml 机器学习模块 SCM_create() 创建
svm.setType(cv2.ml.SVM_C_SVC) # svm type
svm.setKernel(cv2.ml.SVM_LINEAR) # line #线性分类器
svm.setC(0.01)
# 进行训练
result= svm.train(data,cv2.ml.ROW_SAMPLE,lable)
# 预测
pt_data = np.vstack([[170,55],[170,80]]) #女 男
pt_data = np.array(pt_data,dtype='float32')
print(pt_data)
(par1,par2)=svm.predict(pt_data)
print("________________")
print(par1,par2)
# 1 思想 分类器 解决分类问题
# 2 如何？ 寻求一个最优的超平面 分类
# 3 核 ： line 核
# 4 数据的准备 （训练样本） 正样本 负样本 数量可以不一样，一定要标签
# 5 训练 SVM_create train predict

