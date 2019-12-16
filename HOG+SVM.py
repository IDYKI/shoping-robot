import cv2
import os
import numpy as np
import xml.dom.minidom as xmldom




def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    # print(eles.tagName)
    xmin = eles.getElementsByTagName("xmin")[0].firstChild.data
    xmax = eles.getElementsByTagName("xmax")[0].firstChild.data
    ymin = eles.getElementsByTagName("ymin")[0].firstChild.data
    ymax = eles.getElementsByTagName("ymax")[0].firstChild.data
    print(xmin, xmax, ymin, ymax)
    return xmin, xmax, ymin, ymax


def test_parse_xml(num):
    parse_xml('./'+'/cup_label/'+str(num,)+'.xml')


# 把目标图放在64x128的灰色图片中间，方便计算描述子
def get_hog_descriptor(image,label):
    global photoname

    hog = cv2.HOGDescriptor()
    # if label==1:
    #     photoname +=1
    #     try:
    #         x1, x2, y1, y2 = test_parse_xml(photoname)
    #         image = image[x1:x2,y1:y2]
    #     except :
    #         pass

    h ,w = image.shape[:2]
    rate = 64 / w
    image = cv2.resize(image, (64, np.int(rate*h)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = np.zeros((128, 64), dtype=np.uint8)
    bg[:,:] = 127
    h, w = gray.shape
    dy = (128 - h) // 2
    bg[dy:h+dy,:] = gray
    descriptors = hog.compute(bg, winStride=(8, 8), padding=(0, 0))
    return descriptors

def get_data(train_data, labels, path, lableType):
    
    for file_name in os.listdir(path):
        img_dir = os.path.join(path, file_name)
        img = cv2.imread(img_dir)
        hog_desc = get_hog_descriptor(img,lableType)    
        one_fv = np.zeros([len(hog_desc)], dtype=np.float32)
        for i in range(len(hog_desc)):
            one_fv[i] = hog_desc[i][0]
        train_data.append(one_fv)
        labels.append(lableType)
        if lableType==1:
            print(1)
        else:
            print(2)
    return train_data, labels

def get_dataset(pdir, ndir):
    train_data = []
    labels = []
    # 获取正样本
    train_data, labels =  get_data(train_data, labels, pdir, lableType=1)
    # 获取负样本
    train_data, labels =  get_data(train_data, labels, ndir, lableType=-1)

    return np.array(train_data, dtype=np.float32), np.array(labels, dtype=np.int32)

def svm_train(pdir, ndir):
    # 创建SVM
    svm = cv2.ml.SVM_create()
    # 设置相应的SVM参数
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.setC(10)
    svm.setGamma(0.01)
    # 获取正负样本和labels
    trainData, responses = get_dataset(pdir, ndir)
    # reshape (n,)-->(n,1)
    responses = np.reshape(responses, [-1, 1])
    # 训练
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')



if __name__ == '__main__':
    # train_data的shape为(n, 3780), labels(n,)
    # n为样本数  
    photoname = 0
    svm_train("./"+ "/cup/",'./'+ '/cup-nega/')
    
    
    
    