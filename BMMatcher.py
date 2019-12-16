import numpy as np
import cv2
import matplotlib.pyplot as plt



camera1 = cv2.VideoCapture(0)

img1 =cv2.imread('3.png')
# height, width = img1.shape[:2]
# img1 = cv2.resize(img1, (int(width/10), int(height/10)), interpolation=cv2.INTER_CUBIC)
minHessian = 8000
detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)

while(True):
    ret, img2 = camera1.read()
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    matches = matcher.match(descriptors1, descriptors2)
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
    
    cv2.imshow('Vision',img_matches)
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break















# img1 =cv2.imread('1.jpg')
# img2 =cv2.imread('2.jpg')
# height, width = img1.shape[:2]
# img1 = cv2.resize(img1, (int(width/10), int(height/10)), interpolation=cv2.INTER_CUBIC)
# img2 = cv2.resize(img2, (int(width/10), int(height/10)), interpolation=cv2.INTER_CUBIC)


# minHessian = 400
# detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
# keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
# keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
# #-- Step 2: Matching descriptor vectors with a brute force matcher
# # Since SURF is a floating-point descriptor NORM_L2 is used
# matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
# matches = matcher.match(descriptors1, descriptors2)
# #-- Draw matches
# img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
# cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
# #-- Show detected matches
# cv2.imshow('Matches', img_matches)
# cv2.waitKey()













# gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
# img1=cv2.drawKeypoints(gray,kp,img1)
# cv2.imshow("SIFT", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

