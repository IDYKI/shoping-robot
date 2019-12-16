import cv2
import os

camera1 = cv2.VideoCapture(0)
i = 0
while(True):
    print(i)
    key = cv2.waitKey(200)
    ret, frame1 = camera1.read()
    i = i+1
    cv2.imwrite('./'+ '/cup-posi/'+str(i)+'.jpg',frame1)
    cv2.imshow('Vision',frame1)
    
    
    if key == 27: # exit on ESC
        break

