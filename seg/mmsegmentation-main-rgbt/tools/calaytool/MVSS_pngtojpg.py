import cv2
import os

PATH = "/home/calay/DATASET4/SEGMENTATION/MVSEG2_T/images/training/"

for file in os.listdir(PATH):
    if '.png' in file:
        img = cv2.imread(PATH+file)
        print(file)
        cv2.imwrite(PATH+file.split('.')[0]+'.jpg',img)