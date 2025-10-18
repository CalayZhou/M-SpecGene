import os
import cv2
import numpy as np
from PIL import Image


# PATH = "/home/calay/DATASET4/SEGMENTATION/MVSEG2/annotations/validation/"
# PATH = "/home/calay/DATASET4/SEGMENTATION/MVSEG2/annotations/training/"
# PATH = "/home/calay/DATASET4/SEGMENTATION/ADE20K/ADEChallengeData2016/annotations/training/"
PATH = "/home/calay/DATASET4/SEGMENTATION/SemanticRT/annotations/training/"

total_count = np.zeros(256)

for file in os.listdir(PATH):
    img = cv2.imread(PATH+file,cv2.IMREAD_GRAYSCALE)
    # label = Image.open(f).convert('P')
    # hist = cv2.calcHist(img,channels=0)
    hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    for i in range(256):
        total_count[i] = total_count[i] + hist[i]
    print(np.max(img),np.min(img))
print(total_count)