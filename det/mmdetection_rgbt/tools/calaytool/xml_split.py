import os
import shutil

PATH = "/home/calay/DATASET4/object detection/LLVIP/Annotations/"
OUTPATH = "/home/calay/DATASET4/object detection/LLVIP/Annotations_train/"
IMAGEPATH = '/home/calay/DATASET4/object detection/LLVIP/infrared/train/'
for file in os.listdir(PATH):
    file_name = file.split('.')[0]+'.jpg'
    if file_name in  os.listdir(IMAGEPATH):
        ori_file = PATH+file
        dst_file = OUTPATH+file
        shutil.copy(ori_file,dst_file)