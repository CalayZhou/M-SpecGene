import os
import numpy as np
import cv2
from tqdm import tqdm
# 将标签为0与1变为0与255

def process_and_check_images(folder_path, threshold=100):
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        
        try:
            # 使用cv2读取图像，确保是灰度图
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise ValueError(f"Image '{filename}' cannot be read.")


            cv2.imwrite(img_path, img*255)
            # print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# 使用示例
folder_path = '/home/wsx/SOD/SOD_Evaluation_Metrics-main/pred_mask/54000/VT5000'  # 替换为你的文件夹路径
process_and_check_images(folder_path)
# VT5000
# VT821
# VT1000
# VIRGBT1500