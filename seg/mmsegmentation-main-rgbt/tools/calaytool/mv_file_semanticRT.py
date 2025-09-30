
import os
import shutil
count = 0
PATH = "/home/calay/DATASET4/SEGMENTATION/33.SemanticRT_dataset/SemanticRT_dataset/"
#train
OUTPATH1 = "/home/calay/DATASET4/SEGMENTATION/SemanticRT/annotations/training/"
OUTPATH2 = "/home/calay/DATASET4/SEGMENTATION/SemanticRT/images/training/"
count_l = 0
count_v = 0
with open(PATH+"train.txt",'r') as f:
    for line in  f.readlines():
        line = line.strip('\n')
        # print(line,count)
        count = count+1
        # print(line)
        label_img_path = PATH +'labels/' + line +'.png'
        # for label_img_i in os.listdir(label_img_path):
            # print(label_img_path+label_img_i)
        ori_file = label_img_path#+label_img_i
        dst_file = OUTPATH1+ line +'.png'
        shutil.copy(ori_file,dst_file)


        img_path = PATH + 'rgb/' + line +'.jpg'
        ori_file2 = img_path
        dst_file2 = OUTPATH2+ line +'.jpg'
        # if not os.path.exists(ori_file2):
        #     ori_file2 = img_path + label_img_i.split('.')[0][:-1] + 'v.png'
        #     dst_file2 = OUTPATH2 + str(count_l)+'_'+label_img_i.split('.')[0][:-1] + '.png'

        shutil.copy(ori_file2,dst_file2)
        count_l = count_l + 1
print(count_l,count_v)