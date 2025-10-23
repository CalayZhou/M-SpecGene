import os
import shutil
count = 0
PATH = "/home/calay/DATASET4/SEGMENTATION/MVSeg_Dataset/"
#train
OUTPATH1 = "/home/calay/DATASET4/SEGMENTATION/MVSEG2_T/annotations/validation/"
OUTPATH2 = "/home/calay/DATASET4/SEGMENTATION/MVSEG2_T/images/validation/"
count_l = 0
count_v = 0
with open(PATH+"test.txt",'r') as f:
    for line in  f.readlines():
        line = line.strip('\n')
        # print(line,count)
        count = count+1
        # print(line)
        label_img_path = PATH +'data/'+line+'/label/'
        for label_img_i in os.listdir(label_img_path):
            # print(label_img_path+label_img_i)
            ori_file = label_img_path+label_img_i
            dst_file = OUTPATH1+str(count_l)+'_'+label_img_i.split('.')[0][:-1]+'.png'
            if os.path.exists(dst_file):
                print(ori_file,dst_file)
                count_v = count_v + 1
            shutil.copy(ori_file,dst_file)
            # print(label_img_i, label_img_i.split('.')[0][:-1]+'.png')



            img_path = PATH + 'data/' + line + '/infrared/'
            ori_file2 = img_path+label_img_i.split('.')[0][:-1]+'i.jpg'
            dst_file2 = OUTPATH2+ str(count_l)+'_'+label_img_i.split('.')[0][:-1]+'.jpg'
            if not os.path.exists(ori_file2):
                print("ori_file2")
                ori_file2 = img_path + label_img_i.split('.')[0][:-1] + 'i.png'
                dst_file2 = OUTPATH2 + str(count_l)+'_'+label_img_i.split('.')[0][:-1] + '.png'

            shutil.copy(ori_file2,dst_file2)
            count_l = count_l + 1
print(count_l,count_v)