
-------------------------------------
calayzhou 2024.10.17
利用本代码 验证upernet（VIT-B）在RGBT显著性检测数据集 VI-RGBT VT5000  语义分割MFNet PST900 的效果
Rgbt salient object detection:A large-scale dataset and benchmark. （VT5000 优先）
Multiple Graph Affinity Interactive Network and A Variable Illumination Dataset for RGBT Image Salient Object Detection（VI-RGBT优先）
PST900: RGB-Thermal Calibration, Dataset and Segmentation Network， ICRA2019
Multispectral object detection for autonomous vehicles，ACMMM 2017

-------------------------------------

#1.环境安装
mmsegmentation环境要求即可，无特殊要求


#2.本代码改动
本代码为了处理RGB+T多光谱图像，而原始mmsegmentation是针对RGB单模态图像做分割的，因此在数据读取、网络部分做了
简单的修改
【有道云笔记】ViT UperNet代码修改  【主要修改部分在黄色框——2024年6月20日14:53:16读取两个模态的图像】
https://note.youdao.com/s/YOo5DWAz


#3.运行环境示例
4090*8服务器 （用户名spectrum 密码citepsec  ssp -p 10151 test@114.212.163.92)
docker exec -it god2 /bin/bash
conda activate mmseg
cd /home/calay/mmsegmentation-main-rgbt_v1_0622_copy_semanticRT_1007


注：可以通过该god2容器或者god_dockerimages_202312.tar镜像创建自己的docker容器


数据集：
按照mmsegmentation要求的格式


数据集示例：
/data/ade 目录下  设置超链接
ADEChallengeData2016 -> /home/calay/DATASET5/SemanticRT_ALL/SemanticRT/
ADEChallengeData2016_T -> /home/calay/DATASET5/SemanticRT_ALL/SemanticRT_T


SemanticRT_T.zip
链接: https://pan.baidu.com/s/16ypLO5184uXs8jH9m55qIA?pwd=sp8n 提取码: sp8n 
--来自百度网盘超级会员v4的分享
SemanticRT.zip
链接: https://pan.baidu.com/s/1D1LRj3d_o3Glb1eUe63maw?pwd=qpv3 提取码: qpv3 
--来自百度网盘超级会员v4的分享




#4.本代码训练

bash tools/dist_train.sh configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-768x768.py 2
或
python tools/train.py configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-768x768.py


#5.预训练模型
configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-768x768.py中，默认采用
1006_vit-b_v9.2_54w_bs1024_decode4_epoch_500_mmseg_transform.pth


更换数据集步骤：
# step1 dataset info
/mmsegmentation-main/mmseg/datasets/ade.py

#step2  class num
/configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-768x768.py

# step3
/data/ade/
