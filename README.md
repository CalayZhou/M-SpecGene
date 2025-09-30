# M-SpecGene
M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision (ICCV 2025)





## Brief Introduction

RGB-Thermal (RGBT) multispectral vision is essential for robust perception in complex environments. Most RGBT tasks follow a case-by-case research paradigm, relying on manually customized models to learn task-oriented representations. Nevertheless, this paradigm is inherently constrained by artificial inductive bias, modality bias, and data bottleneck. To address these limitations, we make the initial attempt to build a Generalized RGBT MultiSpectral foundation model (M-SpecGene), which aims to learn modality-invariant representations from large-scale broad data in a self-supervised manner. M-SpecGene provides new insights into multispectral fusion and integrates prior case-by-case studies into a unified paradigm. Considering the unique characteristic of information imbalance in RGBT data, we introduce the Cross-Modality Structural Sparsity
(CMSS) metric to quantify the information density across two modalities. Then we develop the GMM-CMSS progressive masking strategy to facilitate a flexible, easy-to-hard, and object-centric pre-training process. Comprehensive experiments validate M-SpecGene’s generalizability across eleven datasets for four RGBT downstream tasks.


<div align="center" style="width:image width px;">
  <img  src="img/methodv2.png" width=2000>
</div>


## RGBT550K Dataset
<div align="center" style="width:image width px;">
  <img  src="img/dataset.png" width=2000>
</div>

To pretrain a multispectral foundation model with robust generalization capabilities, we exert our utmost efforts to make a comprehensive collection of available RGBT datasets. The  multispectral (RGBT) image datasets can be found at [A Summary of Multispectral (RGBT) Image Datasets](https://github.com/CalayZhou/A-Summary-of-Multispectral-Image-Datasets). Our meticulous collection and preprocessing yields RGBT550K, a comprehensive dataset comprising 548,238 high-quality samples. It encompasses diverse scenarios, tasks, lighting conditions, resolutions, and object categories, providing a solid foundation for the self-supervised pre-training of the multispectral foundation model. You can download the RGBT550K dataset from  [Baidu Cloud (code: rwf7)](https://pan.baidu.com/s/1Hv3E74ILsk_rmbQVXDpr6w?pwd=rwf7) or [One Dirve](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/calayzhou_smail_nju_edu_cn/EjVdEZ6zjzJHk2Q8n8Swgl8BAqEPVg3jbN62Y096PXlRwQ?e=CCImyC).

```
# RGBT550K Usage
sudo apt install p7zip-full
7z x RGBT550K_archive.7z.partaa
```

## Pretrained Models

### a) Pretrained Foundation Model
| Foundation Model | Backbone | Model Weights |
|:----------------------------| :------: |:-------------:|
| M-SpecGene                  | ViT-B |     [M-SpecGene_VIT-B.pth](https://drive.google.com/file/d/1COVlrnQPoqFjK-aJpSvNb3diSDblw0nb/view?usp=sharing)            | 


### b) Transform of M-SpecGene for Training on Downstream Tasks
Since the above pretrained foundation model M-SpecGene retains all parameters during
self-supervised training, we extract the encoder  for detection (ViTDet) and segmentation (UperNet) task.
```
cd tool
python M-SpecGeneTransform_det.py  # M-SpecGene_VIT-B_det_transform.pth
python M-SpecGeneTransform_seg.py  # M-SpecGene_VIT-B_seg_transform.pth
```

| Task         | Backbone |             Model Weights              |
|:-------------| :------: |:--------------------------------------:|
| Detection    | ViT-B | [M-SpecGene_VIT-B_det_transform.pth](https://drive.google.com/file/d/111OG0Ejv8pd8nLdLs74f1rq7NgvSTd8e/view?usp=sharing) | 
| Segmentation | ViT-B | [M-SpecGene_VIT-B_seg_transform.pth](https://drive.google.com/file/d/1xUH48fAqTtznNHh0B_WG04ww5ps0BdNk/view?usp=drive_link) | 

### c) Trained Models for Different RGBT Datasets 

| Task         |    Dataset     |      Trained Models       | Performance |
|:-------------|:--------------:|:-------------------------:|:-----------:|
| Detection    |   [KAIST](https://drive.google.com/file/d/1UpIfZkqH1ry-252HVF_iUDXJX3GcYYvR/view?usp=sharing)    |                           |             |
| Detection  |   [LLVIP](https://drive.google.com/file/d/1vUY8zr5RaQvs0Umz4nQCCU5pEkSkWAtw/view?usp=sharing)    |                           |             | 
| Detection    |    [FLIR](https://drive.google.com/file/d/1kn4YXlUmKU-OuWwDmXLIbRfWuTJ5-D-P/view?usp=sharing)    |                           |             | 
| Segmentation | [SemanticRT](https://drive.google.com/file/d/16rrFDl468R3TGp-C3bPkwVSlhElIAh-4/view?usp=sharing) |  [SRT_iter_320000.pth](https://drive.google.com/file/d/1vEuW_a3n7_-IXpphOQnJv5CMZFGW_N9y/view?usp=sharing)  | mIoU 79.84% | 
| Segmentation   |   [MVSEG](https://drive.google.com/file/d/1AM0ln1ZzDQ_9nHolfMNugyHS6Ls2pHbZ/view?usp=sharing)    | [MVSEG_iter_240000.pth](https://drive.google.com/file/d/1INkxRRygPObU3-WIVx84CT42p_LczXb6/view?usp=sharing) | mIoU 63.02% |
| Segmentation |    [FMB](https://drive.google.com/file/d/11n_S8SMD2mSzWYw-V8Bhmu_jhnkXAl7-/view?usp=drive_link)     |  [FMB_iter_224000.pth](https://drive.google.com/file/d/1tP1sCzEdBya2G3sWEggglKkxrNX94NGC/view?usp=sharing)  |  mIoU ~60%  |


## Usage
### Pretraining
code will come soon.

### Finetuning

#### 1) RGBT Multispectral Object Detection


#### 2) RGBT Multispectral Semantic Segmentation



a. dataset preparation

PLease download the  [SemanticRT](https://drive.google.com/file/d/16rrFDl468R3TGp-C3bPkwVSlhElIAh-4/view?usp=sharing),  [MVSEG](https://drive.google.com/file/d/1AM0ln1ZzDQ_9nHolfMNugyHS6Ls2pHbZ/view?usp=sharing) and  [FMB](https://drive.google.com/file/d/11n_S8SMD2mSzWYw-V8Bhmu_jhnkXAl7-/view?usp=drive_link)  datasets to the proposal path.
```
# link the dataset (MVSEG by default)
cd SEG/mmsegmentation-main-rgbt
ln -s /path/to/MVSEG_ALL/MVSEG  ./data/ade/ADEChallengeData2016
ln -s /path/to/MVSEG_ALL/MVSEG_T  ./data/ade/ADEChallengeData2016_T
```

b. Installation

Please refer to [mmsegmentation-v1.2.2 get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/docs/en/get_started.md#installation)
for installation. You can also refer to the [mmseg_env_refer.txt](./seg/mmsegmentation-main-rgbt/mmseg_env_refer.txt) to check the version.

```
# after installation
cd seg/mmsegmentation-main-rgbt
pip install -v -e .
```


c.Evalution  (MVSEG by default)
```
python tools/test.py configs/mae/mae-base_upernet_8xb2-amp-320k_ade20k-768x768.py /path/to/SRT_iter_320000.pth
```

d.Train (MVSEG by default)

Please download the [M-SpecGene_VIT-B_seg_transform.pth](https://drive.google.com/file/d/1xUH48fAqTtznNHh0B_WG04ww5ps0BdNk/view?usp=drive_link), and change the pretrained model path in `configs/mae/mae-base_upernet_8xb2-amp-320k_ade20k-768x768.py`
```
bash tools/dist_train.sh configs/mae/mae-base_upernet_8xb2-amp-320k_ade20k-768x768.py 2
```

e. Evalution or Train on the other datasets

```
1. change the dataset link in /dataset/ade/
2. change the mmseg/datasets/ade.py (refer to ade_FMB.py ade_MVSEG.py ade_SRT.py)
3. change num_classes (FMB->15, MVSEG->26, SRT->13) in configs/mae
/mae-base_upernet_8xb2-amp-320k_ade20k-768x768.py
4. train or evalution as above
```



## Citation
```
@article{zhou2025m,
  title={M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision},
  author={Zhou, Kailai and Yang, Fuqiang and Wang, Shixian and Wen, Bihan and Zi, Chongde and Chen, Linsen and Shen, Qiu and Cao, Xun},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
