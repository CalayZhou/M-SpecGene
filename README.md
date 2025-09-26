# M-SpecGene
M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision (ICCV 2025)

## Brief Introduction
RGB-Thermal (RGBT) multispectral vision is essential for robust perception in complex environments. Most RGBT tasks follow a case-by-case research paradigm, relying on manually customized models to learn task-oriented representations. Nevertheless, this paradigm is inherently constrained by artificial inductive bias, modality bias, and data bottleneck. To address these limitations, we make the initial attempt to build a Generalized RGBT MultiSpectral foundation model (M-SpecGene), which aims to learn modality-invariant representations from large-scale broad data in a self-supervised manner. M-SpecGene provides new insights into multispectral fusion and integrates prior case-by-case studies into a unified paradigm. Considering the unique characteristic of information imbalance in RGBT data, we introduce the Cross-Modality Structural Sparsity
(CMSS) metric to quantify the information density across two modalities. Then we develop the GMM-CMSS progressive masking strategy to facilitate a flexible, easy-to-hard, and object-centric pre-training process. Comprehensive experiments validate M-SpecGene’s generalizability across eleven datasets for four RGBT downstream tasks.



## RGBT550K Dataset
To pretrain a multispectral foundation model with robust generalization capabilities, we exert our utmost efforts to make a comprehensive collection of available RGBT datasets. The  multispectral (RGBT) image datasets can be found at [A Summary of Multispectral (RGBT) Image Datasets](https://github.com/CalayZhou/A-Summary-of-Multispectral-Image-Datasets). Our meticulous collection and preprocessing yields RGBT550K, a comprehensive dataset comprising 548,238 high-quality samples. It encompasses diverse scenarios, tasks, lighting conditions, resolutions, and object categories, providing a solid foundation for the self-supervised pre-training of the multispectral foundation model. You can download the RGBT550K dataset from  [Baidu Cloud (code: rwf7)](https://pan.baidu.com/s/1Hv3E74ILsk_rmbQVXDpr6w?pwd=rwf7) or [One Dirve](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/calayzhou_smail_nju_edu_cn/EjVdEZ6zjzJHk2Q8n8Swgl8BAqEPVg3jbN62Y096PXlRwQ?e=CCImyC).

```
`RGBT550K Usage`
sudo apt install p7zip-full
7z x RGBT550K_archive.7z.partaa
```

## Pretrained Models


| Pretrain | Backbone | Model Weights |
| :------- | :------: | :------: |
| M-SpecGene | ViT-S | | 
| M-SpecGene | ViT-B | | 



## Usage
### Pretraining


### Finetuning

1. RGBT Multispectral Object Detection

2. RGBT Multispectral Semantic Segmentation

3. RGBT Cross-modality Feature Matching

4. RGBT Multispectral Salient Object Detection


## Citation
```
@article{zhou2025m,
  title={M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision},
  author={Zhou, Kailai and Yang, Fuqiang and Wang, Shixian and Wen, Bihan and Zi, Chongde and Chen, Linsen and Shen, Qiu and Cao, Xun},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
