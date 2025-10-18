import torch
from collections import OrderedDict

PATH = "/home/calay/PROJECT/TOP1_RGBT_DOWNSTREAM/mmsegmentation-main-rgbt_v0_0618/pretrain/mae_pretrain_vit_base_mmcls.pth"
PATH2 = "/home/calay/PROJECT/TOP1_RGBT_DOWNSTREAM/mmsegmentation-main-rgbt_v0_0618/pretrain/imagenet148w_bs1024_0322_epoch_300_transform.pth"
PATH3 = "/home/calay/PROJECT/TOP1_RGBT_DOWNSTREAM/mmsegmentation-main-rgbt_v0_0618/pretrain/0322cleanv3_dual_in148w_t48w_bs1024_epoch_400.pth"
PATH4 = "/home/calay/PROJECT/TOP1_RGBT_DOWNSTREAM/mmsegmentation-main-rgbt_v0_0618/pretrain/0322cleanv3_dual_in148w_t48w_bs1024_epoch_400_transform.pth"
PATH5 = "/home/calay/PROJECT/TOP1_RGBT_DOWNSTREAM/mmsegmentation-main-rgbt_v0_0618/pretrain/0322_imagenet128w_vit-s_epoch_300.pth"
OUTPATH= "./"
# FILE = "s_epoch_300.pth"
# OUTFILE = "epoch_300.pth"
# PATH1 = "/home/calay/PROJECT/TOP1_RGBTMAE/imagenet_pretrained_model/mine_imagenet/"
# FILE1 = "imagenet_bs736_mine_epoch_300.pth"
model = torch.load(PATH, map_location='cpu')
model2 = torch.load(PATH2, map_location='cpu')
model3 = torch.load(PATH3, map_location='cpu')
model4 = torch.load(PATH4, map_location='cpu')
model5 = torch.load(PATH5, map_location='cpu')
del model['optimizer']
# model1 = torch.load(PATH1+FILE1, map_location='cpu')
# torch.save(model,OUTPATH+OUTFILE)
pass