import torch
from collections import OrderedDict



PATH = "/home/calay/PROJECT/TOP1_RGBT_DOWNSTREAM/mmsegmentation-main-rgbt_v0_0618/pretrain/"
OUTPATH = PATH
FILE = "crossattention_selfpredict_0620_epoch_500.pth"

DSTPATH = "/home/calay/PROJECT/TOP1_RGBT_DOWNSTREAM/mmsegmentation-main-rgbt_v0_0618/pretrain/"
DSTFILE = "mae_pretrain_vit_base_mmcls.pth"

# OUTPATH =  "./transform/"
# test_model = torch.load(OUTPATH+FILE, map_location='cpu')
original_model = torch.load(PATH+FILE, map_location='cpu')
dst_model = torch.load(DSTPATH+DSTFILE, map_location='cpu')
pass

#MAP_DICT = {"cls_token":"","pos_embed":"","patch_embed":"","cls_token":""}
#step1: construct empty original model
dst_model_copy = dst_model.copy()
state_dict = OrderedDict()
for key, value in dst_model_copy.items():
    new_key = key
    # if 'patch_embed.proj' in key:
    #     new_key = new_key.replace('patch_embed.proj','patch_embed.projection')
    if 'norm.weight' == key:
        new_key = new_key.replace('norm.weight','ln1.weight')
    if 'norm.bias' == key:
        new_key = new_key.replace('norm.bias','ln1.bias')
    # if 'patch_embed.proj' in key:
    #     new_key = key.replace('patch_embed.proj','patch_embed.projection')
    # if 'patch_embed.proj' in key:
    #     new_key = key.replace('patch_embed.proj','patch_embed.projection')
    ori_key = 'backbone.' + new_key
    if ori_key not in original_model['state_dict'].keys():
        print("del",key,ori_key)
    else:
        old_value = dst_model[key]
        new_value = original_model['state_dict'][ori_key]
        #if old_value.size() ==new_value.size():
        state_dict[key] = new_value
        print("save",key)
pass
# out_model = {}
# out_model['model'] = out_model
out_model = state_dict
torch.save(out_model,OUTPATH+FILE[:-4]+'_transform.pth')
#step2: copy the model parameter