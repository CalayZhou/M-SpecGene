import torch
from collections import OrderedDict
import json
PATH = "/home/calay/PROJECT/TOP1_M-SpecGene/GITHUB/PRETRAIN_MODEL/"
OUTPATH = "./"
FILE = "M-SpecGene_VIT-B.pth"

original_model = torch.load(PATH+FILE, map_location='cpu')


state_dict = OrderedDict()
with open('model_keys.json', 'r') as f:
    loaded_keys = json.load(f)
for key in loaded_keys:
    new_key = key
    if 'patch_embed.proj' in key:
        new_key = new_key.replace('patch_embed.proj','patch_embed.projection')
    if 'block' in key:
        new_key = new_key.replace('blocks','layers')
    if 'norm.weight' in key:
        new_key = new_key.replace('norm.weight','ln1.weight')
    if 'norm.bias' in key:
        new_key = new_key.replace('norm.bias','ln1.bias')
    if 'norm' in key:
        new_key = new_key.replace('norm','ln')
    if 'mlp.fc1' in key:
        new_key = new_key.replace('mlp.fc1','ffn.layers.0.0')
    if 'mlp.fc2' in key:
        new_key = new_key.replace('mlp.fc2','ffn.layers.1')
    # if 'patch_embed.proj' in key:
    #     new_key = key.replace('patch_embed.proj','patch_embed.projection')
    # if 'patch_embed.proj' in key:
    #     new_key = key.replace('patch_embed.proj','patch_embed.projection')
    ori_key = 'backbone.' + new_key
    if ori_key not in original_model['state_dict'].keys():
        print('lack',key,ori_key)
    # old_value = dst_model['model'][key]
    new_value = original_model['state_dict'][ori_key]
    # if old_value.size() == new_value.size():
    state_dict[key] = new_value
    print(key, new_key)
pass
out_model = {}
out_model['model'] = state_dict
#DET
torch.save(out_model,OUTPATH+FILE[:-4]+'_det_transform.pth')
#SEG
# torch.save(state_dict,OUTPATH+FILE[:-4]+'_seg_transform.pth')
# print("done!")
