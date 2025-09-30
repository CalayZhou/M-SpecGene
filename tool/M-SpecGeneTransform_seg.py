import torch
from collections import OrderedDict
import json
PATH = "/home/calay/PROJECT/TOP1_M-SpecGene/GITHUB/PRETRAIN_MODEL/"
OUTPATH = "./"
FILE = "M-SpecGene_VIT-B.pth"

original_model = torch.load(PATH+FILE, map_location='cpu')

state_dict = OrderedDict()
for key in original_model['state_dict'].keys():
    value = original_model['state_dict'][key]
    new_key = key
    if 'backbone'  in key:
        if key == 'backbone.ln1.weight':
            new_key = 'backbone.norm.weight'
        if key == 'backbone.ln1.bias':
            new_key = 'backbone.norm.bias'
        new_key = new_key[9:]
        state_dict[new_key] = value

#SEG
torch.save(state_dict,OUTPATH+FILE[:-4]+'_seg_transform.pth')
print("done!")
