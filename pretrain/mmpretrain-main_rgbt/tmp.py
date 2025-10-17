import torch
from mmpretrain import get_model

model = get_model('mae_vit-base-p16_8xb512-amp-coslr-300e_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
