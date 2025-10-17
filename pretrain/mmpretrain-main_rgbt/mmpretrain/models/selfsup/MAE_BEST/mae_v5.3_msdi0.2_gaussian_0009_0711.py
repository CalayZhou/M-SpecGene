# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import random
from mmpretrain.models import HiViT, VisionTransformer
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import build_2d_sincos_position_embedding
from .base import BaseSelfSupervisor
import matplotlib.pyplot as plt

@MODELS.register_module()
class MAEViT(VisionTransformer):
    """Vision Transformer for MAE pre-training.

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    This module implements the patch masking in MAE and initialize the
    position embedding with sine-cosine position embedding.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            It only works without input mask. Defaults to ``"avg_featmap"``.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 out_indices: Union[Sequence, int] = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 out_type: str = 'raw',
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 mask_ratio: float = 0.75,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            out_type=out_type,
            with_cls_token=True,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        # position embedding is not learnable during pretraining
        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.count = 0
        self.epoch = 0
        self.noisebias = 0.0

        # self.CosineSimilarity = torch.nn.CosineSimilarity(dim=1,eps=1e-6)

    def CosineSimilarity(self, tensor_1,tensor_2 ):
        normalized_tensor_1 = tensor_1 /tensor_1.norm(dim=-1,keepdim=True)
        normalized_tensor_2 = tensor_2 /tensor_2.norm(dim=-1,keepdim=True)
        # print("norm",torch.min(normalized_tensor_1),torch.min(normalized_tensor_2))
        # print("norm mul",torch.min((normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)))
        # print("norm vector1", normalized_tensor_1[0, :])
        # print("norm vector2", normalized_tensor_2[0, :])
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
        # return (torch.abs(normalized_tensor_1) + torch.abs(normalized_tensor_2)).sum(dim=-1)

    def PearsonSimilarity(self, tensor_1,tensor_2 ):
        mean_tensor_1 = torch.mean(tensor_1, dim=1, keepdim=True)
        mean_tensor_2 = torch.mean(tensor_2, dim=1, keepdim=True)
        tensor_m1 = tensor_1 - mean_tensor_1
        tensor_m2 = tensor_2 - mean_tensor_2
        r_num = torch.sum(tensor_m1*tensor_m2,dim=1)
        r_den = torch.sqrt(torch.sum(tensor_m1**2,dim=1)*torch.sum(tensor_m2**2,dim=1))
        return r_num / r_den

    def PearsonVarSimilarity(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 /tensor_1.norm(dim=-1,keepdim=True)
        normalized_tensor_2 = tensor_2 /tensor_2.norm(dim=-1,keepdim=True)
        r_num = torch.sum(normalized_tensor_1 * normalized_tensor_2, dim=1)
        r_num = (r_num+1.0)*0.5
        var_tensor_1 = torch.var(tensor_1, dim=1)
        var_tensor_2 = torch.var(tensor_1, dim=1)
        # var_tensor_1 = var_tensor_1**2
        # var_tensor_2 = var_tensor_2**2
        r_num = torch.sqrt(r_num)
        measure = r_num / (var_tensor_1*var_tensor_2)
        # mean_tensor_1 = torch.mean(tensor_1, dim=1, keepdim=True)
        # mean_tensor_2 = torch.mean(tensor_2, dim=1, keepdim=True)
        # tensor_m1 = tensor_1 - mean_tensor_1
        # tensor_m2 = tensor_2 - mean_tensor_2
        # r_num = torch.sum(tensor_m1*tensor_m2,dim=1)
        # r_den = torch.sqrt(torch.sum(tensor_m1**2,dim=1)*torch.sum(tensor_m2**2,dim=1))
        #
        # var_tensor_1 = torch.var(tensor_1, dim=1)
        # var_tensor_2 = torch.var(tensor_2, dim=1)
        # # pearson_corr = r_num/r_den
        # corr = (r_num/r_den+1.0)
        # measure = (var_tensor_1 + var_tensor_2) /corr
        # print(torch.max(measure),torch.min(measure))
        return measure


    def cal_ids_shuffle(self,noise,cos_sim,cos_sim2,x,len_keep,first_modality=True):
        N, L, D = x.shape
        ids_shuffle = torch.zeros(N,L,device=x.device)
        for l in range(len_keep):
            abs_diff = torch.abs(noise[:, l].unsqueeze(1) - cos_sim[:, :])#B L
            nearest_index = torch.argsort(abs_diff)#[0]
            ids_shuffle[:,l] =nearest_index[:,0]
            cos_sim[range(N), nearest_index[:,0]] = 1e5 #nearest_index[:,0]-> 12
            if first_modality:
                cos_sim2[range(N), nearest_index[:,0]] = 1e3 #nearest_index[:,0]-> 12

        abs_diff = torch.abs(noise - cos_sim)
        nearest_index = torch.argsort(abs_diff)
        ids_shuffle[:, len_keep:] = nearest_index[:, :L-len_keep ]
        ids_shuffle = ids_shuffle.to(torch.int64)
        if first_modality:
            return ids_shuffle, cos_sim2
        else:
            return ids_shuffle


    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.pos_embed.shape[-1],
            cls_token=True)
        self.pos_embed.data.copy_(pos_embed.float())

        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)


    def random_masking(
        self,
        x: torch.Tensor,
        x2: torch.Tensor,
        # x_corr: torch.Tensor,
        # x2_corr: torch.Tensor,
        # ids_restore_pre: torch.Tensor,
        # ids_shuffle_pre: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the mask for MAE Pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))


        #cal the corr
        x_flat = x.clone().detach().contiguous().view(N*L,D)
        x2_flat = x2.clone().detach().contiguous().view(N*L,D)
        # cos_sim = F.cosine_similarity(x_flat,x2_flat) #N*L
        # cos_sim = self.CosineSimilarity(x_flat,x2_flat) #N*L
        # cos_sim = self.PearsonSimilarity(x_flat,x2_flat) #N*L
        # cos_sim = self.VarSimilarity(x_flat,x2_flat) #N*L
        cos_sim = self.PearsonVarSimilarity(x_flat,x2_flat) #N*L
        cos_sim  =cos_sim.view(N, L)# N L

        cos_sim2 = cos_sim.clone().detach()
        cos_sim3 = cos_sim.clone().detach()

        #VIS CORR HIST
        # cos_sim3_np = cos_sim3.cpu().numpy()
        # plt.hist(cos_sim3_np[0,:],bins=20)
        # plt.show()

        # noise = torch.rand(N, L, device=x.device)*0.1#-1#+0.9#0~1noise = {Tensor: (12, 196)} tensor([[0.2584, 0.8953, 0.4121,  ..., 0.9540, 0.3227, 0.5171],\n        [0.2006, 0.6923, 0.4589,  ..., 0.4050, 0.5022, 0.0339],\n        [0.2504, 0.4164, 0.3633,  ..., 0.8755, 0.5532, 0.2289],\n        ...,\n        [0.2844, 0.8122, 0.6207,  ..., 0.6554, 0.29... View
        # noise = noise * (cos_sim_max - cos_sim_min).unsqueeze(1) + cos_sim_min.unsqueeze(1)

        noise_ori = torch.randn(N, L, device=x.device)*0.2+self.noisebias #+bias#-1#+0.9#0~1noise = {Tensor: (12, 196)} tensor([[0.2584, 0.8953, 0.4121,  ..., 0.9540, 0.3227, 0.5171],\n        [0.2006, 0.6923, 0.4589,  ..., 0.4050, 0.5022, 0.0339],\n        [0.2504, 0.4164, 0.3633,  ..., 0.8755, 0.5532, 0.2289],\n        ...,\n        [0.2844, 0.8122, 0.6207,  ..., 0.6554, 0.29... View
        # noise = torch.randn(N, L, device=x.device)*0.1+0.5#+bias#-1#+0.9#0~1noise = {Tensor: (12, 196)} tensor([[0.2584, 0.8953, 0.4121,  ..., 0.9540, 0.3227, 0.5171],\n        [0.2006, 0.6923, 0.4589,  ..., 0.4050, 0.5022, 0.0339],\n        [0.2504, 0.4164, 0.3633,  ..., 0.8755, 0.5532, 0.2289],\n        ...,\n        [0.2844, 0.8122, 0.6207,  ..., 0.6554, 0.29... View
        noise_ori[noise_ori < 0.0] = 0.0 #0.0~1.0
        noise_ori[noise_ori > 1.0] = 1.0 #0.0~1.0
        cos_sim_max, cos_sim_min = torch.max(cos_sim, dim=1)[0],torch.min(cos_sim,dim=1)[0]
        noise = noise_ori * (cos_sim_max-cos_sim_min).unsqueeze(1)+cos_sim_min.unsqueeze(1)
        ids_shuffle,cos_sim2 = self.cal_ids_shuffle(noise, cos_sim, cos_sim2, x, len_keep,first_modality=True)

        ids_restore = torch.argsort(ids_shuffle, dim=1)# 1.reorder  2.after reorder -> id
        ids_keep = ids_shuffle[:, :len_keep]
        # print(cos_sim3[0, ids_keep[0,:]])
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        #x2 mask
        # noise2 = torch.rand(N, L, device=x.device)*0.1#-1#+0.9
        # noise2 = noise2 * (cos_sim_max-cos_sim_min).unsqueeze(1)+cos_sim_min.unsqueeze(1)
        noise2_ori = torch.randn(N, L, device=x.device)*0.2+self.noisebias #+bias#-1#+0.9#0~1noise = {Tensor: (12, 196)} tensor([[0.2584, 0.8953, 0.4121,  ..., 0.9540, 0.3227, 0.5171],\n        [0.2006, 0.6923, 0.4589,  ..., 0.4050, 0.5022, 0.0339],\n        [0.2504, 0.4164, 0.3633,  ..., 0.8755, 0.5532, 0.2289],\n        ...,\n        [0.2844, 0.8122, 0.6207,  ..., 0.6554, 0.29... View
        # noise2 = torch.randn(N, L, device=x.device)*0.1+0.5#+bias#-1#+0.9#0~1noise = {Tensor: (12, 196)} tensor([[0.2584, 0.8953, 0.4121,  ..., 0.9540, 0.3227, 0.5171],\n        [0.2006, 0.6923, 0.4589,  ..., 0.4050, 0.5022, 0.0339],\n        [0.2504, 0.4164, 0.3633,  ..., 0.8755, 0.5532, 0.2289],\n        ...,\n        [0.2844, 0.8122, 0.6207,  ..., 0.6554, 0.29... View
        noise2_ori[noise2_ori < 0.0] = 0.0 #0.0~1.0
        noise2_ori[noise2_ori > 1.0] = 1.0 #0.0~1.0
        noise2 = noise2_ori * (cos_sim_max-cos_sim_min).unsqueeze(1)+cos_sim_min.unsqueeze(1)
        ids_shuffle2  = self.cal_ids_shuffle(noise2, cos_sim2, cos_sim2, x, len_keep, first_modality=False)
        ids_restore2 = torch.argsort(ids_shuffle2, dim=1)  # 1.reorder  2.after reorder -> id
        ids_keep2 = ids_shuffle2[:, :len_keep]
        x_masked2 = torch.gather(
            x2, dim=1, index=ids_keep2.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask2 = torch.ones([N, L], device=x2.device)
        mask2[:, :len_keep] = 0
        mask2 = torch.gather(mask2, dim=1, index=ids_restore2)

        # #--------------- vis mask---------------
        # num = (len(os.listdir("./mask_vis/")) -2)// 6
        # mask_vis  = mask.view(-1,14, 14)
        # mask_vis_0_np = mask_vis[0, :, :].cpu().numpy()#.transpose((1, 2, 0))
        # cv2.imwrite('./mask_vis/'+str(num)+'mask2.png', mask_vis_0_np*255)
        # mask_vis2  = mask2.view(-1,14, 14)
        # mask_vis_0_np2 = mask_vis2[0, :, :].cpu().numpy()#.transpose((1, 2, 0))
        # cv2.imwrite('./mask_vis/'+str(num)+'mask.png', mask_vis_0_np2*255)
        #--------------- vis mask---------------

        self.count =  self.count +1
        if self.count>231:
            self.count = 0
            self.epoch = self.epoch+1
            self.noisebias = 0.005*self.epoch
            print("---------------self.epoch",self.epoch,self.noisebias,torch.max(noise_ori),torch.min(noise_ori),torch.max(noise2_ori),torch.min(noise2_ori),"---------------")
        return x_masked, mask, ids_restore, \
            x_masked2, mask2, ids_restore2

    def forward(
        self,
        x: torch.Tensor,
        x2: torch.Tensor,
        # ids_restore_pre: torch.Tensor,
        # ids_shuffle_pre: torch.Tensor,
        mask: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B = x.shape[0]
            x = self.patch_embed(x)[0]
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

            # 0322 x2
            x2 = self.patch_embed(x2)[0]
            # add pos embed w/o cls token
            x2 = x2 + self.pos_embed[:, 1:, :]

            # masking: length -> length * mask_ratio
            x, mask, ids_restore,\
                x2, mask2, ids_restore2,\
                = self.random_masking(x, x2, self.mask_ratio)



            #  append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for _, layer in enumerate(self.layers):
                x = layer(x)
            # Use final norm
            x = self.norm1(x)

            # --------------  x2 --------------
            # append cls token
            x2 = torch.cat((cls_tokens, x2), dim=1)

            for _, layer in enumerate(self.layers):
                x2 = layer(x2)
            # Use final norm
            x2 = self.norm1(x2)

            # return (x, mask, ids_restore, ids_shuffle)
            return x, mask, ids_restore, \
                   x2, mask2, ids_restore2


@MODELS.register_module()
class MAE(BaseSelfSupervisor):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """

    # def __init__(self,):d
    #     super(MAE, self).__init__(self,)
    #     self.count = 0
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 target_generator: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            target_generator=target_generator,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.count = 0
        self.epoch = 0

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        input_rgb  = inputs[:,0:3,:,:]
        input_ir  = inputs[:,3:6,:,:]
        # ####CEHCK####
        # img_rgb = inputs[0, 0:3, :, :].cpu().numpy().transpose((1, 2, 0))
        # img_ir = inputs[0, 3:6, :, :].cpu().numpy().transpose((1, 2, 0))
        # img_rgb[:,:,0] = img_rgb[:,:,0]*58.395+123.675
        # img_rgb[:,:,1] = img_rgb[:,:,1]*57.12+116.28
        # img_rgb[:,:,2] = img_rgb[:,:,2]*57.375+103.53
        # img_ir[:,:,0] = img_ir[:,:,0]*58.395+123.675
        # img_ir[:,:,1] = img_ir[:,:,1]*57.12+116.28
        # img_ir[:,:,2] = img_ir[:,:,2]*57.375+103.53
        # num = len(os.listdir("./mask_vis/"))//6
        # cv2.imwrite('./mask_vis/'+str(num)+'test_img_rgb.jpg',img_rgb)
        # cv2.imwrite('./mask_vis/'+str(num)+'test_img_ir.jpg',img_ir)
        # img_rgb = inputs[1, 0:3, :, :].cpu().numpy().transpose((1, 2, 0))
        # img_ir = inputs[1, 3:6, :, :].cpu().numpy().transpose((1, 2, 0))
        # img_rgb[:,:,0] = img_rgb[:,:,0]*58.395+123.675
        # img_rgb[:,:,1] = img_rgb[:,:,1]*57.12+116.28
        # img_rgb[:,:,2] = img_rgb[:,:,2]*57.375+103.53
        # img_ir[:,:,0] = img_ir[:,:,0]*58.395+123.675
        # img_ir[:,:,1] = img_ir[:,:,1]*57.12+116.28
        # img_ir[:,:,2] = img_ir[:,:,2]*57.375+103.53
        # cv2.imwrite('test_img_rgb1.jpg',img_rgb)
        # cv2.imwrite('test_img_ir1.jpg',img_ir)
        ####CEHCK####
        # if random.randint(0, 1) ==0:
        #     input_in = input_ir
        #     input_out = input_rgb
        # else:
        #     input_in = input_rgb
        #     input_out = input_ir
        # VERSION4 siam encoder siam decoder
        # latent, mask, ids_restore,ids_shuffle = self.backbone(input_rgb, None, None)#64 3 224 224
        # latent2, mask2, ids_restore2, ids_shuffle2 = self.backbone(input_ir,ids_restore,ids_shuffle)  # 64 3 224 224

        #VERSION5
        latent, mask, ids_restore, latent2, mask2, ids_restore2\
            = self.backbone(input_rgb, input_ir)#64 3 224 224

        #VIS MASK IMAGE
        # vis_mask,vis_mask_T = np.zeros((224,224,3)),np.zeros((224,224,3))
        # for i in range(mask.shape[1]):
        #     if mask[0,i]==0:
        #         x_th, y_th = i//14,i%14
        #         vis_mask[16*x_th:16*x_th+16, 16*y_th:16*y_th+16,:] =1
        #     if mask2[0,i]==0:
        #         x_th, y_th =  i//14,i%14
        #         vis_mask_T[16*x_th:16*x_th+16, 16*y_th:16*y_th+16,:] =1
        # num = len(os.listdir("./mask_vis/"))//6
        # cv2.imwrite('./mask_vis/'+str(num)+'mask_img_rgb.jpg',img_rgb*vis_mask)
        # cv2.imwrite('./mask_vis/'+str(num)+'mask_img_ir.jpg',img_ir*vis_mask_T)

        pred,pred2 = self.neck(latent, ids_restore, latent2, ids_restore2)
        loss = self.head.loss(pred, input_rgb, mask)#rgb
        loss2 = self.head.loss(pred2, input_ir, mask2)#ir
        losses = dict(loss=loss)
        losses.update({"loss2": loss2})

        # VERSION3 siam encoder single decoder
        # latent, mask, ids_restore,ids_shuffle = self.backbone(input_rgb, None, None)#64 3 224 224
        # latent2, mask2, ids_restore2, ids_shuffle2 = self.backbone(input_ir,ids_restore,ids_shuffle)  # 64 3 224 224
        # pred = self.neck(latent, ids_restore, latent2, ids_restore2)
        # loss = self.head.loss(pred, input_rgb, mask)
        # losses = dict(loss=loss)



        # VERSION1: single branch
        # input_in = input_rgb
        # input_out = input_rgb#input_ir
        # latent, mask, ids_restore = self.backbone(input_in)#64 3 224 224
        # pred = self.neck(latent, ids_restore)
        # loss = self.head.loss(pred, input_out, mask)
        # losses = dict(loss=loss)
        #
        # VERSION2: dual branch
        # latent, mask, ids_restore = self.backbone(input_rgb)#64 3 224 224
        # pred = self.neck(latent, ids_restore)
        # loss = self.head.loss(pred, input_rgb, mask)
        # losses = dict(loss=loss)

        # latent2, mask2, ids_restore2 = self.backbone(input_ir)#64 3 224 224
        # pred2 = self.neck(latent2, ids_restore2)
        # loss2 = self.head.loss(pred2, input_ir, mask2)
        # losses.update({"loss2":loss2})

        #VERSION0: ORI IMPLEMENT
        # latent, mask, ids_restore = self.backbone(inputs)#64 3 224 224
        # pred = self.neck(latent, ids_restore)
        # loss = self.head.loss(pred, inputs, mask)
        # losses = dict(loss=loss)

        return losses


@MODELS.register_module()
class MAEHiViT(HiViT):
    """HiViT for MAE pre-training.

    A PyTorch implement of: `HiViT: A Simple and More Efficient Design
    of Hierarchical Vision Transformer <https://arxiv.org/abs/2205.14949>`_.
    This module implements the patch masking in MAE and initialize the
    position embedding with sine-cosine position embedding.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
            Defaults to 4, to downsample 4x at the first stage
        inner_patches (int): The inner patches within a token
            Defaults to 4
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        ape (bool): the absolute position embedding
        rpe (bool): the relative position embedding
            Defaults to False
        layer_scale_init_value (float): the layer scale init value
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 inner_patches: int = 4,
                 out_indices: Union[list, int] = [23],
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 ape: bool = True,
                 rpe: bool = False,
                 layer_scale_init_value: float = 0.0,
                 mask_ratio: float = 0.75,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            ape=ape,
            rpe=rpe,
            layer_scale_init_value=layer_scale_init_value,
            init_cfg=init_cfg)

        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_embed.num_patches

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding."""
        super().apply(self._init_weights)
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.pos_embed.shape[-1],
            cls_token=False)
        self.pos_embed.data.copy_(pos_embed.float())

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def masking_id(
            self, batch_size,
            mask_ratio) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the mask for MAE Pre-training.

        Args:
            batch_size: The batch size of input data
            mask_ratio: The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the ids
            for the tokens retained, the ids to restore original image,
            and the mask
        """
        N, L = batch_size, self.pos_embed.size(1)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(
            N, L, device=self.pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B, C, H, W = x.shape
            ids_keep, ids_restore, mask = self.masking_id(B, self.mask_ratio)

            x = self.patch_embed(x)

            x = torch.gather(
                x,
                dim=1,
                index=ids_keep[:, :, None, None,
                               None].expand(-1, -1, *x.shape[2:]))

            for blk in self.blocks[:-self.num_main_blocks]:
                x = blk(x)

            x = x[..., 0, 0, :]
            if self.ape:
                pos_embed = self.interpolate_pos_encoding(x, H, W)
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1,
                                                      pos_embed.shape[2]),
                )
                x = x + pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks[-self.num_main_blocks:]:
                x = blk(x)

            return (x, mask, ids_restore)
