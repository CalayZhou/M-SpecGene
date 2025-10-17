# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from ..backbones.vision_transformer import TransformerEncoderLayer
from ..backbones.vision_transformer import CrossTransformerEncoderLayer
from ..utils import build_2d_sincos_position_embedding


@MODELS.register_module()
class MAEPretrainRGBTDualDecoder(BaseModule):
    """Decoder for MAE Pre-training.

    Some of the code is borrowed from `https://github.com/facebookresearch/mae`. # noqa

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 1024.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmpretrain.models import MAEPretrainDecoder
        >>> import torch
        >>> self = MAEPretrainDecoder()
        >>> self.eval()
        >>> inputs = torch.rand(1, 50, 1024)
        >>> ids_restore = torch.arange(0, 196).unsqueeze(0)
        >>> level_outputs = self.forward(inputs, ids_restore)
        >>> print(tuple(level_outputs.shape))
        (1, 196, 768)
    """

    def __init__(self,
                 num_patches: int = 196,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: int = 4,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 predict_feature_dim: Optional[float] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_patches = num_patches

        # used to convert the dim of features from encoder to the dim
        # compatible with that of decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed2 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token2 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # create new position embedding, different from that in encoder
        # and is not learnable
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False)
        self.decoder_pos_embed2 = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])
        
        
        #2024.4.4
        self.decoder_blocks2 = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

        self.cross_attention = CrossTransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg)


        self.cross_attention2 = CrossTransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg)

        self.decoder_norm_name, decoder_norm = build_norm_layer(
            norm_cfg, decoder_embed_dim, postfix=1)
        self.add_module(self.decoder_norm_name, decoder_norm)
        self.decoder_norm_name2, decoder_norm2 = build_norm_layer(
            norm_cfg, decoder_embed_dim, postfix=1)
        self.add_module(self.decoder_norm_name2, decoder_norm2)
        # Used to map features to pixels
        if predict_feature_dim is None:
            predict_feature_dim = patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, predict_feature_dim, bias=True)

        self.decoder_pred2 = nn.Linear(
            decoder_embed_dim, predict_feature_dim, bias=True)
    def init_weights(self) -> None:
        """Initialize position embedding and mask token of MAE decoder."""
        super().init_weights()

        decoder_pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.decoder_pos_embed.shape[-1],
            cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

        torch.nn.init.normal_(self.mask_token, std=.02)

        decoder_pos_embed2 = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.decoder_pos_embed2.shape[-1],
            cls_token=True)
        self.decoder_pos_embed2.data.copy_(decoder_pos_embed2.float())

        torch.nn.init.normal_(self.mask_token2, std=.02)
    @property
    def decoder_norm(self):
        """The normalization layer of decoder."""
        return getattr(self, self.decoder_norm_name)
    @property
    def decoder_norm2(self):
        """The normalization layer of decoder."""
        return getattr(self, self.decoder_norm_name2)

    def forward(self, x: torch.Tensor,
                ids_restore: torch.Tensor,
                x2: torch.Tensor,
                ids_restore2: torch.Tensor) -> torch.Tensor:
        """The forward function.

        The process computes the visible patches' features vectors and the mask
        tokens to output feature vectors, which will be used for
        reconstruction.

        Args:
            x (torch.Tensor): hidden features, which is of shape
                    B x (L * mask_ratio) x C.
            ids_restore (torch.Tensor): ids to restore original image.

        Returns:
            torch.Tensor: The reconstructed feature vectors, which is of
            shape B x (num_patches) x C.
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # embed tokens
        x2 = self.decoder_embed2(x2)

        # append mask tokens to sequence
        mask_tokens2 = self.mask_token2.repeat(
            x2.shape[0], ids_restore2.shape[1] + 1 - x2.shape[1], 1)
        x2_ = torch.cat([x2[:, 1:, :], mask_tokens2], dim=1)
        x2_ = torch.gather(
            x2_,
            dim=1,
            index=ids_restore2.unsqueeze(-1).repeat(1, 1, x2.shape[2]))
        x2 = torch.cat([x2[:, :1, :], x2_], dim=1)

        # add pos embed
        x2 = x2 + self.decoder_pos_embed2

        new_x2 = self.cross_attention2(x2, x)
        new_x = self.cross_attention(x, x2)

        for blk_i in range(len(self.decoder_blocks)):
            new_x = self.decoder_blocks[blk_i](new_x)
            new_x2 = self.decoder_blocks2[blk_i](new_x2)

        x = self.decoder_norm(new_x)
        x2 = self.decoder_norm2(new_x2)

        # predictor projection
        x = self.decoder_pred(x)
        x2 = self.decoder_pred2(x2)

        # remove cls token
        x = x[:, 1:, :]
        x2 = x2[:, 1:, :]

        return x,x2


@MODELS.register_module()
class RGBTDUalClsBatchNormNeck(BaseModule):
    """Normalize cls token across batch before head.

    This module is proposed by MAE, when running linear probing.

    Args:
        input_features (int): The dimension of features.
        affine (bool): a boolean value that when set to ``True``, this module
            has learnable affine parameters. Defaults to False.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 input_features: int,
                 affine: bool = False,
                 eps: float = 1e-6,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg)
        self.bn = nn.BatchNorm1d(input_features, affine=affine, eps=eps)

    def forward(
            self,
            inputs: Tuple[List[torch.Tensor]]) -> Tuple[List[torch.Tensor]]:
        """The forward function."""
        # Only apply batch norm to cls_token
        inputs = [self.bn(input_) for input_ in inputs]
        return tuple(inputs)
