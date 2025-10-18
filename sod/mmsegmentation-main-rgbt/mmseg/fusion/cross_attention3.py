# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, Sequential
from mmcv.cnn import (Linear, build_activation_layer)
# from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn.bricks.drop import build_dropout


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_list = [128,64,32,16]):
        super(CrossAttentionFusion,self).__init__()
        self.norm_cfg = {'eps': 1e-6, 'type': 'LN'}
        #0625
        layers = []
        for dim in dim_list:
            layers.append(CrossTransformerEncoderLayer(
                embed_dims=dim*dim,
                num_heads=8,
                qkv_bias=True,
                norm_cfg=self.norm_cfg))
        self.cross_attention = Sequential(*layers)


    def forward(self, x_rgb_list, x_ir_list):
        outs = []
        for i in range(len(x_rgb_list)):
            b,c,h,w = x_rgb_list[i].shape
            x_rgb_i = x_rgb_list[i].view(b,c,h*w)
            x_ir_i = x_ir_list[i].view(b,c,h*w)
            x_fusion_i = self.cross_attention[i](x_rgb_i, x_ir_i)
            #2 768 128 128->
            x_fusion_i = x_fusion_i.view(b,c,h,w)
            outs.append(x_fusion_i)
        return tuple(outs)





def build_norm_layer(cfg: dict, num_features: int) -> nn.Module:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.

        num_features (int): Number of input channels.

    Returns:
        nn.Module: The created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = torch.nn.modules.normalization.LayerNorm#MODELS.get(layer_type) #2040625
    if norm_layer is None:
        raise KeyError(f'Cannot find {layer_type} in registry under scope ')

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        layer = norm_layer(num_channels=num_features, **cfg_)

    if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
        layer._specify_ddp_gpu_num(1)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer
# 2024.01.20 calayzhou
class CrossTransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 # feedforward_channels,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 ffn_type='origin',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(CrossTransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = CrossMultiheadAttention(  # MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate)
        # dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        # layer_scale_init_value=layer_scale_init_value)

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)
        #
        # if ffn_type == 'origin':
        #     self.ffn = FFN(
        #         embed_dims=embed_dims,
        #         feedforward_channels=feedforward_channels,
        #         num_fcs=num_fcs,
        #         ffn_drop=drop_rate,
        #         dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #         act_cfg=act_cfg,
        #         layer_scale_init_value=layer_scale_init_value)
        # elif ffn_type == 'swiglu_fused':
        #     self.ffn = SwiGLUFFNFused(
        #         embed_dims=embed_dims,
        #         feedforward_channels=feedforward_channels,
        #         layer_scale_init_value=layer_scale_init_value)
        # else:
        #     raise NotImplementedError

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    # def init_weights(self):
    #     super(CrossTransformerEncoderLayer, self).init_weights()
    #     for m in self.ffn.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x, x2):
        # x =  self.attn(self.ln1(x2),self.ln1(x),self.ln1(x))# q k v
        # x = self.ffn(self.ln2(x), identity=x)

        # x = x + self.attn(self.ln1(x), self.ln1(x2), self.ln1(x2))
        x = x + self.attn(x, x2, x2)
        # x = x + self.attn(self.ln1(x2),self.ln1(x),self.ln1(x))#q k v
        # x = self.ffn(self.ln2(x), identity=x)

        return x

class CrossMultiheadAttention(BaseModule):
    """Cross attention between queries and the union of keys and values.

    This module is different from ``MultiheadAttention``, for the attention
    is computed between queries and the union of keys and values.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        compressed_dims = 256
        self.q = nn.Linear(embed_dims, compressed_dims, bias=False)
        self.k = nn.Linear(embed_dims, compressed_dims, bias=False)
        #self.v = nn.Linear(embed_dims, embed_dims, bias=False)


        self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(embed_dims, embed_dims)
        #self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                x: torch.Tensor,
                k: torch.Tensor = None,
                v: torch.Tensor = None) -> None:
        """Forward function."""
        B, N, _ = x.shape

        N_k = k.shape[1]
        N_v = v.shape[1]

        q = F.linear(
            input=x, weight=self.q.weight, bias=None)  # (B, N_q, dim)
        k = F.linear(
            input=k, weight=self.k.weight, bias=None)  # (B, N_k, dim)
        #v = F.linear(input=v, weight=self.v.weight, bias=None)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape(B, N_k, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape(B, N_v, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        #x = self.proj(x)
        # x = self.proj_drop(x)

        return x

class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        layer_scale_init_value (float): Initial value of scale factor in
            LayerScale. Default: 1.0
    """
    #
    # @deprecated_api_warning(
    #     {
    #         'dropout': 'ffn_drop',
    #         'add_residual': 'add_identity'
    #     },
    #     cls_name='FFN')
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 layer_scale_init_value=0.):
        super().__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    build_activation_layer(act_cfg), nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    # @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)



class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        inplace (bool): Whether performs operation in-place.
            Default: `False`.
        data_format (str): The input data format, could be 'channels_last'
            or 'channels_first', representing (B, C, H, W) and
            (B, N, C) format data respectively. Default: 'channels_last'.
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 data_format: str = 'channels_last',
                 scale: float = 1e-5):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x) -> torch.Tensor:
        if self.data_format == 'channels_first':
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)






class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """  # noqa

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.,
        bias: bool = True,
        dropout_layer: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, hidden_dims)
        else:
            self.norm = nn.Identity()

        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(
                dim=embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self,
                x: torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.norm(hidden)
        out = self.w3(hidden)
        out = self.gamma2(out)
        out = self.dropout_layer(out)

        if self.out_dims != self.embed_dims or not self.add_identity:
            # due to the dimension inconsistence or user setting
            # not to apply residual operation
            return out

        if identity is None:
            identity = x
        return identity + out


class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU FFN layer with fusing.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """  # noqa

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.,
        bias: bool = True,
    ) -> None:
        out_dims = out_dims or embed_dims
        feedforward_channels = feedforward_channels or embed_dims
        feedforward_channels = (int(feedforward_channels * 2 / 3) + 7) // 8 * 8
        super().__init__(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            out_dims=out_dims,
            layer_scale_init_value=layer_scale_init_value,
            bias=bias,
        )
