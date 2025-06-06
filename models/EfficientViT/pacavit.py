"""Patch-to-Cluster Attention (PaCa) Vision Trasnformer (ViT)
    https://arxiv.org/abs/2203.11987
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial
from torch.nn.modules.batchnorm import _BatchNorm

import os
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, LayerNorm2d

from mmcv.cnn import (
    build_norm_layer,
    build_activation_layer,
)

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from mmcv.cnn import ConvModule, build_norm_layer
import numpy as np


has_xformers = False

__all__ = ["PaCaViT"]


class BlurPoolConv2d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


def apply_blurpool(mod: nn.Module):
    for name, child in mod.named_children():
        if (
            isinstance(child, nn.Conv2d)
            and not isinstance(child, BlurPoolConv2d)
            and (np.max(child.stride) > 1 and child.in_channels >= 16)
        ):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


class DownsampleV1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        patch_size,
        kernel_size,
        norm_cfg=dict(type="LayerNorm2d", eps=1e-6),
        img_size=224,
    ):
        super().__init__()

        assert patch_size in (2, 4)
        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        if patch_size <= kernel_size:
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=patch_size,
                padding=(kernel_size - 1) // 2,
            )
        else:
            dim = out_channels // 2
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.Conv2d(
                    out_channels // 2,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=patch_size // 2,
                    padding=(kernel_size - 1) // 2,
                ),
            )

        self.norm = (
            build_norm_layer(norm_cfg, out_channels)[1] if norm_cfg else nn.Identity()
        )

    def forward(self, x):
        # x: B C H W
        x = self.proj(x)
        x = self.norm(x)
        return x


class DownsampleV2(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        img_size=224,
        kernel_size=3,
        patch_size=4,
        ratio=0.5,
        conv_cfg=None,
        conv_bias=True,
        norm_cfg=dict(type="LayerNorm2d"),
        act_cfg=dict(type="GELU"),
        with_blurpool=False,
        order=("conv", "norm", "act"),
        **kwargs
    ):
        super().__init__()
        assert patch_size in (2, 4)

        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        if patch_size == 4:
            mid_chs = int(out_chs * ratio)
            self.conv1 = ConvModule(
                in_chs,
                mid_chs,
                kernel_size=kernel_size,
                stride=2,
                padding=(kernel_size - 1) // 2,
                bias=conv_bias,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order,
            )
        else:
            mid_chs = in_chs
            self.conv1 = nn.Identity()

        self.conv2 = ConvModule(
            mid_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size - 1) // 2,
            bias=conv_bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            order=order,
        )
        if with_blurpool:
            apply_blurpool(self.conv1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


downsampler_cfg = {
    # layer_abbreviation: module
    "DownsampleV1": DownsampleV1,
    "DownsampleV2": DownsampleV2,
}


def build_downsample_layer(cfg):
    """Build downsample (stem or transition) layer

    Args:
        cfg (dict): cfg should contain:
            type (str): Identify activation layer type.
            layer args: args needed to instantiate a stem layer.

    Returns:
        layer (nn.Module): Created stem layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in downsampler_cfg:
        raise KeyError("Unrecognized stem type {}".format(layer_type))
    else:
        layer = downsampler_cfg[layer_type]
        if layer is None:
            raise NotImplementedError

    layer = layer(**cfg_)
    return layer

def c_rearrange(x, H, W, dim=1):
    channels_last = x.is_contiguous(memory_format=torch.channels_last)
    if dim == 1:
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
    elif dim == 2:
        x = rearrange(x, "B C (H W) -> B C H W", H=H, W=W)
    else:
        raise NotImplementedError

    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    else:
        x = x.contiguous()
    return x


class DWConv(nn.Module):
    def __init__(self, dim, kernel_size=3, bias=True, with_shortcut=False):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size, 1, (kernel_size - 1) // 2, bias=bias, groups=dim
        )
        self.with_shortcut = with_shortcut

    def forward(self, x, H, W):
        shortcut = x
        x = c_rearrange(x, H, W)
        x = self.dwconv(x)
        x = rearrange(x, "B C H W -> B (H W) C", H=H, W=W)
        if self.with_shortcut:
            return x + shortcut
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        with_dwconv=False,
        with_shortcut=False,
        act_cfg=dict(type="GELU"),
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = (
            DWConv(hidden_features, with_shortcut=with_shortcut)
            if with_dwconv
            else None
        )
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.dwconv is not None:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn, drop_path_rate=0.0):
        super().__init__()
        self.fn = fn

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        return self.drop_path(self.fn(x)) + x


class P2CLinear(nn.Module):
    def __init__(self, dim, num_clusters, **kwargs) -> None:
        super().__init__()

        self.clustering = nn.Sequential(
            nn.Linear(dim, num_clusters, bias=False),
            Rearrange("B N M -> B M N"),
        )

    def forward(self, x, H, W):
        return self.clustering(x)


class P2CMlp(nn.Module):
    def __init__(
        self, dim, num_clusters, mlp_ratio=4.0, act_cfg=dict(type="GELU"), **kwargs
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.clustering = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            build_activation_layer(act_cfg),
            nn.Linear(hidden_dim, num_clusters),  # TODO: train w/ bias=False
            Rearrange("B N M -> B M N"),
        )

    def forward(self, x, H, W):
        return self.clustering(x)


class P2CConv2d(nn.Module):
    def __init__(
        self,
        dim,
        num_clusters,
        kernel_size=7,
        act_cfg=dict(type="GELU"),
        **kwargs,
    ) -> None:
        super().__init__()

        self.clustering = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=dim,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_clusters, 1, 1, 0, bias=False),
            Rearrange("B M H W -> B M (H W)"),
        )

    def forward(self, x, H, W):
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()
        return self.clustering(x)


class PaCaLayer(nn.Module):
    """Patch-to-Cluster Attention Layer"""

    def __init__(
        self,
        paca_cfg,
        dim,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_ratio=4.0,
        act_cfg=dict(type="GELU"),
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads

        self.use_xformers = has_xformers and (dim // num_heads) % 32 == 0

        self.num_clusters = paca_cfg["clusters"]
        self.onsite_clustering = paca_cfg["onsite_clustering"]
        if self.num_clusters > 0:
            self.cluster_norm = (
                build_norm_layer(paca_cfg["cluster_norm_cfg"], dim)[1]
                if paca_cfg["cluster_norm_cfg"]
                else nn.Identity()
            )

            if self.onsite_clustering:
                self.clustering = eval(paca_cfg["type"])(
                    dim=dim,
                    num_clusters=self.num_clusters,
                    mlp_ratio=mlp_ratio,
                    kernel_size=paca_cfg["clustering_kernel_size"],
                    act_cfg=act_cfg,
                )

            self.cluster_pos_embed = paca_cfg["cluster_pos_embed"]
            if self.cluster_pos_embed:
                self.cluster_pos_enc = nn.Parameter(
                    torch.zeros(1, self.num_clusters, dim)
                )
                trunc_normal_(self.cluster_pos_enc, std=0.02)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity()  # get attn weights for viz.

    def forward(self, x, z):
        # x: B N C
        N = x.shape[1]
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        if self.num_clusters > 0:
            if self.onsite_clustering:
                z_raw = self.clustering(x, H, W)  # B M N
                z = z_raw.softmax(dim=-1)
                # TODO: how to auto-select the 'meaningful' subset of clusters
            # c = z @ x  # B M C
            c = einsum("bmn,bnc->bmc", z, x)
            if self.cluster_pos_embed:
                c = c + self.cluster_pos_enc.expand(c.shape[0], -1, -1)
            c = self.cluster_norm(c)
        else:
            c = x

        if self.use_xformers:
            q = self.q(x)  # B N C
            k = self.k(c)  # B M C
            v = self.v(c)
            q = rearrange(q, "B N (h d) -> B N h d", h=self.num_heads)
            k = rearrange(k, "B M (h d) -> B M h d", h=self.num_heads)
            v = rearrange(v, "B M (h d) -> B M h d", h=self.num_heads)

            #x = xops.memory_efficient_attention(q, k, v)  # B N h d
            x = rearrange(x, "B N h d -> B N (h d)")

            x = self.proj(x)
        else:
            x = rearrange(x, "B N C -> N B C")
            c = rearrange(c, "B M C -> M B C")

            x, attn = F.multi_head_attention_forward(
                query=x,
                key=c,
                value=c,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q.weight,
                k_proj_weight=self.k.weight,
                v_proj_weight=self.v.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q.bias, self.k.bias, self.v.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=self.attn_drop,
                out_proj_weight=self.proj.weight,
                out_proj_bias=self.proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=not self.training,  # for visualization
                #average_attn_weights=False,
            )

            x = rearrange(x, "N B C -> B N C")

            if not self.training:
                attn = self.attn_viz(attn)

        x = self.proj_drop(x)

        return x, z


class PaCaBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        paca_cfg,
        mlp_ratio=4.0,
        drop_path=0.0,
        attn_drop=0.0,
        drop=0.0,
        act_cfg=dict(type="GELU"),
        layer_scale=None,
        input_resolution=None,
        with_pos_embed=False,
        post_norm=False,
        sub_ln=False,  # https://arxiv.org/abs/2210.06423
        **kwargs,
    ):
        super().__init__()

        self.post_norm = post_norm

        self.with_pos_embed = with_pos_embed
        self.input_resolution = input_resolution
        if self.with_pos_embed:
            assert self.input_resolution is not None
            self.input_resolution = to_2tuple(self.input_resolution)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.input_resolution[0] * self.input_resolution[1], dim)
            )
            self.pos_drop = nn.Dropout(p=drop)
            trunc_normal_(self.pos_embed, std=0.02)

        self.norm1_before = (
            build_norm_layer(paca_cfg["norm_cfg1"], dim)[1]
            if sub_ln or not post_norm
            else nn.Identity()
        )
        self.attn = PaCaLayer(
            paca_cfg=paca_cfg,
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            mlp_ratio=mlp_ratio,
            act_cfg=act_cfg,
        )
        self.norm1_after = (
            build_norm_layer(paca_cfg["norm_cfg1"], dim)[1]
            if sub_ln or post_norm
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2_before = (
            build_norm_layer(paca_cfg["norm_cfg2"], dim)[1]
            if sub_ln or not post_norm
            else nn.Identity()
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = eval(paca_cfg["mlp_func"])(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            with_dwconv=paca_cfg["with_dwconv_in_mlp"],
            with_shortcut=paca_cfg["with_shortcut_in_mlp"],
            act_cfg=act_cfg,
        )
        self.norm2_after = (
            build_norm_layer(paca_cfg["norm_cfg2"], dim)[1]
            if sub_ln or post_norm
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x, H, W, z):
        # x: B N C
        if self.with_pos_embed:
            if self.input_resolution != (H, W):
                pos_embed = rearrange(self.pos_embed, "B (H W) C -> B C H W")
                pos_embed = F.interpolate(
                    pos_embed, size=(H, W), mode="bilinear", align_corners=True
                )
                pos_embed = rearrange(pos_embed, "B C H W -> B (H W) C")
            else:
                pos_embed = self.pos_embed

            x = self.pos_drop(x + pos_embed)

        a, z = self.attn(self.norm1_before(x), H, W, z)
        a = self.norm1_after(a)
        if not self.layer_scale:
            x = x + self.drop_path1(a)
            x = x + self.drop_path2(
                self.norm2_after(self.mlp(self.norm2_before(x), H, W))
            )
        else:
            x = x + self.drop_path1(self.gamma1 * a)
            x = x + self.drop_path2(
                self.gamma2 * self.norm2_after(self.mlp(self.norm2_before(x), H, W))
            )

        return x, z


class PaCaViT(nn.Module):
    """Patch-to-Cluster Attention (PaCa) ViT
    https://arxiv.org/abs/2203.11987
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        img_size=224,  # for cls only
        stem_cfg=dict(
            type="DownsampleV1",
            patch_size=4,
            kernel_size=3,
            norm_cfg=dict(type="LayerNorm2d", eps=1e-6),
        ),
        trans_cfg=dict(
            type="DownsampleV1",
            patch_size=2,
            kernel_size=3,
            norm_cfg=dict(type="LayerNorm2d", eps=1e-6),
        ),
        arch_cfg=dict(
            embed_dims=[96, 192, 320, 384],
            num_heads=[2, 4, 8, 16],
            mlp_ratios=[4, 4, 4, 4],
            depths=[2, 2, 4, 2],
        ),
        paca_cfg=dict(
            # default: onsite stage-wise conv-based clustering
            type="P2CConv2d",
            clusters=[100, 100, 100, 0],  # per stage
            # 0: the first block in a stage, 1: true for all blocks in a stage, i > 1: every i blocks
            onsite_clustering=[0, 0, 0, 0],
            clustering_kernel_size=[7, 7, 7, 7],
            cluster_norm_cfg=dict(type="LN", eps=1e-6),  # for learned clusters
            cluster_pos_embed=False,
            norm_cfg1=dict(type="LN", eps=1e-6),
            norm_cfg2=dict(type="LN", eps=1e-6),
            mlp_func="Mlp",
            with_dwconv_in_mlp=True,
            with_shortcut_in_mlp=True,
        ),
        paca_teacher_cfg=None,  # or
        # paca_teacher_cfg=dict(
        #     type="PaCaTeacher_ConvMixer",
        #     stem_cfg=None,
        #     embed_dims=[96, 192, 320, 384],
        #     depths=[2, 2, 2, 2],
        #     kernel_size=7,
        #     norm_cfg=dict(type="BN"),
        #     act_cfg=dict(type="GELU"),
        #     drop_path_rate=0.0,
        #     return_outs=True,
        # ),
        drop_path_rate=0.0,
        attn_drop=0.0,
        drop=0.0,
        norm_cfg=dict(type="LN", eps=1e-6),
        act_cfg=dict(type="GELU"),
        layer_scale=None,
        post_norm=False,
        sub_ln=False,
        with_pos_embed=False,
        out_indices=[],
        downstream_cluster_num=None,
        pretrained=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = arch_cfg["depths"]
        self.num_stages = len(self.depths)
        self.out_indices = out_indices

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]  # stochastic depth decay rule
        cur = 0

        paca_cfg_copy = paca_cfg.copy()
        if downstream_cluster_num is not None:
            assert len(downstream_cluster_num) == len(paca_cfg_copy["clusters"])
            paca_cfg_copy["clusters"] = downstream_cluster_num
        clusters = paca_cfg_copy["clusters"]
        onsite_clustering = paca_cfg_copy["onsite_clustering"]
        clustering_kernel_sizes = paca_cfg_copy["clustering_kernel_size"]

        embed_dims = arch_cfg["embed_dims"]
        num_heads = arch_cfg["num_heads"]
        mlp_ratios = arch_cfg["mlp_ratios"]

        # if paca teacher will be used
        self.paca_teacher = None
        if paca_teacher_cfg is not None:
            teacher = paca_teacher_cfg.pop("type")

            return_outs = (
                paca_teacher_cfg["return_outs"]
                and paca_teacher_cfg["embed_dims"] == embed_dims
            )
            paca_teacher_cfg.update(
                dict(
                    in_chans=in_chans,
                    clusters=clusters,
                    return_outs=return_outs,
                )
            )
            self.paca_teacher = eval(teacher)(**paca_teacher_cfg)
            self.share_stem = paca_teacher_cfg["stem_cfg"] is None

        # stem
        stem_cfg_ = stem_cfg.copy()
        stem_cfg_.update(
            dict(
                in_channels=in_chans,
                out_channels=embed_dims[0],
                img_size=img_size,
            )
        )
        self.patch_embed = build_downsample_layer(stem_cfg_)
        self.patch_grid = self.patch_embed.grid_size

        # stages
        for i in range(self.num_stages):
            paca_cfg_ = paca_cfg_copy.copy()
            paca_cfg_["clusters"] = clusters[i]
            paca_cfg_["clustering_kernel_size"] = clustering_kernel_sizes[i]

            blocks = nn.ModuleList()
            for j in range(self.depths[i]):
                paca_cfg_cur = paca_cfg_.copy()
                if self.paca_teacher is not None and clusters[i] > 0:
                    paca_cfg_cur["onsite_clustering"] = False
                else:
                    if j == 0:
                        paca_cfg_cur["onsite_clustering"] = True
                    else:
                        if onsite_clustering[i] < 2:
                            paca_cfg_cur["onsite_clustering"] = onsite_clustering[i]
                        else:
                            paca_cfg_cur["onsite_clustering"] = (
                                True if j % onsite_clustering[i] == 0 else False
                            )

                blocks.append(
                    PaCaBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        paca_cfg=paca_cfg_cur,
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        attn_drop=attn_drop,
                        drop=drop,
                        act_cfg=act_cfg,
                        layer_scale=layer_scale,
                        input_resolution=(
                            self.patch_grid[0] // (2**i),
                            self.patch_grid[1] // (2**i),
                        ),
                        with_pos_embed=with_pos_embed if j == 0 else False,
                        post_norm=post_norm,
                        sub_ln=sub_ln,
                    )
                )
            cur += self.depths[i]

            setattr(self, f"stage{i + 1}", blocks)

            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            setattr(self, f"norm{i + 1}", norm)

            if i < self.num_stages - 1:
                cfg_ = trans_cfg.copy()
                cfg_.update(
                    dict(
                        in_channels=embed_dims[i],
                        out_channels=embed_dims[i + 1],
                        img_size=(
                            self.patch_grid[0] // (2**i),
                            self.patch_grid[1] // (2**i),
                        ),
                    )
                )
                transition = build_downsample_layer(cfg_)
                setattr(self, f"transition{i + 1}", transition)

        # classification head
        self.head = None
        if num_classes > 0:
            self.head = nn.Linear(embed_dims[-1], num_classes)

        self.init_weights()
        self.load_pretrained_chkpt(pretrained)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_features(self, x):
        # x: B C H W
        x_ = x

        x = self.patch_embed(x)

        # paca teacher
        cluster_assignment = None
        teacher_outs = None
        if self.paca_teacher is not None:
            if self.share_stem:
                x_ = x
            cluster_assignment, teacher_outs = self.paca_teacher(x_)

        outs = []
        HWs = []
        for i in range(self.num_stages):
            H, W = x.shape[2:]
            x = rearrange(x, "B C H W -> B (H W) C").contiguous()
            blocks = getattr(self, f"stage{i + 1}")
            z = None
            if cluster_assignment is not None and i < len(cluster_assignment):
                z = cluster_assignment[i]
            for block in blocks:
                x, z = block(x, H, W, z)

            if teacher_outs is not None and i < len(teacher_outs):
                x = x + teacher_outs[i]

            norm = getattr(self, f"norm{i+1}")
            x = norm(x)

            if self.head is None and i in self.out_indices:
                outs.append(x)
                HWs.append((H, W))

            if i != self.num_stages - 1:
                x = c_rearrange(x, H, W)
                transition = getattr(self, f"transition{i + 1}")
                x = transition(x)

        if self.head is None:
            outs_ = []
            for out, HW in zip(outs, HWs):
                out = c_rearrange(out, HW[0], HW[1])
                outs_.append(out)
            return outs_

        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.head is not None:
            x = x.mean(dim=1)
            x = self.head(x)
            return x

        return x

    def forward_dummy(self, x):
        x = self.forward_features(x)

        if self.head is not None:
            x = x.mean(dim=1)
            x = self.head(x)
            return x

        return x



# ------------------------- build transformer block ----------------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm1(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, z, **kwargs):
        return self.fn(self.norm(x), z, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Transformer_paca(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        paca_cfg=dict(
            # default: onsite stage-wise conv-based clustering
            type="P2CConv2d",
            clusters=[100, 100, 100, 0],  # per stage
            # 0: the first block in a stage, 1: true for all blocks in a stage, i > 1: every i blocks
            onsite_clustering=[0, 0, 0, 0],
            clustering_kernel_size=[7, 7, 7, 7],
            cluster_norm_cfg=dict(type="LN", eps=1e-6),  # for learned clusters
            cluster_pos_embed=False,
            norm_cfg1=dict(type="LN", eps=1e-6),
            norm_cfg2=dict(type="LN", eps=1e-6),
            mlp_func="Mlp",
            with_dwconv_in_mlp=True,
            with_shortcut_in_mlp=True,
        )
        paca_cfg_copy = paca_cfg.copy()
        clusters = paca_cfg_copy["clusters"]
        onsite_clustering = paca_cfg_copy["onsite_clustering"]
        clustering_kernel_sizes = paca_cfg_copy["clustering_kernel_size"]
        for j in range(depth):
            i = j//3
            paca_cfg_ = paca_cfg_copy.copy()
            paca_cfg_["clusters"] = clusters[i]
            paca_cfg_["clustering_kernel_size"] = clustering_kernel_sizes[i]
            paca_cfg_cur = paca_cfg_.copy()
            if j == 0:
                paca_cfg_cur["onsite_clustering"] = True
            else:
                if onsite_clustering[i] < 2:
                    paca_cfg_cur["onsite_clustering"] = onsite_clustering[i]
                else:
                    paca_cfg_cur["onsite_clustering"] = (True if j % onsite_clustering[i] == 0 else False)
            self.layers.append(nn.ModuleList([
                PreNorm1(dim, PaCaLayer(paca_cfg=paca_cfg_cur, dim=dim, num_heads=heads)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        z = None
        for attn, ff in self.layers:
            ax, z = attn(x, z)
            x = ax + x
            x = ff(x) + x
        return x