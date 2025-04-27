import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision.models import resnet50
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from .model_utils import *
from .CTM_utils import *

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1+ch_2+ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        
    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g*W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

#----------------------------------------------------------------------------------------
from .components.anti_over_smoothing import Transformer_Vanilla, Transformer_Refiner, Transformer_Layerscale, Transformer_Reattention
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TransformerDown(nn.Module):
    def __init__(self, in_channels, out_channels=384, image_size=256, depth=8, dmodel=384, mlp_dim=384*4, patch_size=16, heads=6, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_Vanilla(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer layer
        ax = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out
    
    def infere(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        xin = self.dropout(x)
        # encoder
        ax, ftokens, attmaps = self.transformer.infere(xin)  # b h*w ppc
        ftokens.insert(0, xin)
        out = self.recover_patch_embedding(ax)
        return out, ftokens, attmaps


class TransformerDown_AS(nn.Module):
    def __init__(
            self, in_channels=1, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[384, 768], num_heads=[4, 8], mlp_ratios=[4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2], as_depth=[1, 1],
            sr_ratios=[8, 4], num_stages=4, pretrained=None,
            k=5, sample_ratios=0.125, classes=4,
            # k=5, sample_ratios=0.0625,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.as_depth = as_depth
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)

        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.recover = MTA_TransFuse_light_smallModule()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([Block_AS(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(as_depth[0])])
        self.norm0_as = norm_layer(embed_dims[0])
        self.ctm_as = CTM_as(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1_as = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(as_depth[1])])
        self.norm1_as = norm_layer(embed_dims[1])

        self.stage0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])
        self.ctm1 = CTM_as(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        H = self.patch_num
        W = self.patch_num

        for blk in self.stage0_as:
            x, attn = blk(x, H, W)
        x = self.norm0_as(x)
        self.cur += self.as_depth[0]

        # attn :torch.Size([4, 4, 1024, 1024])    attn[:,0:2] :torch.Size([4, 2, 1024, 1024])
        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)  # attn_map :torch.Size([4, 1024])

        # 归一化方法1：sigmoid-0.5 (0-0.5)
        # as_out = torch.sigmoid(attn_map) - 0.5
        # 归一化方法2：(sigmoid-0.5)*2 (0-1)
        # as_out = (torch.sigmoid(attn_map) - 0.5) * 2
        # 归一化方法3：均匀归一化 (0-1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        as_out = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)

        as_out = rearrange(as_out, 'b (h w)-> b h w', h=H, w=W)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm_as(token_dict, as_out, ctm_stage=1)

        for j, blk in enumerate(self.stage1_as):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1_as(token_dict['x'])
        self.cur += self.as_depth[1]
        outs.append(token_dict)

        x = self.recover.forward(outs)

        for i in range(int((self.total_depth - sum(self.as_depth)) / sum(self.depths))):
            outs = []
            H = self.patch_num
            W = self.patch_num

            for blk in self.stage0:
                x = blk(x, H, W)
            x = self.norm0(x)
            self.cur += self.depths[0]

            B, N, _ = x.shape
            idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
            agg_weight = x.new_ones(B, N, 1)
            token_dict = {'x': x,
                          'token_num': N,
                          'map_size': [H, W],
                          'init_grid_size': [H, W],
                          'idx_token': idx_token,
                          'agg_weight': agg_weight}
            outs.append(token_dict.copy())

            # encoder:stage1
            token_dict = self.ctm1(token_dict, as_out, ctm_stage=i + 2)
            # token_dict = self.stage1(token_dict)
            for j, blk in enumerate(self.stage1):
                token_dict = blk(token_dict)
            token_dict['x'] = self.norm1(token_dict['x'])
            self.cur += self.depths[1]
            outs.append(token_dict)

            x = self.recover.forward(outs)

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, H, W)



        return x, as_out


class TransformerDown_AS0(nn.Module):
    def __init__(
            self, in_channels=1, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[96, 192], num_heads=[4, 8], mlp_ratios=[4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2], as_depth=[1, 1],
            sr_ratios=[8, 4], num_stages=4, pretrained=None,
            k=5, sample_ratios=0.125, classes=4,
            # k=5, sample_ratios=0.0625,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.as_depth = as_depth
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes

        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], 192),
            Rearrange('b s c -> b c s')
        )

        self.recover = MTA_TransFuse_light_smallModule(in_channels=[96, 192], out_channels=96)

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([Block_AS(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(as_depth[0])])
        self.norm0_as = norm_layer(embed_dims[0])
        self.ctm_as = CTM_as(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1_as = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(as_depth[1])])
        self.norm1_as = norm_layer(embed_dims[1])

        self.stage0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])
        self.ctm1 = CTM_as(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        H = 16
        W = 16

        for blk in self.stage0_as:
            x, attn = blk(x, H, W)
        x = self.norm0_as(x)
        self.cur += self.as_depth[0]

        # attn :torch.Size([4, 4, 1024, 1024])    attn[:,0:2] :torch.Size([4, 2, 1024, 1024])
        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)  # attn_map :torch.Size([4, 1024])

        # 归一化方法1：sigmoid-0.5 (0-0.5)
        # as_out = torch.sigmoid(attn_map) - 0.5
        # 归一化方法2：(sigmoid-0.5)*2 (0-1)
        # as_out = (torch.sigmoid(attn_map) - 0.5) * 2
        # 归一化方法3：均匀归一化 (0-1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        as_out = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)

        as_out = rearrange(as_out, 'b (h w)-> b h w', h=16, w=16)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm_as(token_dict, as_out, ctm_stage=1)

        for j, blk in enumerate(self.stage1_as):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1_as(token_dict['x'])
        self.cur += self.as_depth[1]
        outs.append(token_dict)

        x = self.recover.forward(outs)

        for i in range(int((self.total_depth - sum(self.as_depth)) / sum(self.depths))):
            outs = []
            H = 16
            W = 16

            for blk in self.stage0:
                x = blk(x, H, W)
            x = self.norm0(x)
            self.cur += self.depths[0]

            B, N, _ = x.shape
            idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
            agg_weight = x.new_ones(B, N, 1)
            token_dict = {'x': x,
                          'token_num': N,
                          'map_size': [H, W],
                          'init_grid_size': [H, W],
                          'idx_token': idx_token,
                          'agg_weight': agg_weight}
            outs.append(token_dict.copy())

            # encoder:stage1
            token_dict = self.ctm1(token_dict, as_out, ctm_stage=i + 2)
            # token_dict = self.stage1(token_dict)
            for j, blk in enumerate(self.stage1):
                token_dict = blk(token_dict)
            token_dict['x'] = self.norm1(token_dict['x'])
            self.cur += self.depths[1]
            outs.append(token_dict)

            x = self.recover.forward(outs)

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, 16, 16)



        return x, as_out


class Transformer_sparse1(nn.Module):
    def __init__(
            self, in_channels, out_channels, image_size, depth=12, dmodel=1024, mlp_dim=2048,
            patch_size=16, heads=12, dim_head=64, emb_dropout=0.1,

            embed_dims=[24, 48, 96, 192],  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, depths=[3, 3, 3, 3], sr_ratios=[8, 4, 2, 1], num_stages=4,
            k=5, sample_ratios=[0.25, 0.25, 0.25],
            return_map=False):
        super().__init__()

        # 第一部分
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width

        self.dmodel = embed_dims[0]

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
        )


        # 第二部分
        self.depths = depths
        self.embed_dims = embed_dims
        self.sample_ratios = sample_ratios
        self.k = k
        # self.recover = MTA_FAT1()
        self.recover = MTA_FAT1_light()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0
        self.stage0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])

        self.ctm1 = CTM(self.sample_ratios[0], self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = CTM(self.sample_ratios[1], self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([TCBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ctm3 = CTM(self.sample_ratios[2], self.embed_dims[2], self.embed_dims[3], self.k)
        self.stage3 = nn.ModuleList([TCBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for j in range(depths[3])])
        self.norm3 = norm_layer(embed_dims[3])



    def forward(self, img):
        # torch.Size([4, 3, 256, 256])
        x = self.to_patch_embedding(img)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        # torch.Size([4, 256, 24])
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)


        outs = []
        H = int(math.sqrt(n))
        W = H

        # encoder:stage0
        # x = self.stage1(x)
        for blk in self.stage0:
            x = blk(x, H, W)
        x = self.norm0(x)
        self.cur += self.depths[0]

        # init token dict
        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        # encoder:stage1
        token_dict = self.ctm1(token_dict)
        # token_dict = self.stage1(token_dict)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        # encoder:stage2
        token_dict = self.ctm2(token_dict)
        # token_dict = self.stage2(token_dict)
        for j, blk in enumerate(self.stage2):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        # encoder:stage3
        token_dict = self.ctm3(token_dict)
        # token_dict = self.stage3(token_dict)
        for j, blk in enumerate(self.stage3):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[3]
        outs.append(token_dict)

        # MTA and recover
        x = self.recover.forward(outs)  # torch.Size([4, 1024, 128])


        # torch.Size([4, 256, 192])
        x = self.recover_patch_embedding(x)
        # torch.Size([4, 192, 16, 16])
        return x

class Transformer_TransFuse_sparse1(nn.Module):
    def __init__(
            self, in_channels, out_channels, image_size, depth=12, dmodel=1024, mlp_dim=2048,
            patch_size=16, heads=12, dim_head=64, emb_dropout=0.1,

            embed_dims=[48, 96, 192, 768],  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, depths=[3, 3, 3, 3], sr_ratios=[8, 4, 2, 1], num_stages=4,
            k=5, sample_ratios=[0.5, 0.5, 0.5],
            return_map=False):
        super().__init__()

        # 第一部分
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width

        self.dmodel = embed_dims[0]

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
        )


        # 第二部分
        self.depths = depths
        self.embed_dims = embed_dims
        self.sample_ratios = sample_ratios
        self.k = k
        # self.recover = MTA_TransFuse()
        self.recover = MTA_TransFuse_light()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0
        self.stage0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])

        self.ctm1 = CTM(self.sample_ratios[0], self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = CTM(self.sample_ratios[1], self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([TCBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ctm3 = CTM(self.sample_ratios[2], self.embed_dims[2], self.embed_dims[3], self.k)
        self.stage3 = nn.ModuleList([TCBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for j in range(depths[3])])
        self.norm3 = norm_layer(embed_dims[3])



    def forward(self, img):
        # torch.Size([4, 3, 256, 256])
        x = self.to_patch_embedding(img)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        # torch.Size([4, 256, 24])
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)


        outs = []
        H = int(math.sqrt(n))
        W = H

        # encoder:stage0
        # x = self.stage1(x)
        for blk in self.stage0:
            x = blk(x, H, W)
        x = self.norm0(x)
        self.cur += self.depths[0]

        # init token dict
        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        # encoder:stage1
        token_dict = self.ctm1(token_dict)
        # token_dict = self.stage1(token_dict)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        # encoder:stage2
        token_dict = self.ctm2(token_dict)
        # token_dict = self.stage2(token_dict)
        for j, blk in enumerate(self.stage2):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        # encoder:stage3
        token_dict = self.ctm3(token_dict)
        # token_dict = self.stage3(token_dict)
        for j, blk in enumerate(self.stage3):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[3]
        outs.append(token_dict)

        # MTA and recover
        x = self.recover.forward(outs)  # torch.Size([4, 1024, 128])


        # torch.Size([4, 256, 192])
        x = self.recover_patch_embedding(x)
        # torch.Size([4, 192, 16, 16])
        return x

class TransformerDown_DTMFormerV2(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)


        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)



        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict = blk(token_dict)
        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)



        return x, attnScore

class TransformerDown_DTMFormerV2_attnloss(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)


        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)



        return x, attnScore, attns

class TransformerDown_DTMFormerV2_attnlossDS(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[4, 6, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)


        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 2, out_channels // 4, kernel_size=4, stride=2, padding=1),
            # 32x32 -> 64x64
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 4, out_channels // 8, kernel_size=4, stride=2, padding=1),
            # 64x64 -> 128x128
            nn.BatchNorm2d(out_channels // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 8, classes, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256
        )


    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)

        x_ds = self.decoder(x)

        return x, attnScore, attns, x_ds

class TransformerDown_DTMFormerV2FirstStageChoose (nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[4, 4, 4], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size / patch_size)

        self.out_channels = out_channels


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.atm1 = HTM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.atm2 = HTM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])

    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        attnScores = []
        variances = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore0 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = rearrange(attnScore0, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict, token_weight1, idx_cluster1, variance, score1 = self.atm1(token_dict, outs, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)
        variances.append(variance)

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore1 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = var_downup(attnScore1, outs[0], outs[1])
        attnScore = rearrange(attnScore, 'b (h w) 1-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        token_dict, token_weight2, idx_cluster2, variance, score2 = self.atm2(token_dict, outs, attnScore1, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)
        variances.append(variance)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)

        return x, attnScores, attns, outs, variances, [score1, score2]


class TransformerDown_DTMFormerV2FirstStageChoose2 (nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[256, 512, 768], num_heads=[4, 4, 4], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size / patch_size)

        self.out_channels = out_channels


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.atm1 = HTM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.atm2 = HTM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])

    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        attnScores = []
        variances = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore0 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = rearrange(attnScore0, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict, token_weight1, idx_cluster1, variance, score1 = self.atm1(token_dict, outs, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)
        variances.append(variance)

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore1 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = var_downup(attnScore1, outs[0], outs[1])
        attnScore = rearrange(attnScore, 'b (h w) 1-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        token_dict, token_weight2, idx_cluster2, variance, score2 = self.atm2(token_dict, outs, attnScore1, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)
        variances.append(variance)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)

        return x, attnScores, attns, outs, variances, [score1, score2]


class TransformerDown_DTMFormerV2FirstStageChoose3 (nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[384, 576, 768], num_heads=[4, 4, 4], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size / patch_size)

        self.out_channels = out_channels


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.atm1 = HTM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.atm2 = HTM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])

    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        attnScores = []
        variances = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore0 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = rearrange(attnScore0, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict, token_weight1, idx_cluster1, variance, score1 = self.atm1(token_dict, outs, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)
        variances.append(variance)

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore1 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = var_downup(attnScore1, outs[0], outs[1])
        attnScore = rearrange(attnScore, 'b (h w) 1-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        token_dict, token_weight2, idx_cluster2, variance, score2 = self.atm2(token_dict, outs, attnScore1, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)
        variances.append(variance)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)

        return x, attnScores, attns, outs, variances, [score1, score2]


class TransFuse(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse, self).__init__()

        self.resnet = resnet50()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown(in_channels=in_channels, image_size=img_size, out_channels=768, depth=12, heads=12)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
            )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
            )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate/2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate/2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b = self.transformer(imgs) # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2) # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u) # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)   # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1) # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16


        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1) # joint predict low supervise here


        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x#, map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)
    
    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs) # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2) # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u) # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)   # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1) # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16


        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1) # joint predict low supervise here


        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps

class TransFuse_AS(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_AS, self).__init__()

        self.resnet = resnet50()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown_AS(in_channels=in_channels, image_size=img_size, out_channels=768, total_depth=12)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, as_out = self.transformer(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, as_out  # , map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps

class TransFuse_AS0(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_AS0, self).__init__()

        self.resnet = resnet50()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown_AS0(in_channels=in_channels, image_size=img_size, out_channels=192, total_depth=12)

        self.up1 = Up(in_ch1=192, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        # bottom-up path
        x_b, as_out = self.transformer(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64


        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, as_out  # , map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps

class TransFuse_sparse1(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_sparse1, self).__init__()

        self.resnet = resnet50()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.transformer = TransformerDown(in_channels=in_channels, image_size=img_size, out_channels=768, depth=12,
        #                                    heads=12)
        self.transformer = Transformer_TransFuse_sparse1(in_channels=in_channels, out_channels=768, image_size=img_size)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        # torch.Size([4, 1, 256, 256])
        x_b = self.transformer(imgs)  # b 768 16 16

        # torch.Size([4, 768, 16, 16])
        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x  # , map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps

class TransFuse_DTMFormerV2(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_DTMFormerV2, self).__init__()

        self.resnet = resnet50()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown_DTMFormerV2(in_channels=in_channels, image_size=img_size, out_channels=768, total_depth=12)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, as_out = self.transformer(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, as_out  # , map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps

class TransFuse_DTMFormerV2_attnloss(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_DTMFormerV2_attnloss, self).__init__()

        self.resnet = resnet50()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown_DTMFormerV2_attnloss(in_channels=in_channels, image_size=img_size, out_channels=768, total_depth=12)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, as_out, attns = self.transformer(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, as_out, attns  # , map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps

class TransFuse_DTMFormerV2_attnlossDS(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_DTMFormerV2_attnlossDS, self).__init__()

        self.resnet = resnet50()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown_DTMFormerV2_attnlossDS(in_channels=in_channels, image_size=img_size, out_channels=768, total_depth=12)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, as_out, attns, feature_DS = self.transformer(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, as_out, attns, feature_DS

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps


class TransFuse_DTMFormerV2FirstStageChoose(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_DTMFormerV2FirstStageChoose, self).__init__()

        self.resnet = resnet50()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown_DTMFormerV2FirstStageChoose(in_channels=in_channels, image_size=img_size, out_channels=768, total_depth=12)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, attnScores, attns, outs, variances, scores = self.transformer(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, attnScores, attns, outs, variances, scores


    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps

class TransFuse_DTMFormerV2FirstStageChoose2(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_DTMFormerV2FirstStageChoose2, self).__init__()

        self.resnet = resnet50()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown_DTMFormerV2FirstStageChoose2(in_channels=in_channels, image_size=img_size, out_channels=768, total_depth=12)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, attnScores, attns, outs, variances, scores = self.transformer(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, attnScores, attns, outs, variances, scores


    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps

class TransFuse_DTMFormerV2FirstStageChoose3(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, img_size=256, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_DTMFormerV2FirstStageChoose3, self).__init__()

        self.resnet = resnet50()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = TransformerDown_DTMFormerV2FirstStageChoose3(in_channels=in_channels, image_size=img_size, out_channels=768, total_depth=12)

        self.up1 = Up(in_ch1=768, out_ch=512)
        self.up2 = Up(512, 256)

        self.final_x = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(256, 256, 3, bn=True, relu=True),
            Conv(256, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=768, r_2=4, ch_int=1024, ch_out=1024, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=512, r_2=2, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=1024, out_ch=512, in_ch2=512, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=256, r_2=1, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(512, 256, 256, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, attnScores, attns, outs, variances, scores = self.transformer(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, attnScores, attns, outs, variances, scores


    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def infere(self, imgs, labels=None):
        # bottom-up path
        x_b, ftokens, attmaps = self.transformer.infere(imgs)  # b 768 16 16

        x_b_1 = self.up1(x_b)  # b 512 32 32
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # b 256 64 64

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # b 64 64 64

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)  # b 256 64 64

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)  # b 512 32 32

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)  # b 1024 16 16

        # joint path
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)

        return map_x, ftokens, attmaps


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x