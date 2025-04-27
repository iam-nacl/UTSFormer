import torch
import torch.nn as nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from .model_utils import *
from einops import rearrange
from models.transformer_pruning_partsR import *

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



class Setr_UTSFormer(nn.Module):
    def __init__(
            self, img_size=256, in_chans=1, embed_dims=[128, 256, 384, 512],
            num_heads=[4, 4, 4, 4], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
            # depths=[2, 2, 2, 4], total_depth=16,sr_ratios=[8, 4, 2 ,1],
            depths=[2, 2, 2, 4], total_depth=16, sr_ratios=[1, 1, 1, 1],
            pretrained=None, dmodel=192, k=5, sample_ratios=0.25, classes=4,
            patch_size=8, return_map=False):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k
        self.classes = classes
        self.patch_num = int(img_size/patch_size)
        self.dmodel = dmodel

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size*patch_size*in_chans, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.dmodel),
            Rearrange('b s c -> b c s')
        )


        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
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
        # self.atm1 = HTM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], 2)
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

        self.atm3 = HTM(self.sample_ratios, self.embed_dims[2], self.embed_dims[3], self.k)
        self.stage3 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for j in range(self.depths[3])])
        self.norm3 = norm_layer(embed_dims[3])

        self.ltr4 = PTRV2(in_channels=[embed_dims[2], embed_dims[3]], out_channels=embed_dims[2])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm4 = norm_layer(embed_dims[2])

        self.ltr5 = PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage5 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm5 = norm_layer(embed_dims[1])

        self.ltr6 = PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage6 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm6 = norm_layer(embed_dims[0])


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

        attn_map = attn[:,0:2].sum(dim=1).sum(dim=1)
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

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore2 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = var_downup(attnScore2, outs[1], outs[2])
        attnScore = var_downup(attnScore, outs[0], outs[1])
        attnScore = rearrange(attnScore, 'b (h w) 1-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)


        token_dict, token_weight3, idx_cluster3, variance, score3 = self.atm3(token_dict, outs, attnScore2, ctm_stage=3)
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[3]
        outs.append(token_dict)
        variances.append(variance)


        token_dict = outs[2]
        token_dict['x'] = self.ltr4([outs[2], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm4(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)


        token_dict = outs[1]
        token_dict['x'] = self.ltr5([outs[1], outs[4]])
        for blk in self.stage5:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm5(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr6([outs[0], outs[5]])
        for blk in self.stage6:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        x = self.norm6(token_dict['x'])

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.dmodel, self.patch_num, self.patch_num)

        # decoder
        x = self.decoder(x)


        return x, attnScores, attns, outs, variances, [score1,score2,score3]




def compute_attn_score(attn):
    b, head, n, _ = attn.shape
    patch_num = int(math.sqrt(n))
    attn_map = attn[:, 0:int(head/2)].sum(dim=1).sum(dim=1)

    # 获取 batch size 和 token 数量
    b = attn_map.shape[0]  # batch size
    n = attn_map.shape[1]  # token 数量

    # 计算每个 batch 的最小值和最大值
    min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
    max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)

    # 对 attn_map 进行归一化处理
    attn_score = (attn_map - min_as_out) / (max_as_out - min_as_out)

    # 将 attn_score 调整为二维的 patch embedding 形状
    attn_score = rearrange(attn_score, 'b (h w) -> b h w', h=patch_num, w=patch_num)

    return attn_score

class Setr_DTMFormerV3(nn.Module):
    def __init__(
            self, img_size=256, in_chans=1, embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
            depths=[4, 4, 4, 4], total_depth=16, sr_ratios=[1, 1, 1, 1],
            pretrained=None, dmodel = 128, k=5, sample_ratios=0.25, classes=4,
            patch_size=8, return_map=False):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k
        self.classes = classes
        self.patch_num = int(img_size/patch_size)
        self.dmodel = dmodel

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size*patch_size*in_chans, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[3], self.dmodel),
            Rearrange('b s c -> b c s')
        )


        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
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

        self.ctm2 = ATM(self.sample_ratios/4, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ctm3 = ATM(self.sample_ratios/16, self.embed_dims[2], self.embed_dims[3], self.k)
        self.stage3 = nn.ModuleList([UTSBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for j in range(self.depths[3])])
        self.norm3 = norm_layer(embed_dims[3])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:,0:2].sum(dim=1).sum(dim=1)
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

        down_dict, token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict = blk([token_dict, down_dict])
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]

        down_dict, token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict = blk([token_dict, down_dict])
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]

        down_dict, token_dict = self.ctm3(token_dict, attnScore, ctm_stage=3)
        for blk in self.stage3:
            token_dict = blk([token_dict, down_dict])
        x = self.norm3(token_dict['x'])
        self.cur += self.depths[3]

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.dmodel, self.patch_num, self.patch_num)

        # decoder
        x = self.decoder(x)

        return [x,attnScore]


class Setr(nn.Module):
    def __init__(
            self, n_classes=4, patch_size=8, img_size=256, in_chans=1, embed_dims=512,
            num_heads=8, mlp_ratios=4, depths=16, sr_ratios=1):
        super().__init__()

        self.depths = depths
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.in_channs = in_chans

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_chans, embed_dims),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.patch_num = int(img_size / patch_size)
        self.mlp_dim = [self.embed_dims * self.mlp_ratios]
        self.dropout = 0.1
        self.num_heads = num_heads
        self.dim_head0 = self.embed_dims / self.num_heads
        self.patches_height = img_size / patch_size
        self.patches_width = img_size / patch_size
        self.num_patches = self.patches_height * self.patches_width

        self.transformer = nn.ModuleList([Block_saveAttn(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, sr_ratio=sr_ratios)
            for j in range(depths)])

        self.dmodel = embed_dims
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        attns = []

        for blk in self.transformer:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.dmodel, self.patch_num, self.patch_num)

        x = self.decoder(x)
        # return x, attns
        return x


class Setr_Biformer(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = biformer.Transformer_biformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x

class Setr_da(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = deformabletransformer.Transformer_da(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x

class Setr_kNN(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = knntransformer.Transformer_kNN(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x

class Setr_paca(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = pacavit.Transformer_paca(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x

class Setr_shunted(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = shuntedtransformer.Transformer_shunted(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x

class Setr_sta(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = svit.Transformer_sta(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x

class Setr_Swin(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = swintransformer.Transformer_Swin(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x

class Setr_MyCTM(nn.Module):
    def __init__(
            self, img_size=256, patch_size=8,in_chans=1, embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            num_stages=4, pretrained=None, classes=4,
            k=5, sample_ratios=[0.5, 0.5, 0.5],
            return_map=False,):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.depths = depths
        self.classes = classes
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size*patch_size*in_chans, 64),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.recover = MTA()


        self.dmodel = 128
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0= nn.ModuleList([Block(
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

        # image_height, image_width = pair(img_size)
        # patch_height, patch_width = pair(patch_size)
        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dims[0]))


    def forward(self, img):
        x = self.to_patch_embedding(img)

        # b, n, _ = x.shape
        # x += self.pos_embedding[:, :n]

        outs = []
        H = int(self.img_size/self.patch_size)
        W = int(self.img_size/self.patch_size)

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
        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.dmodel, H, W)

        # decoder
        x = self.decoder(x)
        return x

class Setr_Dvit(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerDvit(dim, depth, heads, dim_head, mlp_dim, dropout, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x, _ = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x