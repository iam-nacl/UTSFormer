from distutils.archive_util import make_archive
from re import M
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
np.set_printoptions(threshold=1000)
import cv2
import random
#from utils.visualization import featuremap_visual, featuremap1d_visual

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def heat_pos_embed(height=16, width=16, sigma=0.2): #0.6
    heatmap = np.zeros((1, height*width, height, width))
    factor = 1/(2*sigma*sigma)
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x)/width) ** 2
            y_vec = ((np.arange(0, height) - y)/height) ** 2
            xv, yv = np.meshgrid(x_vec, y_vec)
            exponent = factor * (xv + yv)
            exp = np.exp(-exponent)
            heatmap[0, y*height + x, :, :] = exp
    return heatmap


def pos_grid(height=32, width=32):
    grid = np.zeros((1, height * width, height, width))
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x) / width) ** 2
            y_vec = ((np.arange(0, height) - y) / height) ** 2
            yv, xv = np.meshgrid(x_vec, y_vec)
            disyx = (yv + xv)
            grid[0, y * width + x, :, :] = disyx
    return grid


def pos_grid_mask(height=32, width=32, thresh=0.25):
    grid = np.zeros((1, height * width, height, width))
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x) / width) ** 2
            y_vec = ((np.arange(0, height) - y) / height) ** 2
            yv, xv = np.meshgrid(x_vec, y_vec)
            disyx = (yv + xv)
            disyx[disyx > thresh] = -1
            disyx[disyx >= 0] = 1
            disyx[disyx == -1] = 0
            grid[0, y * width + x, :, :] = disyx
    return grid


def relative_pos_index(height=32, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += height - 1  # shift to start from 0 # the relative pos in y axial
    relative_coords[:, :, 1] += weight - 1  # the relative pos in x axial
    relative_coords[:, :, 0] *= 2*weight - 1 # the 1d pooling pos to recoard the pos
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index


def relative_pos_index_dis(height=32, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    relative_coords[:, :, 0] += height - 1  # shift to start from 0 # the relative pos in y axial
    relative_coords[:, :, 1] += weight - 1  # the relative pos in x axial
    relative_coords[:, :, 0] *= 2*weight - 1 # the 1d pooling pos to recoard the pos
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index, dis


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        #self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2pm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, prob, **kwargs):
        return self.fn(self.norm(x), prob, **kwargs)


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


class FeedForwardVTP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., pruning_rate=0.2, num_patches=128):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(int(dim*(1-pruning_rate)), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(hidden_dim*(1-pruning_rate)), dim),
            nn.Dropout(dropout)
        )
        self.pruning_rate = pruning_rate
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.get_scores1 = nn.Linear(num_patches, 1)
        self.get_scores2 = nn.Linear(num_patches, 1)

    def forward(self, x):
        b, n, d = x.shape
        x = rearrange(x, "b n d -> b d n")
        a_soft = self.get_scores1(x).reshape(b, d) # b d
        a_soft_order, a_soft_index = torch.sort(a_soft, -1, descending=True)
        for i in range(b):
            a_soft[i, a_soft_index[i, :int(self.dim * (1 - self.pruning_rate))]] = 1
            a_soft[i, a_soft_index[i, int(self.dim * (1 - self.pruning_rate)):]] = 0
        x = x[a_soft == 1, :].reshape(b, -1, n)
        x = rearrange(x, "b d n -> b n d")
        x = self.net1(x)

        b, n, d = x.shape
        x = rearrange(x, "b n d -> b d n")
        a_soft = self.get_scores2(x).reshape(b, d) # b d
        a_soft_order, a_soft_index = torch.sort(a_soft, -1, descending=True)
        for i in range(b):
            a_soft[i, a_soft_index[i, :int(self.hidden_dim * (1 - self.pruning_rate))]] = 1
            a_soft[i, a_soft_index[i, int(self.hidden_dim * (1 - self.pruning_rate)):]] = 0
        x = x[a_soft == 1, :].reshape(b, -1, n)
        x = rearrange(x, "b d n -> b n d")
        x = self.net2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), torch.cat((q, k, v), dim=-1), attn #attn


class AttentionGRPE(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=1024):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        height, width = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        # ------------------ relative position embedding -------------------
        relative_position_index, dis = relative_pos_index_dis(height, width)
        self.relative_position_index = relative_position_index  # hw hw
        self.dis = dis.cuda()  # hw hw
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.range(1, heads) * 0.1, requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.height = height
        self.weight = width

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, rpe=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.weight, self.height * self.weight, -1)  # n n h
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            factor = 1 / (2 * self.headsita ** 2 + 1e-10)  # g
            exponent = factor[:, None, None] * self.dis[None, :, :]  # g hw hw
            pos_embed = torch.exp(-exponent)  # g hw hw
            dots = dots0 + relative_position_bias.unsqueeze(0) + 0.01*pos_embed[None, :, :, :]
        else:
            dots = dots0

        attn = self.attend(dots)  # b g n n
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), self.attend(dots0)


class AttentionDvit(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=1024):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        height, width = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        # ------------------ relative position embedding -------------------
        relative_position_index, dis = relative_pos_index_dis(height, width)
        self.relative_position_index = relative_position_index  # hw hw
        self.dis = dis.cuda()  # hw hw
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.range(1, heads) * 0.1, requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.height = height
        self.weight = width

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy=None, rpe=True):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.weight, self.height * self.weight, -1)  # n n h
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            factor = 1 / (2 * self.headsita ** 2 + 1e-10)  # g
            exponent = factor[:, None, None] * self.dis[None, :, :]  # g hw hw
            pos_embed = torch.exp(-exponent)  # g hw hw
            dots = dots0 #+ relative_position_bias.unsqueeze(0) + 0.01*pos_embed[None, :, :, :]
        else:
            dots = dots0
        
        if policy is None:
            attn = self.attend(dots)
        else:
            attn = self.softmax_with_policy(dots, policy)
            
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), self.attend(dots0)


class AttentionVTP(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=1024, pruning_rate=0.2):
        super().__init__()
        self.inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        height, width = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
        dim_pruned = int (dim * (1 - pruning_rate))

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim_pruned, self.inner_dim*3, bias=False)

        # ------------------ relative position embedding -------------------
        relative_position_index, dis = relative_pos_index_dis(height, width)
        self.relative_position_index = relative_position_index  # hw hw
        self.dis = dis.cuda()  # hw hw
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.range(1, heads) * 0.1, requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.height = height
        self.weight = width

        # -----------------add function-----------------
        self.dim = dim
        self.dim_head  = dim_head
        self.pruning_rate = pruning_rate
        self.get_scores = nn.Linear(num_patches, 1)
        self.get_scoresx = nn.Linear(num_patches, 1)

        self.to_out = nn.Sequential(
            nn.Linear(int(self.dim_head * (1-pruning_rate))*heads, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, rpe=True):
        # ---------- pruning the X --------------
        b, n, d = x.shape
        x = rearrange(x, "b n d -> b d n")
        a_soft = self.get_scoresx(x).reshape(b, d) # b d
        a_soft_order, a_soft_index = torch.sort(a_soft, -1, descending=True)
        for i in range(b):
            a_soft[i, a_soft_index[i, :int(self.dim * (1 - self.pruning_rate))]] = 1
            a_soft[i, a_soft_index[i, int(self.dim * (1 - self.pruning_rate)):]] = 0
        x = x[a_soft == 1, :].reshape(b, -1, n)
        x = rearrange(x, "b d n -> b n d")

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.weight, self.height * self.weight, -1)  # n n h
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            factor = 1 / (2 * self.headsita ** 2 + 1e-10)  # g
            exponent = factor[:, None, None] * self.dis[None, :, :]  # g hw hw
            pos_embed = torch.exp(-exponent)  # g hw hw
            dots = dots0 + relative_position_bias.unsqueeze(0) + 0.01*pos_embed[None, :, :, :]
        else:
            dots = dots0

        attn = self.attend(dots)  # b g n n
        out = torch.matmul(attn, v)

        # ---------- pruning the out --------------
        b, g, n, d = out.shape
        out = rearrange(out, "b g n d -> b g d n")
        a_soft = self.get_scores(out).reshape(b, g, d) # b g d
        a_soft_order, a_soft_index = torch.sort(a_soft, -1, descending=True)
        for i in range(b):
            for j in range(g):
                a_soft[i, j, a_soft_index[i, j, :int(d * (1 - self.pruning_rate))]] = 1
                a_soft[i, j, a_soft_index[i, j, int(d * (1 - self.pruning_rate)):]] = 0
        out = out[a_soft == 1, :].reshape(b, g, -1, n)
        out = rearrange(out, "b g d n -> b g n d")
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), self.attend(dots0)


class TransformerVTP(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128, pruning_rate = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionVTP(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches, pruning_rate=pruning_rate)),
                PreNorm(dim, FeedForwardVTP(dim, mlp_dim, dropout=dropout, pruning_rate=pruning_rate, num_patches=num_patches))
            ]))
    def forward(self, x):
        attns = []
        for attn, ff in self.layers:
            ax, attnmap = attn(x)
            attns.append(attnmap)
            x = ax + x
            x = ff(x) + x
        return x, attns


class TransformerDown_VTP(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
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

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

       # -------------- parmameters for vision transformer pruning ------------
        self.pruning_rate = 0.2
        self.transformer = TransformerVTP(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches, pruning_rate=self.pruning_rate)

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        #x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer layer
        ax, attns = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out, x, attns

# ==================================================================================================================================================================================
class TransformerAdavit(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionGRPE(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.depth = depth
        self.cumul_score = torch.zeros(1, num_patches).cuda()
        self.remainder = torch.ones(1, num_patches).cuda()
        self.ponder = torch.zeros(1, num_patches).cuda()
        self.mask = torch.ones(1, num_patches).cuda()
        self.e = 0.01
        self.sig = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)
    def forward(self, x):
        p_loss, attns = [], []
        b, n, _ = x.shape
        cumul_score = self.cumul_score.repeat(b, 1)
        remainder = self.remainder.repeat(b, 1)
        ponder = self.ponder.repeat(b, 1)
        mask = self.mask.repeat(b, 1)
        for index, (attn, ff) in enumerate(self.layers):
            x = x * mask[:, :, None]
            if index < self.depth-1:
                h = self.sig(self.alpha*x[:, :, 0] + self.beta) # b n
            else:
                h = (x[:, :, 0] + 1)/(x[:, :, 0] + 1) # i.e., 1
            cumul_score = cumul_score + h
            ponder = ponder + mask
            remainder[cumul_score < 1-self.e] = remainder[cumul_score < 1-self.e] - h[cumul_score < 1-self.e]
            ponder[cumul_score >= 1-self.e] = ponder[cumul_score >= 1-self.e] + remainder[cumul_score >= 1-self.e]
            mask = cumul_score < 1 - self.e
            # -----------------------
            ax, attnmap = attn(x)
            p_loss.append(torch.sum(ponder)/n)
            attns.append(attnmap)
            x = ax + x
            x = ff(x) + x
        return x, attns


class TransformerAdavit_inference(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionGRPE(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.depth = depth
        self.cumul_score = torch.zeros(1, num_patches).cuda()
        self.remainder = torch.ones(1, num_patches).cuda()
        self.ponder = torch.zeros(1, num_patches).cuda()
        self.mask = torch.ones(1, num_patches).cuda()
        self.e = 0.01
        self.sig = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)
    def forward(self, x):
        p_loss, attns = [], []
        b, n, d = x.shape
        cumul_score = self.cumul_score.repeat(b, 1)
        remainder = self.remainder.repeat(b, 1)
        ponder = self.ponder.repeat(b, 1)
        mask = self.mask.repeat(b, 1)
        for index, (attn, ff) in enumerate(self.layers):
            #x = x * mask[:, :, None]
            mask[0, 0:1024-960] = 0
            xp = x[0, mask[0, :]>0, :][None, :, :]
            #print(xp.shape)
            if index < self.depth-1:
                h = self.sig(self.alpha*xp[:, :, 0] + self.beta) # b n
            else:
                h = (xp[:, :, 0] + 1)/(xp[:, :, 0] + 1) # i.e., 1
            cumul_score[:, mask[0, :]>0] = cumul_score[:, mask[0, :]>0] + h
            # -----------------------
            ax, attnmap = attn(xp)
            p_loss.append(torch.sum(ponder)/n)
            attns.append(attnmap)
            xp = ax + xp
            xp = ff(xp) + xp
            x[:, mask[0, :]>0, :] = xp
            mask = cumul_score < 1 - self.e
        return x, attns


class TransformerDown_Adavit(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
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

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

       # -------------- parmameters for vision transformer pruning ------------
        self.transformer = TransformerAdavit(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        #x += self.pos_embedding[:, :n] # or using relative position embedding?
        x = self.dropout(x)
        # transformer layer
        ax, attns = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out, x, attns

# ==================================================================================================================================================================================

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


class TransformerDvit(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, AttentionDvit(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.depth = depth
        self.obtain_score = PredictorLG(embed_dim=dim)
    
    def forward(self, x):
        attns = []
        b, n, _ = x.shape
        prev_decision = torch.ones(b, n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(b, n, 1, dtype=x.dtype, device=x.device)
        for index, (attn, ff) in enumerate(self.layers):
            if index==1:
                pred_score = self.obtain_score(x, prev_decision) # b n 2
                policy = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                ax, attnmap = attn(x, policy)
                prev_decision = pred_score
            else:
                ax, attnmap = attn(x, policy)
            attns.append(attnmap)
            x = ax + x
            x = ff(x) + x
        return x, attns


class TransformerDvit_inference(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, AttentionDvit(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.depth = depth
        self.obtain_score = PredictorLG(embed_dim=dim)
    
    def forward(self, x):
        attns = []
        b, n, d = x.shape
        prev_decision = torch.ones(b, n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(b, n, 1, dtype=x.dtype, device=x.device)
        for index, (attn, ff) in enumerate(self.layers):
            if index==1:
                pred_score = self.obtain_score(x, prev_decision) # b n 2
                policy = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision # policy b n 1, x: b n 128
                policy_ = policy.repeat(1, 1, d)
                ax = x[policy_ > 0].reshape(b, -1, d)
                policy_ = policy_[policy_ > 0].reshape(b, -1, d)
                ax, attnmap = attn(ax, policy_[:, :, :1])
                prev_decision = pred_score
            else:
                policy_ = policy.repeat(1, 1, d)
                ax = x[policy_ > 0].reshape(b, -1, d)
                policy_ = policy_[policy_ > 0].reshape(b, -1, d)
                ax, attnmap = attn(ax, policy_[:, :, :1])
            attns.append(attnmap)
            xx = x[:, policy[0, :, 0]>0, :]
            #print(index, xx.shape, ax.shape)
            x[:, policy[0, :, 0]>0, :] = ax + xx
            x = ff(x) + x
        return x, attns


class TransformerDown_Dvit(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
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

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

       # -------------- parmameters for vision transformer pruning ------------
        self.transformer = TransformerDvit_inference(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        #x += self.pos_embedding[:, :n]  # or using relative position embedding?
        x = self.dropout(x)
        # transformer layer
        ax, attns = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out, x, attns