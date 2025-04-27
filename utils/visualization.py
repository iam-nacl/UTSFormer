import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import torchvision
import os
import torch
import cv2
import numpy as np
from utils.imgname import read_img_name
import matplotlib.pyplot as plt
import seaborn as sns

import math
from einops import rearrange
from torch.nn.functional import interpolate
from torchvision.utils import save_image
import torch.nn.functional as F
import re
import time
import random


def visualize_attention_maps(attns, save_dir, epoch, step):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loop over each layer
    for l, attn in enumerate(attns):
        b, h, n, _ = attn.shape  # [batch, heads, n, n]

        # Loop over each batch
        for b_idx in range(b):
            # Loop over each head
            for h_idx in range(h):
                layer_dir = os.path.join(save_dir, f"L{l}", f"B{b_idx}")
                os.makedirs(layer_dir, exist_ok=True)

                plt.imshow(attn[b_idx, h_idx].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
                # plt.imshow(attn[b_idx, h_idx].cpu().detach().numpy(), cmap="coolwarm", vmin=-0.01, vmax=0.01)
                plt.colorbar()
                plt.savefig(os.path.join(layer_dir, f"H{h_idx}.png"))
                plt.close()



def visualize_tokenscore(token_weights, save_dir, epoch, step):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, token_weight in enumerate(token_weights):
        B, HW, _ = token_weight.shape
        H = W = int(np.sqrt(HW))  # 假设 H=W

        layer_dir = os.path.join(save_dir, f"token_weight{idx + 1}")
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)

        for b in range(B):
            batch_dir = os.path.join(layer_dir, f"B{b + 1}")
            if not os.path.exists(batch_dir):
                os.makedirs(batch_dir)

            weight_map = token_weight[b].view(H, W).cpu().detach().numpy()

            plt.figure(figsize=(6, 6))
            plt.imshow(weight_map, cmap='hot', interpolation='nearest')
            plt.colorbar()

            save_path = os.path.join(batch_dir, f"tokenscore_epoch{epoch}_step{step}.png")
            plt.savefig(save_path)
            plt.close()


def save_images_and_labels(imgs, label_imgs, save_dir, img_name):
    img_save_dir = os.path.join(save_dir, "img")
    label_save_dir = os.path.join(save_dir, "label")

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    imgs = imgs.cpu().detach()
    label_imgs = label_imgs.cpu().detach()

    B = imgs.size(0)  # 获取 batch 大小

    for b in range(B):
        img = imgs[b]  # 获取第 b 个 batch 的图片，shape: (3, H, W)
        label_img = label_imgs[b]  # 获取第 b 个 batch 的标签，shape: (H, W)

        img_path = os.path.join(img_save_dir, f"{img_name}.png")
        save_image(img, img_path)

        label_img_path = os.path.join(label_save_dir, f"{img_name}.png")
        plt.imsave(label_img_path, label_img.numpy(), cmap='gray')


global_image_counter = 0

import matplotlib.pyplot as plt
import seaborn as sns

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from einops import rearrange


# 可视化 token_weights 的函数
def visualize_token_weights(token_weights, save_dir, global_image_counter, step):
    for b in range(token_weights.shape[0]):  # 遍历每个 batch
        token_weight = token_weights[b].squeeze(-1).detach().cpu().numpy()  # 去掉最后一个维度
        plt.imshow(token_weight, cmap='hot', interpolation='nearest')  # 使用热力图显示
        plt.colorbar()  # 显示颜色条
        plt.title(f'Token Weight Step {step}')

        # 保存图片
        weight_path = os.path.join(save_dir, f'token_weight_{global_image_counter+1+b}_step{step}.png')
        plt.savefig(weight_path)
        plt.close()


def visualize_attnScores(token_weights, save_dir, global_image_counter, step):
    for b in range(token_weights.shape[0]):  # 遍历每个 batch
        token_weight = token_weights[b].squeeze(-1).clone().detach().cpu().numpy()  # 去掉最后一个维度
        plt.imshow(token_weight, cmap='hot', interpolation='nearest')  # 使用热力图显示
        plt.colorbar()  # 显示颜色条
        plt.title(f'Attn Scores Step {step}')

        # 保存图片
        weight_path = os.path.join(save_dir, f'attnScores_{global_image_counter+1+b}_step{step}.png')
        plt.savefig(weight_path)
        plt.close()

def visualize_token_fusion_with_overlay2_attnScores(idx_clusters, token_weights, attnScores, imgs, labels, patch_size=8, output_size=(256, 256),
                                        save_dir='output', ratio = [1/4,1/4,1/4], image_size = 256):

    shape0 = int(1024)
    # shape1 = int(image_size * ratio[0])
    shape1 = int(shape0 * ratio[0])
    shape2 = int(shape1 * ratio[1])
    shape3 = int(shape2 * ratio[2])

    global global_image_counter  # 引入全局变量

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    # 创建保存文件夹
    result_dirs = [os.path.join(save_dir, f'mergeresult{i + 1}') for i in range(3)]
    img_dir = os.path.join(save_dir, 'imgs')
    label_dir = os.path.join(save_dir, 'labels')
    token_weight_dir = os.path.join(save_dir, 'token_weights')
    attnScore_dir = os.path.join(save_dir, 'attnScores')


    for directory in result_dirs + [img_dir, label_dir, token_weight_dir, attnScore_dir]:
        os.makedirs(directory, exist_ok=True)

    merge_maps = []
    B, C0, H, W = imgs.shape
    num_patches = (H // patch_size) * (W // patch_size)  # token数目

    # 第一次融合, idx_cluster1 直接用于还原 (保持1D形式)
    idx_cluster1 = idx_clusters[0]  # [b, 1024]
    token_weight1 = token_weights[0]  # [b, 1024, 1]
    token_weight_map1 = token_weights[0]
    attnScore1 = rearrange(attnScores[0], 'b h w -> b (h w) 1')# [b, 1024, 1]
    attnScore_map1 = rearrange(attnScores[0], 'b h w -> b (h w) 1')# [b, 1024, 1]
    merge_map1 = idx_cluster1.clone()  # 保持 [b, 1024] 不进行reshape

    # 第二次融合，使用 idx_cluster1 映射 idx_cluster2
    idx_cluster2 = idx_clusters[1]  # [b, 256]
    token_weight2 = token_weights[1]  # [b, 256, 1]
    attnScore2 = rearrange(attnScores[1], 'b h w -> b (h w) 1')  # [b, 256, 1]
    merge_map2 = torch.zeros((B, shape0), device=imgs.device)  # 创建与 merge_map1 相同尺度的tensor [b, 1024]
    token_weight_map2 = torch.zeros((B, shape0, 1), device=imgs.device)  # 同样创建 token_weight 的对应尺寸
    attnScore_map2 = torch.zeros((B, shape0, 1), device=imgs.device)


    for b in range(B):
        for idx in range(shape1):  # 遍历 idx_cluster2 的每个 token
            cluster_idx = idx_cluster2[b, idx].item()  # 获取 cluster_idx
            mask = (idx_cluster1[b] == idx)  # 找到在 merge_map1 中与 idx_cluster2 对应的 token 位置
            merge_map2[b][mask] = cluster_idx  # 将 idx_cluster2 的值映射到 merge_map2
            token_weight_map2[b][mask] = token_weight2[b, idx]  # 相应地将 token_weight2 映射到 token_weight_map2
            attnScore_map2[b][mask] = attnScore2[b, idx]  # 相应地将 token_weight2 映射到 token_weight_map2


    # 第三次融合，使用前两次的融合结果进行还原
    idx_cluster3 = idx_clusters[2]  # [b, 64]
    token_weight3 = token_weights[2]  # [b, 64, 1]
    attnScore3 = rearrange(attnScores[2], 'b h w -> b (h w) 1')  # [b, 64, 1]
    merge_map3 = torch.zeros((B, shape0), device=imgs.device)  # 与 merge_map2 对应的tensor [b, 1024]
    token_weight_map3 = torch.zeros((B, shape0, 1), device=imgs.device)  # 与 token_weight_map2 对应的尺寸
    attnScore_map3 = torch.zeros((B, shape0, 1), device=imgs.device)


    for b in range(B):
        for idx in range(shape2):  # 遍历 idx_cluster3 的每个 token
            cluster_idx = idx_cluster3[b, idx].item()  # 获取 cluster_idx
            mask_2 = (idx_cluster2[b] == idx)  # 先找到与 idx_cluster3 对应的 idx_cluster2 位置
            indices_2 = torch.nonzero(mask_2, as_tuple=True)[0]  # 获取 mask_2 中为 True 的索引
            mask_1 = torch.zeros_like(merge_map2[b], dtype=torch.bool)  # 初始化 mask_1
            for idx_2 in indices_2:  # 对每个索引，在 merge_map2 中找到对应的部分
                # mask_1 |= (merge_map2[b] == idx_2.item())
                mask_1 = mask_1 | (merge_map2[b] == idx_2.item())  # 避免原地操作
            merge_map3[b][mask_1] = cluster_idx  # 最后将结果映射到 merge_map3 中
            token_weight_map3[b][mask_1] = token_weight3[b, idx]  # 将 token_weight3 同步映射到 token_weight_map3
            attnScore_map3[b][mask_1] = attnScore3[b, idx]  # 将 token_weight3 同步映射到 token_weight_map3


    # 还原为2D形状 [b, 32, 32]，再上采样到 [b, 256, 256]
    merge_map1 = rearrange(merge_map1, 'b (h w) -> b h w', h=32, w=32)
    merge_map2 = rearrange(merge_map2, 'b (h w) -> b h w', h=32, w=32)
    merge_map3 = rearrange(merge_map3, 'b (h w) -> b h w', h=32, w=32)

    # 将 token_weight_map 同样转换为 2D 格式，但无需上采样和腐蚀
    token_weight_map1 = rearrange(token_weight_map1, 'b (h w) c -> b h w c', h=32, w=32)
    token_weight_map2 = rearrange(token_weight_map2, 'b (h w) c -> b h w c', h=32, w=32)
    token_weight_map3 = rearrange(token_weight_map3, 'b (h w) c -> b h w c', h=32, w=32)

    attnScores_map1 = rearrange(attnScore_map1, 'b (h w) c -> b h w c', h=32, w=32).squeeze(dim=-1).contiguous()
    attnScores_map2 = rearrange(attnScore_map2, 'b (h w) c -> b h w c', h=32, w=32).squeeze(dim=-1).contiguous()
    attnScores_map3 = rearrange(attnScore_map3, 'b (h w) c -> b h w c', h=32, w=32).squeeze(dim=-1).contiguous()

    attnScore_maps = [attnScores_map1, attnScores_map2, attnScores_map3]
    merge_maps = [merge_map1, merge_map2, merge_map3]

    # 生成token_weights的热力图并保存
    visualize_token_weights(token_weight_map1, token_weight_dir, global_image_counter, 1)
    visualize_token_weights(token_weight_map2, token_weight_dir, global_image_counter, 2)
    visualize_token_weights(token_weight_map3, token_weight_dir, global_image_counter, 3)

    visualize_attnScores(attnScores_map1, attnScore_dir, global_image_counter, 1)
    visualize_attnScores(attnScores_map2, attnScore_dir, global_image_counter, 2)
    visualize_attnScores(attnScores_map3, attnScore_dir, global_image_counter, 3)





    merge_maps = upsample_merge_maps(merge_maps, patch_size)  # 仅对merge_maps上采样


    # 生成图片并进行可视化处理
    kernel = np.ones((3, 3), np.uint8)  # 用于腐蚀的核
    for b in range(B):
        # 保存原图像和标签
        img = imgs[b].squeeze(0).cpu().numpy()  # [H, W] (单通道)
        label = labels[b].squeeze(0).cpu().numpy()  # [H, W] (单通道)

        global_image_counter += 1  # 全局计数器

        img_path = os.path.join(img_dir, f'image_{global_image_counter}.png')
        cv2.imwrite(img_path, (img * 255).astype(np.uint8))  # 保存灰度图

        label_path = os.path.join(label_dir, f'label_{global_image_counter}.png')
        cv2.imwrite(label_path, (label * 255).astype(np.uint8))  # 保存灰度图

        for i, merge_map in enumerate(merge_maps):
            # 将 merge_map 转为 numpy 格式并腐蚀
            merge_map_np = merge_map[b].cpu().numpy().astype(np.uint8)
            eroded_map = cv2.erode(merge_map_np, kernel, iterations=1)  # 腐蚀操作
            boundary = merge_map_np != eroded_map  # 边界线

            # 将边界线叠加到原图上，白线表示边界
            overlay_img = img.copy()
            overlay_img[boundary] = 255  # 用白色线条显示融合边界

            # 保存融合结果
            result_path = os.path.join(result_dirs[i], f'image_{global_image_counter}_merge{i + 1}.png')
            cv2.imwrite(result_path, overlay_img)

    return merge_maps, attnScore_maps
    # return merge_maps, attnScores[0], attnScores_map2, attnScores_map3



def visualize_token_fusion_with_overlay(idx_clusters, token_weights, imgs, labels, patch_size=8, output_size=(256, 256),
                                        save_dir='output', ratio = [1/4,1/4,1/4], image_size = 256):

    shape0 = int(1024)
    # shape1 = int(image_size * ratio[0])
    shape1 = int(shape0 * ratio[0])
    shape2 = int(shape1 * ratio[1])
    shape3 = int(shape2 * ratio[2])

    global global_image_counter  # 引入全局变量

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建保存文件夹
    result_dirs = [os.path.join(save_dir, f'mergeresult{i + 1}') for i in range(3)]
    img_dir = os.path.join(save_dir, 'imgs')
    label_dir = os.path.join(save_dir, 'labels')
    token_weight_dir = os.path.join(save_dir, 'token_weights')

    for directory in result_dirs + [img_dir, label_dir, token_weight_dir]:
        os.makedirs(directory, exist_ok=True)

    merge_maps = []
    B, C0, H, W = imgs.shape
    num_patches = (H // patch_size) * (W // patch_size)  # token数目

    # 第一次融合, idx_cluster1 直接用于还原 (保持1D形式)
    idx_cluster1 = idx_clusters[0]  # [b, 1024]
    token_weight1 = token_weights[0]  # [b, 1024, 1]
    token_weight_map1 = token_weights[0]
    merge_map1 = idx_cluster1.clone()  # 保持 [b, 1024] 不进行reshape

    # 第二次融合，使用 idx_cluster1 映射 idx_cluster2
    idx_cluster2 = idx_clusters[1]  # [b, 256]
    token_weight2 = token_weights[1]  # [b, 256, 1]
    merge_map2 = torch.zeros((B, shape0), device=imgs.device)  # 创建与 merge_map1 相同尺度的tensor [b, 1024]
    token_weight_map2 = torch.zeros((B, shape0, 1), device=imgs.device)  # 同样创建 token_weight 的对应尺寸

    for b in range(B):
        for idx in range(shape1):  # 遍历 idx_cluster2 的每个 token
            cluster_idx = idx_cluster2[b, idx].item()  # 获取 cluster_idx
            mask = (idx_cluster1[b] == idx)  # 找到在 merge_map1 中与 idx_cluster2 对应的 token 位置
            merge_map2[b][mask] = cluster_idx  # 将 idx_cluster2 的值映射到 merge_map2
            token_weight_map2[b][mask] = token_weight2[b, idx]  # 相应地将 token_weight2 映射到 token_weight_map2

    # 第三次融合，使用前两次的融合结果进行还原
    idx_cluster3 = idx_clusters[2]  # [b, 64]
    token_weight3 = token_weights[2]  # [b, 64, 1]
    merge_map3 = torch.zeros((B, shape0), device=imgs.device)  # 与 merge_map2 对应的tensor [b, 1024]
    token_weight_map3 = torch.zeros((B, shape0, 1), device=imgs.device)  # 与 token_weight_map2 对应的尺寸

    for b in range(B):
        for idx in range(shape2):  # 遍历 idx_cluster3 的每个 token
            cluster_idx = idx_cluster3[b, idx].item()  # 获取 cluster_idx
            mask_2 = (idx_cluster2[b] == idx)  # 先找到与 idx_cluster3 对应的 idx_cluster2 位置
            indices_2 = torch.nonzero(mask_2, as_tuple=True)[0]  # 获取 mask_2 中为 True 的索引
            mask_1 = torch.zeros_like(merge_map2[b], dtype=torch.bool)  # 初始化 mask_1
            for idx_2 in indices_2:  # 对每个索引，在 merge_map2 中找到对应的部分
                mask_1 |= (merge_map2[b] == idx_2.item())
            merge_map3[b][mask_1] = cluster_idx  # 最后将结果映射到 merge_map3 中
            token_weight_map3[b][mask_1] = token_weight3[b, idx]  # 将 token_weight3 同步映射到 token_weight_map3

    # 还原为2D形状 [b, 32, 32]，再上采样到 [b, 256, 256]
    merge_map1 = rearrange(merge_map1, 'b (h w) -> b h w', h=32, w=32)
    merge_map2 = rearrange(merge_map2, 'b (h w) -> b h w', h=32, w=32)
    merge_map3 = rearrange(merge_map3, 'b (h w) -> b h w', h=32, w=32)

    # 将 token_weight_map 同样转换为 2D 格式，但无需上采样和腐蚀
    token_weight_map1 = rearrange(token_weight_map1, 'b (h w) c -> b h w c', h=32, w=32)
    token_weight_map2 = rearrange(token_weight_map2, 'b (h w) c -> b h w c', h=32, w=32)
    token_weight_map3 = rearrange(token_weight_map3, 'b (h w) c -> b h w c', h=32, w=32)

    merge_maps = [merge_map1, merge_map2, merge_map3]

    # 生成token_weights的热力图并保存
    visualize_token_weights(token_weight_map1, token_weight_dir, global_image_counter, 1)
    visualize_token_weights(token_weight_map2, token_weight_dir, global_image_counter, 2)
    visualize_token_weights(token_weight_map3, token_weight_dir, global_image_counter, 3)

    merge_maps = upsample_merge_maps(merge_maps, patch_size)  # 仅对merge_maps上采样


    # 生成图片并进行可视化处理
    kernel = np.ones((3, 3), np.uint8)  # 用于腐蚀的核
    for b in range(B):
        # 保存原图像和标签
        img = imgs[b].squeeze(0).cpu().numpy()  # [H, W] (单通道)
        label = labels[b].squeeze(0).cpu().numpy()  # [H, W] (单通道)

        global_image_counter += 1  # 全局计数器

        img_path = os.path.join(img_dir, f'image_{global_image_counter}.png')
        cv2.imwrite(img_path, (img * 255).astype(np.uint8))  # 保存灰度图

        label_path = os.path.join(label_dir, f'label_{global_image_counter}.png')
        cv2.imwrite(label_path, (label * 255).astype(np.uint8))  # 保存灰度图

        for i, merge_map in enumerate(merge_maps):
            # 将 merge_map 转为 numpy 格式并腐蚀
            merge_map_np = merge_map[b].cpu().numpy().astype(np.uint8)
            eroded_map = cv2.erode(merge_map_np, kernel, iterations=1)  # 腐蚀操作
            boundary = merge_map_np != eroded_map  # 边界线

            # 将边界线叠加到原图上，白线表示边界
            overlay_img = img.copy()
            overlay_img[boundary] = 255  # 用白色线条显示融合边界

            # 保存融合结果
            result_path = os.path.join(result_dirs[i], f'image_{global_image_counter}_merge{i + 1}.png')
            cv2.imwrite(result_path, overlay_img)



    return merge_maps


def visualize_token_fusion_with_overlay_dict(outs, variances, imgs, labels, patch_size=8, output_size=(256, 256),
                                        save_dir='output', ratio = [1/4,1/4,1/4], image_size = 256):

    # save_dir = '/home/wzh/code/DTMFormerv2/visual/ACDC/tokenfusion/test/epoch0/'
    idx_token1 = outs[1]['idx_token'] # torch.Size([4, 1024])
    idx_token2 = outs[2]['idx_token']
    idx_token3 = outs[3]['idx_token']

    agg_weight1 = outs[1]['agg_weight'] # torch.Size([4, 1024, 1])
    agg_weight2 = outs[2]['agg_weight']
    agg_weight3 = outs[3]['agg_weight']

    variance1 = variances[0] # torch.Size([4, 1024, 1])
    variance2 = variances[1]
    variance3 = variances[2]

    idx_token_dir = os.path.join(save_dir, "idx_token")
    agg_weight_dir = os.path.join(save_dir, "agg_weight")
    variance_dir = os.path.join(save_dir, "variance")


    # Create folders if not exist
    os.makedirs(idx_token_dir, exist_ok=True)
    os.makedirs(agg_weight_dir, exist_ok=True)
    os.makedirs(variance_dir, exist_ok=True)

    # Morphological kernel for boundary emphasis in idx_token
    kernel = np.ones((3, 3), np.uint8)
    batch_size, n = idx_token1.shape
    h, w = int(n ** 0.5), int(n ** 0.5)

    idx_token1,idx_token2,idx_token3 = upsample_merge_maps([idx_token1.view(batch_size, h, w),idx_token2.view(batch_size, h, w),idx_token3.view(batch_size, h, w)], patch_size)

    # Assume B, n, h, w are defined, and variables are loaded as in the original code
    variables = [(idx_token1, agg_weight1, variance1), (idx_token2, agg_weight2, variance2),
                 (idx_token3, agg_weight3, variance3)]


    for batch in range(batch_size):
        # Process each variable set (idx_token, agg_weight, variance) for each batch
        for i, (idx_token, agg_weight, variance) in enumerate(variables):
            # Reshape idx_token, agg_weight, and variance to (h, w)
            # idx_token_resized = idx_token[batch].view(h, w).cpu().numpy()
            idx_token_resized = idx_token[batch].cpu().numpy()
            agg_weight_resized = agg_weight[batch].view(h, w).detach().cpu().numpy()
            variance_resized = variance[batch].view(h, w).cpu().numpy()

            # Scale variance between 0 and 1 per batch for visualization
            variance_rescaled = (variance_resized - variance_resized.min()) / (
                        variance_resized.max() - variance_resized.min())

            # Enhanced heatmap for agg_weight with clear granularity
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.resize(agg_weight_resized, (h, w)), cmap='hot', interpolation='nearest')
            plt.colorbar(label='Agg Weight Value')
            agg_weight_path = os.path.join(agg_weight_dir, f"agg_weight_batch{batch}_var{i + 1}.png")
            plt.savefig(agg_weight_path, dpi=200)
            plt.close()

            # Enhanced heatmap for variance with clear granularity
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.resize(variance_rescaled, (h, w)), cmap='hot', interpolation='nearest')
            plt.colorbar(label='Variance Value (0-1 Scaled)')
            variance_path = os.path.join(variance_dir, f"variance_batch{batch}_var{i + 1}.png")
            plt.savefig(variance_path, dpi=200)
            plt.close()

            # Visualize idx_token with boundary emphasis
            idx_token_np = idx_token_resized.astype(np.uint8)
            eroded_token = cv2.erode(idx_token_np, kernel, iterations=1)
            boundary = idx_token_np != eroded_token  # Detect boundaries

            # Create an overlay image (black background)
            overlay_img = np.zeros((h*patch_size, w*patch_size), dtype=np.uint8)
            overlay_img[boundary] = 255  # White lines for boundaries

            # Save idx_token with boundary overlay
            idx_token_path = os.path.join(idx_token_dir, f"idx_token_batch{batch}_var{i + 1}.png")
            cv2.imwrite(idx_token_path, overlay_img)

    return outs




def generate_unique_filename(prefix):
    timestamp = int(time.time() * 1000)  # 获取毫秒级时间戳
    random_suffix = random.randint(1000, 9999)  # 生成4位随机数
    return f"{prefix}_{timestamp}_{random_suffix}.png"

def visualize_token_fusion_with_overlay_dict_attnScores(outs, variances, attnScores, imgs, labels, patch_size=8, output_size=(256, 256),
                                                        save_dir='output', ratio=[1/4, 1/4, 1/4], image_size=256, img_name=None, score=None):
    idx_token1 = outs[1]['idx_token']
    idx_token2 = outs[2]['idx_token']
    idx_token3 = outs[3]['idx_token']

    agg_weight1 = outs[1]['agg_weight']
    agg_weight2 = outs[2]['agg_weight']
    agg_weight3 = outs[3]['agg_weight']

    variance1 = variances[0]
    variance2 = variances[1]
    variance3 = variances[2]

    attnScore1 = attnScores[0]
    attnScore2 = attnScores[1]
    attnScore3 = attnScores[2]

    idx_token_dir = os.path.join(save_dir, "idx_token")
    agg_weight_dir = os.path.join(save_dir, "agg_weight")
    score_map_dir = os.path.join(save_dir, "score_map")

    # Create folders if not exist
    os.makedirs(idx_token_dir, exist_ok=True)
    os.makedirs(agg_weight_dir, exist_ok=True)
    os.makedirs(score_map_dir, exist_ok=True)

    # Morphological kernel for boundary emphasis in idx_token
    kernel = np.ones((3, 3), np.uint8)
    batch_size, n = idx_token1.shape
    h, w = int(n ** 0.5), int(n ** 0.5)

    if score is not None:
        def generate_score_map(score, topk):
            score_map = torch.zeros_like(score)
            top_values, _ = torch.topk(score, topk, dim=-1)
            threshold = top_values[:, -1].unsqueeze(-1)  # 取 topk 中最小的值作为阈值
            score_map[score >= threshold] = score[score >= threshold]

            # 归一化 score map（最大值为1，最小值为0）
            min_val = score_map.min(dim=-1, keepdim=True).values
            max_val = score_map.max(dim=-1, keepdim=True).values
            score_map = (score_map - min_val) / (max_val - min_val + 1e-8)

            # reshape 成 [b, 32, 32]
            shape = int(math.sqrt(score_map.shape[1]))
            score_map = score_map.view(batch_size, shape, shape)
            return score_map

        # 生成 map1 到 map6
        map1 = generate_score_map(score[0], 256)
        map2 = generate_score_map(score[0], 128)
        map3 = generate_score_map(score[0], 64)
        map4 = generate_score_map(score[0], 32)
        map5 = generate_score_map(score[0], 16)
        map6 = generate_score_map(score[0], 8)

        # 可视化并保存
        for batch in range(batch_size):
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            maps = [map1, map2, map3, map4, map5, map6]
            titles = ['Top 256', 'Top 128', 'Top 64', 'Top 32', 'Top 16', 'Top 8']

            for i, ax in enumerate(axes.flatten()):
                ax.imshow(maps[i][batch].cpu().numpy(), cmap='gray')
                ax.set_title(titles[i])
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(score_map_dir, f"{img_name}_score1_maps_batch{batch}.png"), bbox_inches='tight')
            plt.close()

    idx_token1, idx_token2, idx_token3 = upsample_merge_maps(
        [idx_token1.view(batch_size, h, w), idx_token2.view(batch_size, h, w), idx_token3.view(batch_size, h, w)],
        patch_size
    )

    variables = [(idx_token1, agg_weight1, variance1, attnScore1),
                 (idx_token2, agg_weight2, variance2, attnScore2),
                 (idx_token3, agg_weight3, variance3, attnScore3)]

    for batch in range(batch_size):
        # Initialize lists to store each set of images for concatenation
        idx_token_imgs = []
        agg_weight_imgs = []
        variance_imgs = []
        attnScore_imgs = []

        for i, (idx_token, agg_weight, variance, attnScore) in enumerate(variables):
            idx_token_resized = idx_token[batch].cpu().numpy()
            agg_weight_resized = agg_weight[batch].view(h, w).detach().cpu().numpy()
            variance_resized = variance[batch].view(h, w).cpu().numpy()
            attnScore_resized = attnScore[batch].detach().cpu().numpy()

            # Scale variance between 0 and 1 per batch for visualization
            variance_rescaled = (variance_resized - variance_resized.min()) / (variance_resized.max() - variance_resized.min())

            # Visualize idx_token with boundary emphasis (non-heatmap visualization)
            idx_token_np = idx_token_resized.astype(np.uint8)
            eroded_token = cv2.erode(idx_token_np, kernel, iterations=1)
            boundary = idx_token_np != eroded_token  # Detect boundaries
            overlay_img = np.zeros((h * patch_size, w * patch_size), dtype=np.uint8)
            overlay_img[boundary] = 255
            idx_token_imgs.append(overlay_img)

            # Generate agg_weight heatmap (retain color)
            temp_agg_weight_file = generate_unique_filename(f'temp_{batch}_{i}_agg_weight')
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.resize(agg_weight_resized, (h, w)), cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(temp_agg_weight_file, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()
            agg_weight_imgs.append(cv2.imread(temp_agg_weight_file, cv2.IMREAD_COLOR))

            # Generate variance heatmap (retain color)
            temp_variance_file = generate_unique_filename(f'temp_{batch}_{i}_variance')
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.resize(variance_rescaled, (h, w)), cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(temp_variance_file, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()
            variance_imgs.append(cv2.imread(temp_variance_file, cv2.IMREAD_COLOR))

            # Generate attnScore heatmap (retain color)
            temp_attnScore_file = generate_unique_filename(f'temp_{batch}_{i}_attnScore')
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.resize(attnScore_resized, (h, w)), cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(temp_attnScore_file, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()
            attnScore_imgs.append(cv2.imread(temp_attnScore_file, cv2.IMREAD_COLOR))

            # Clean up temporary files
            if os.path.exists(temp_agg_weight_file):
                os.remove(temp_agg_weight_file)
            if os.path.exists(temp_variance_file):
                os.remove(temp_variance_file)
            if os.path.exists(temp_attnScore_file):
                os.remove(temp_attnScore_file)

        # Concatenate idx_token images
        idx_token_concat = cv2.hconcat([cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0) for img in idx_token_imgs])
        cv2.imwrite(os.path.join(idx_token_dir, f"{img_name}_idx_token_combined_batch{batch}.png"), idx_token_concat)

        # Concatenate agg_weight images (retain color)
        agg_weight_concat = cv2.hconcat([cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for img in agg_weight_imgs])
        cv2.imwrite(os.path.join(agg_weight_dir, f"{img_name}_agg_weight_combined_batch{batch}.png"), agg_weight_concat)

    return outs


def visualize_token_fusion_with_overlay_dict_attnScores_patch16(outs, variances, attnScores, imgs, labels,
                                                                patch_size=8, output_size=(256, 256),
                                                                save_dir='output', ratio=[1 / 4, 1 / 4, 1 / 4],
                                                                image_size=256, img_name=None, score=None):

    idx_token1 = outs[1]['idx_token']  # torch.Size([4, 1024])
    idx_token2 = outs[2]['idx_token']

    agg_weight1 = outs[1]['agg_weight']  # torch.Size([4, 1024, 1])
    agg_weight2 = outs[2]['agg_weight']

    variance1 = variances[0]  # torch.Size([4, 1024, 1])
    variance2 = variances[1]

    attnScore1 = attnScores[0]  # torch.Size([4, 1024, 1])
    attnScore2 = attnScores[1]

    idx_token_dir = os.path.join(save_dir, "idx_token")
    agg_weight_dir = os.path.join(save_dir, "agg_weight")
    score_map_dir = os.path.join(save_dir, "score_map")

    # Create folders if not exist
    os.makedirs(idx_token_dir, exist_ok=True)
    os.makedirs(agg_weight_dir, exist_ok=True)
    os.makedirs(score_map_dir, exist_ok=True)

    # Morphological kernel for boundary emphasis in idx_token
    kernel = np.ones((3, 3), np.uint8)
    batch_size, n = idx_token1.shape
    h, w = int(n ** 0.5), int(n ** 0.5)

    if score is not None:
        def generate_score_map(score, topk):
            score_map = torch.zeros_like(score)
            top_values, _ = torch.topk(score, topk, dim=-1)
            threshold = top_values[:, -1].unsqueeze(-1)  # 取 topk 中最小的值作为阈值
            score_map[score >= threshold] = score[score >= threshold]

            # 归一化 score map（最大值为1，最小值为0）
            min_val = score_map.min(dim=-1, keepdim=True).values
            max_val = score_map.max(dim=-1, keepdim=True).values
            score_map = (score_map - min_val) / (max_val - min_val + 1e-8)

            # reshape 成 [b, 32, 32]
            shape = int(math.sqrt(score_map.shape[1]))
            score_map = score_map.view(batch_size, shape, shape)
            return score_map

        # 生成 map1 到 map6
        map1 = generate_score_map(score[0], 128)
        map2 = generate_score_map(score[0], 64)
        map3 = generate_score_map(score[0], 32)
        map4 = generate_score_map(score[0], 16)
        map5 = generate_score_map(score[0], 8)
        map6 = generate_score_map(score[0], 4)

        # 可视化并保存
        for batch in range(batch_size):
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            maps = [map1, map2, map3, map4, map5, map6]
            titles = ['Top 128', 'Top 64', 'Top 32', 'Top 16', 'Top 8', 'Top 4']

            for i, ax in enumerate(axes.flatten()):
                ax.imshow(maps[i][batch].cpu().numpy(), cmap='gray')
                ax.set_title(titles[i])
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(score_map_dir, f"{img_name}_score1_maps_batch{batch}.png"), bbox_inches='tight')
            plt.close()

    idx_token1, idx_token2 = upsample_merge_maps(
        [idx_token1.view(batch_size, h, w), idx_token2.view(batch_size, h, w)],
        patch_size
    )

    variables = [(idx_token1, agg_weight1, variance1, attnScore1),
                 (idx_token2, agg_weight2, variance2, attnScore2)]

    for batch in range(batch_size):
        # Initialize lists to store each set of images for concatenation
        idx_token_imgs = []
        agg_weight_imgs = []
        variance_imgs = []
        attnScore_imgs = []

        for i, (idx_token, agg_weight, variance, attnScore) in enumerate(variables):
            idx_token_resized = idx_token[batch].cpu().numpy()
            agg_weight_resized = agg_weight[batch].view(h, w).detach().cpu().numpy()
            variance_resized = variance[batch].view(h, w).cpu().numpy()
            attnScore_resized = attnScore[batch].detach().cpu().numpy()

            # Scale variance between 0 and 1 per batch for visualization
            variance_rescaled = (variance_resized - variance_resized.min()) / (
                        variance_resized.max() - variance_resized.min())

            # Visualize idx_token with boundary emphasis (non-heatmap visualization)
            idx_token_np = idx_token_resized.astype(np.uint8)
            eroded_token = cv2.erode(idx_token_np, kernel, iterations=1)
            boundary = idx_token_np != eroded_token  # Detect boundaries
            overlay_img = np.zeros((h * patch_size, w * patch_size), dtype=np.uint8)
            overlay_img[boundary] = 255
            idx_token_imgs.append(overlay_img)

            # Generate agg_weight heatmap (retain color)
            temp_agg_weight_file = generate_unique_filename(f'temp_{batch}_{i}_agg_weight')
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.resize(agg_weight_resized, (h, w)), cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(temp_agg_weight_file, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()
            agg_weight_imgs.append(cv2.imread(temp_agg_weight_file, cv2.IMREAD_COLOR))

            # Generate variance heatmap (retain color)
            temp_variance_file = generate_unique_filename(f'temp_{batch}_{i}_variance')
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.resize(variance_rescaled, (h, w)), cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(temp_variance_file, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()
            variance_imgs.append(cv2.imread(temp_variance_file, cv2.IMREAD_COLOR))

            # Generate attnScore heatmap (retain color)
            temp_attnScore_file = generate_unique_filename(f'temp_{batch}_{i}_attnScore')
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.resize(attnScore_resized, (h, w)), cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(temp_attnScore_file, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()
            attnScore_imgs.append(cv2.imread(temp_attnScore_file, cv2.IMREAD_COLOR))

            # Clean up temporary files
            if os.path.exists(temp_agg_weight_file):
                os.remove(temp_agg_weight_file)
            if os.path.exists(temp_variance_file):
                os.remove(temp_variance_file)
            if os.path.exists(temp_attnScore_file):
                os.remove(temp_attnScore_file)

        # Concatenate idx_token images
        idx_token_concat = cv2.hconcat(
            [cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0) for img in idx_token_imgs])
        cv2.imwrite(os.path.join(idx_token_dir, f"{img_name}_idx_token_combined_batch{batch}.png"), idx_token_concat)

        # Concatenate agg_weight images (retain color)
        agg_weight_concat = cv2.hconcat(
            [cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for img in agg_weight_imgs])
        cv2.imwrite(os.path.join(agg_weight_dir, f"{img_name}_agg_weight_combined_batch{batch}.png"), agg_weight_concat)

    return outs



def visualize_token_fusion_with_overlay_rgb(idx_clusters, token_weights, imgs, labels, patch_size=8, output_size=(256, 256),
                                        save_dir='output', ratio = [1/4,1/4,1/4], image_size = 256):
    global global_image_counter  # 引入全局变量

    """
    RGB图像输入，进行token融合可视化和叠加

    :param idx_clusters: 记录token融合的idx_cluster列表 [idx_cluster1, idx_cluster2, idx_cluster3]
    :param imgs: 原图像，形状为[B, 3, H, W]（RGB三通道）
    :param patch_size: 每个patch的大小，默认为8
    :param output_size: 最终输出图片的尺寸，默认是(256, 256)
    :param save_dir: 保存结果的文件夹
    """
    shape0 = int(1024)
    shape1 = int(shape0 * ratio[0])
    shape2 = int(shape1 * ratio[1])
    shape3 = int(shape2 * ratio[2])

    global global_image_counter  # 引入全局变量

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建保存文件夹
    result_dirs = [os.path.join(save_dir, f'mergeresult{i + 1}') for i in range(3)]
    img_dir = os.path.join(save_dir, 'imgs')
    label_dir = os.path.join(save_dir, 'labels')
    token_weight_dir = os.path.join(save_dir, 'token_weights')

    for directory in result_dirs + [img_dir, label_dir, token_weight_dir]:
        os.makedirs(directory, exist_ok=True)

    merge_maps = []
    B, C0, H, W = imgs.shape
    num_patches = (H // patch_size) * (W // patch_size)  # token数目

    # 第一次融合, idx_cluster1 直接用于还原 (保持1D形式)
    idx_cluster1 = idx_clusters[0]  # [b, 1024]
    token_weight1 = token_weights[0]  # [b, 1024, 1]
    token_weight_map1 = token_weights[0]
    merge_map1 = idx_cluster1.clone()  # 保持 [b, 1024] 不进行reshape


    # 第二次融合，使用 idx_cluster1 映射 idx_cluster2
    idx_cluster2 = idx_clusters[1]  # [b, 256]
    token_weight2 = token_weights[1]  # [b, 256, 1]
    merge_map2 = torch.zeros((B, shape0), device=imgs.device)  # 创建与 merge_map1 相同尺度的tensor [b, 1024]
    token_weight_map2 = torch.zeros((B, shape0, 1), device=imgs.device)  # 同样创建 token_weight 的对应尺寸
    for b in range(B):
        for idx in range(shape1):  # 遍历 idx_cluster2 的每个 token
            cluster_idx = idx_cluster2[b, idx].item()  # 获取 cluster_idx
            mask = (idx_cluster1[b] == idx)  # 找到在 merge_map1 中与 idx_cluster2 对应的 token 位置
            merge_map2[b][mask] = cluster_idx  # 将 idx_cluster2 的值映射到 merge_map2
            token_weight_map2[b][mask] = token_weight2[b, idx]  # 相应地将 token_weight2 映射到 token_weight_map2


    # 第三次融合，使用前两次的融合结果进行还原
    idx_cluster3 = idx_clusters[2]  # [b, 64]
    token_weight3 = token_weights[2]  # [b, 64, 1]

    merge_map3 = torch.zeros((B, shape0), device=imgs.device)  # 与 merge_map2 对应的tensor [b, 1024]
    token_weight_map3 = torch.zeros((B, shape0, 1), device=imgs.device)  # 与 token_weight_map2 对应的尺寸

    for b in range(B):
        for idx in range(shape2):  # 遍历 idx_cluster3 的每个 token
            cluster_idx = idx_cluster3[b, idx].item()  # 获取 cluster_idx
            mask_2 = (idx_cluster2[b] == idx)  # 先找到与 idx_cluster3 对应的 idx_cluster2 位置
            indices_2 = torch.nonzero(mask_2, as_tuple=True)[0]  # 获取 mask_2 中为 True 的索引
            mask_1 = torch.zeros_like(merge_map2[b], dtype=torch.bool)  # 初始化 mask_1
            for idx_2 in indices_2:  # 对每个索引，在 merge_map2 中找到对应的部分
                mask_1 |= (merge_map2[b] == idx_2.item())
            merge_map3[b][mask_1] = cluster_idx  # 最后将结果映射到 merge_map3 中
            token_weight_map3[b][mask_1] = token_weight3[b, idx]  # 将 token_weight3 同步映射到 token_weight_map3


    # 还原为2D形状 [b, 32, 32]，再上采样到 [b, 256, 256]
    merge_map1 = rearrange(merge_map1, 'b (h w) -> b h w', h=32, w=32)
    merge_map2 = rearrange(merge_map2, 'b (h w) -> b h w', h=32, w=32)
    merge_map3 = rearrange(merge_map3, 'b (h w) -> b h w', h=32, w=32)

    # 将 token_weight_map 同样转换为 2D 格式，但无需上采样和腐蚀
    token_weight_map1 = rearrange(token_weight_map1, 'b (h w) c -> b h w c', h=32, w=32)
    token_weight_map2 = rearrange(token_weight_map2, 'b (h w) c -> b h w c', h=32, w=32)
    token_weight_map3 = rearrange(token_weight_map3, 'b (h w) c -> b h w c', h=32, w=32)

    merge_maps = [merge_map1, merge_map2, merge_map3]

    visualize_token_weights(token_weight_map1, token_weight_dir, global_image_counter, 1)
    visualize_token_weights(token_weight_map2, token_weight_dir, global_image_counter, 2)
    visualize_token_weights(token_weight_map3, token_weight_dir, global_image_counter, 3)

    merge_maps = upsample_merge_maps(merge_maps, patch_size)

    # 生成图片并进行可视化处理
    kernel = np.ones((3, 3), np.uint8)  # 用于腐蚀的核
    for b in range(B):
        # 保存原图像和标签
        img = imgs[b].permute(1, 2, 0).cpu().numpy()  # [H, W, 3] (RGB三通道)
        label = labels[b].squeeze(0).cpu().numpy()  # [H, W] (单通道)

        # 使用全局计数器来确保唯一命名
        global_image_counter += 1

        img_path = os.path.join(img_dir, f'image_{global_image_counter}.png')
        cv2.imwrite(img_path, (img * 255).astype(np.uint8))  # 保存RGB图

        label_path = os.path.join(label_dir, f'label_{global_image_counter}.png')
        cv2.imwrite(label_path, (label * 255).astype(np.uint8))  # 保存灰度图

        for i, merge_map in enumerate(merge_maps):
            # 将 merge_map 转为 numpy 格式并腐蚀
            merge_map_np = merge_map[b].cpu().numpy().astype(np.uint8)
            eroded_map = cv2.erode(merge_map_np, kernel, iterations=1)  # 腐蚀操作
            boundary = merge_map_np != eroded_map  # 边界线

            # 将边界线叠加到原图上，白线表示边界
            overlay_img = img.copy()
            overlay_img[boundary] = [255, 255, 255]  # 用白色线条显示融合边界

            # 保存融合结果
            result_path = os.path.join(result_dirs[i], f'image_{global_image_counter}_merge{i + 1}.png')
            cv2.imwrite(result_path, overlay_img)


    return merge_maps








def upsample_merge_maps(merge_maps, patch_size):
    # 假设 merge_maps 是一个列表，每个元素是形状为 [b, h, w]
    upsampled_maps = []
    for map in merge_maps:
        # 先将每个 map 的形状调整为 [b, h, w]
        b, h, w = map.shape

        # 使用 repeat_interleave 上采样
        upsampled_map = map.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)

        upsampled_maps.append(upsampled_map)

    return upsampled_maps


def featuremap_visual(feature,
                      out_dir='./utils/visualization',
                      save_feature=True,
                      show_feature=True,
                      feature_title=None,
                      channel = None,
                      num_ch=-1,
                      nrow=1,
                      padding=10,
                      pad_value=1
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0]
    #'''
    feature_c = feature.view(c, h*w)
    fmax = torch.max(feature_c, dim=-1)[0]
    fmin = torch.min(feature_c, dim=-1)[0]
    feature = (feature - fmin[:, None, None]) / (fmax[:, None, None] - fmin[:, None, None])
    #'''

    feature = feature.unsqueeze(1)
    if channel:
        feature = feature[channel]
    else:
        if c > num_ch > 0:
            feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)
    title = title + "_" + str(h) + '-' + str(w)

    plt.title(title)
    plt.imshow(images)
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
        plt.savefig(out_root)
    if show_feature:        plt.show()
    a = 1

def attentionmap_visual(features,
                      out_dir='./utils/visualization',  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=True,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      channel = None,
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = features.shape
    if b > 6:
        b = 6
    for i in range(b):
        figure = np.zeros(((h+30)*2, (w+30)*(c//2)+30), dtype=np.uint8) + 255
        for j in range(c):
            featureij = features[i, j, :, :]
            fmax = torch.max(featureij)
            fmin = torch.min(featureij)
            #featureij = ((featureij - fmin)/(fmax-fmin))*255
            featureij = (featureij / fmax) * 255
            featureij = featureij.cpu().detach().numpy()
            featureij = featureij.astype('uint8')
            #cv2.imshow("attention-" + str(c), featureij)
            #cv2.waitKey(0)
            if j < c//2:
                figure[10:h+10, 10+(w+20)*j: 10+(w+20)*j+w] = featureij
            else:
                figure[30+h:30+h+h, 10 + (w + 20) * (j-c//2): 10 + (w + 20) * (j-c//2) + w] = featureij
        if feature_title:
            cv2.imwrite(out_dir + '/' + 'attention_' + feature_title + '.png', figure)
        else:
            cv2.imwrite(out_dir + '/' + 'attention_' + str(i) + '_' + str(h) + '.png', figure)
        cv2.imshow("attention-" + str(c), figure)
        cv2.waitKey(0)


def featuremap1d_visual(feature,
                      out_dir='./utils/visualization',
                      save_feature=True,
                      show_feature=True,
                      feature_title=None,
                      channel = None,
                      num_ch=-1,
                      nrow=1,
                      padding=10,
                      pad_value=1
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0]
    feature = feature.unsqueeze(1)
    if channel:
        feature = feature[channel]
    else:
        if c > num_ch > 0:
            feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)
    title = title + "_" + str(h) + '-' + str(w)

    plt.title(title)
    plt.imshow(images)
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
        plt.savefig(out_root)
    if show_feature:        plt.show()

def visual_box(bbox, name='layer2', scale=1):
    bbox = np.array(bbox.cpu())  # x y x y
    bbox = bbox * scale
    imgpath = read_img_name()
    img = cv2.imread(imgpath)
    bboxnum = bbox.shape[0]
    if bbox.any():
        for i in range(bboxnum):
            cv2.rectangle(img, (int(bbox[i, 0]), int(bbox[i, 1])), (int(bbox[i, 2]), int(bbox[i, 3])), (0, 0, 255), 1)  # x1y1,x2y2,BGR
    filename = os.path.basename(imgpath)
    filename = filename.split('.')[0]
    save_path = './Visualization/localbox/' + name + '/'
    cv2.imwrite(os.path.join(save_path, filename + '.png'), img)

global layer
layer = 0

def attentionheatmap_visual(features,
                      out_dir='./VisualizationP/heatmap/',
                      save_feature=True,
                      show_feature=True,
                      feature_title=None,
                      channel = None,
                      ):

    # feature = feature.detach().cpu()
    global layer
    b, c, h, w = features.shape
    if b > 1:
        b = 1
    for i in range(b):
        figure = np.zeros(((h+30)*2, (w+30)*(c//2)+30), dtype=np.uint8) + 255
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()
            #featureij = featureij.astype('uint8')
            fig = sns.heatmap(featureij, cmap="YlGnBu", vmin=0.0, vmax=0.005)  #Wistia, YlGnBu
            fig.set_xticks(range(0))
            #fig.set_xticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            fig.set_yticks(range(0))
            #fig.set_yticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            #sns.despine()
            plt.show()
            plt.close()
            fig_heatmap = fig.get_figure()
            imgpath = read_img_name()
            filename = os.path.basename(imgpath)
            filename = filename.split('.')[0]
            fig_heatmap.savefig(os.path.join(out_dir, filename + '_l' + str(layer) + '_' + str(j) + '.png'))
            layer = (layer + 1) % 4