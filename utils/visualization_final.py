import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from hyperopt.vectorize import idxs_map
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def visual_attentionheatmap(features,
                      out_dir='./VisualizationP/heatmap/',
                      layer = 0,
                      filename = None,
                      save_feature=True,
                      show_feature=True,
                      feature_title=None,
                      channel = None,
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = features.shape
    if b > 1:
        b = 1
    for i in range(b):
        figure = np.zeros(((h+30)*2, (w+30)*(c//2)+30), dtype=np.uint8) + 255
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()
            #featureij = featureij.astype('uint8')
            # fig = sns.heatmap(featureij, cmap="YlGnBu", vmin=0.0, vmax=0.005)  #Wistia, YlGnBu
            fig = sns.heatmap(featureij, cmap="YlGnBu", vmin=0.0, vmax=0.01)  #Wistia, YlGnBu

            fig.set_xticks(range(0))
            #fig.set_xticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            fig.set_yticks(range(0))
            #fig.set_yticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            #sns.despine()
            plt.show()
            plt.close()
            fig_heatmap = fig.get_figure()
            # fig_heatmap.savefig(os.path.join(out_dir, filename + '_l' + str(layer) + '_' + f'head{j}' + '.png'))

def visual_attentionheatmap_draw(features,
                      out_dir='./VisualizationP/heatmap/',
                      layer = 0,
                      filename = None,
                      save_feature=True,
                      show_feature=True,
                      feature_title=None,
                      channel = None,
                      ):
    if layer == 1:
        from scipy.ndimage import zoom
        # feature  = (features.squeeze(0))[0, :, :][-1].cpu().numpy().reshape(32,32)
        # feature  = (features.squeeze(0))[6, :, :][-1].cpu().numpy().reshape(16,16)
        feature  = (features.squeeze(0))[0, :, :][32].cpu().numpy().reshape(16,16)
        # feature  = (features.squeeze(0))[3, :, :][-1].cpu().numpy().reshape(16,16)



        # 加载原始图像
        img = cv2.imread("/home/wzh/dataset/ISIC_vis/img/ISIC_0000047.png")

        # 假设feature是已有的(32,32)形状numpy数组
        # 数据预处理
        feature_normalized = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + 1e-8)
        feature_normalized = (feature_normalized * 255).astype(np.uint8)

        # 上采样到256x256
        heatmap = cv2.resize(feature_normalized, (256, 256), interpolation=cv2.INTER_LINEAR)

        # 应用JET颜色映射
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 图像叠加（调整alpha值控制透明度）
        alpha = 0.5  # 热力图透明度，可根据需要调整
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

        # 保存结果
        # cv2.imwrite("/home/wzh/code/DTMFormerv2/visual_final/ISIC/attns/heatmap_setr.png", overlay)
        # cv2.imwrite("/home/wzh/code/DTMFormerv2/visual_final/ISIC/attns/heatmap_fat.png", overlay)
        cv2.imwrite("/home/wzh/code/DTMFormerv2/visual_final/ISIC/attns/heatmap_fat_mtm.png", overlay)






def visual_segmentation_ACDC(seg, image_filename, opt, name_dice, result_path):
    name_dice = name_dice / (opt.classes - 1)

    img_ori = cv2.imread(os.path.join(opt.dataset + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.dataset + '/img', image_filename))
    # img_ori = cv2.imread(os.path.join(opt.dataset + '/img', image_filename))
    # img_ori0 = cv2.imread(os.path.join(opt.dataset + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(opt.imgsize, opt.imgsize))
    img_ori0 = cv2.resize(img_ori0, dsize=(opt.imgsize, opt.imgsize))

    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                      [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135],
                      [237, 145, 33], [176, 224, 230], [61, 145, 64], [227, 207, 87], [189, 252, 201], [245, 222, 179]])
    seg0 = seg[0, :, :]

    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
        # img_r[seg0 == i] = table[i + 1 - 1, 0]
        # img_g[seg0 == i] = table[i + 1 - 1, 1]
        # img_b[seg0 == i] = table[i + 1 - 1, 2]

    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0)  # ACDC
    # img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) # ISIC
    # img = np.uint8(0.3 * overlay + 0.7 * img_ori)

    fulldir = result_path + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    new_image_filename = convert_filename_with_dice(image_filename, name_dice)
    cv2.imwrite(fulldir + new_image_filename, img)

def visual_segmentation_ISIC(seg, image_filename, opt, name_dice, result_path):
    name_dice = name_dice / (opt.classes - 1)

    seg0 = seg[0, :, :]
    img = seg0 * 255
    # img = Image.fromarray(img.astype('uint8'))

    fulldir = result_path + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    new_image_filename = convert_filename_with_dice(image_filename, name_dice)
    cv2.imwrite(fulldir + new_image_filename, img)

def convert_filename_with_dice(image_filename, name_dice):
    # 分离文件名和扩展名
    base_name, extension = image_filename.rsplit('.', 1)

    # 格式化 dice 值，保留小数点后四位
    formatted_dice = f"{name_dice:.4f}"

    # 构建新的文件名
    new_filename = f"{base_name}_{formatted_dice}.{extension}"

    return new_filename

def plot_introduction_mean_values(attns, args, image_filename):
    """
    对于每个attns[i]形状为(1,h,n1,n2),按照h、n1维度取均值，将结果画成折线图。

    参数:
    - attns: 注意力权重列表
    - args: 包含路径等参数的对象
    - image_filename: 图像文件名
    """
    save_dir = os.path.join(args.visualdirec, 'line_charts')
    os.makedirs(save_dir, exist_ok=True)

    for i, attn in enumerate(attns):
        mean_values = attn.mean(dim=(0, 2)).squeeze().cpu().numpy()  # 对h和n1维度求平均
        plt.figure()
        plt.plot(mean_values)
        plt.title(f'Attention Map {i} Mean Values')
        plt.xlabel('Index')
        plt.ylabel('Mean Value')

        save_path = os.path.join(save_dir, f'{os.path.splitext(image_filename)[0]}_attention_{i}_mean.png')
        plt.savefig(save_path)
        plt.close()

def visualize_introduction_attention_heatmaps(attns, args, image_filename):
    """
    对于每个attns[i]形状为(1,h,n1,n2),做h张(n1*n1)形状的可视化热力图。

    参数:
    - attns: 注意力权重列表
    - args: 包含路径等参数的对象
    - image_filename: 图像文件名
    """
    base_name = os.path.splitext(image_filename)[0]
    save_dir = os.path.join(args.visualdirec, 'heatmaps', base_name)
    os.makedirs(save_dir, exist_ok=True)

    for i, attn in enumerate(attns):
        attn_cpu = attn.squeeze().cpu().numpy()  # 将tensor转换为numpy数组
        for head in range(attn_cpu.shape[0]):  # 遍历每个head
            n1 = min(attn_cpu.shape[1], attn_cpu.shape[2])  # 确保取的是(n1*n1)部分
            heatmap_data = attn_cpu[head, :n1, :n1]

            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=False, fmt=".2f", cmap='viridis')
            plt.title(f'Attention Head {head} of Layer {i}')

            save_path = os.path.join(save_dir, f'{base_name}_attention_layer_{i}_head_{head}.png')
            plt.savefig(save_path)
            plt.close()




















def visual_datas_patch8(scores, merge_data, weight_data, attns, scores_png, merge_png, weight_png, attn_png):
    # Ensure directories exist
    os.makedirs(os.path.dirname(scores_png), exist_ok=True)
    os.makedirs(os.path.dirname(merge_png), exist_ok=True)
    os.makedirs(os.path.dirname(weight_png), exist_ok=True)
    os.makedirs(os.path.dirname(attn_png), exist_ok=True)

    # Resize and save each set of data

    save_score_images(scores, scores_png)
    save_merge_images(merge_data, merge_png)
    save_weight_images(weight_data, weight_png)
    save_attn_images(attns, attn_png)


# 保存score图像
def save_score_images(score, scores_png_path):
    n = len(score)  # 需要可视化的图像数量
    h = int(np.sqrt(score[0].shape[1]))  # 假设每个score[i]的形状是[1, h*h]，h是图像的高度和宽度
    for i in range(n):
        score_resized = score[i].reshape(h, h)
        score_path = os.path.join(scores_png_path, f'score_{i}.png')
        cv2.imwrite(score_path, score_resized)
    print(f"Saved {n} score images to {scores_png_path}")


# 保存merge_data图像
def save_merge_images(merge_data, merge_png_path):
    n = len(merge_data)  # 需要可视化的图像数量
    h = int(np.sqrt(merge_data[0].shape[1]))  # 假设每个merge_data[i]的形状是[1, h*h]，h是图像的高度和宽度
    for i in range(n):
        merge_resized = merge_data[i].reshape(h, h)
        merge_path = os.path.join(merge_png_path, f'merge_{i}.png')
        cv2.imwrite(merge_path, merge_resized)
    print(f"Saved {n} merge images to {merge_png_path}")


# 保存weight_data图像
def save_weight_images(weight_data, weight_png_path):
    n = len(weight_data)  # 需要可视化的图像数量
    h = int(np.sqrt(weight_data[0].shape[1]))  # 假设每个weight_data[i]的形状是[1, h*h, 1]，h是图像的高度和宽度
    for i in range(n):
        weight_resized = weight_data[i].reshape(h, h)
        weight_path = os.path.join(weight_png_path, f'weight_{i}.png')
        cv2.imwrite(weight_path, weight_resized)
    print(f"Saved {n} weight images to {weight_png_path}")


# 保存attns图像
def save_attn_images(attns, attn_png_path):
    m = len(attns)  # attns包含16个元素，每个元素的形状为[1, m, n, n]
    h = attns[0].shape[2]  # 假设attns中的每个n*n图像的大小相同
    for i in range(m):
        attn_data = attns[i]
        attn_imgs = []  # 用于存放多个n*n的图像

        # 将m个n*n的图像拼接在一起
        for j in range(attn_data.shape[1]):  # 对于每一个m
            attn_img = attn_data[0, j, :, :]  # 获取一个n*n的图像
            attn_imgs.append(attn_img)

        # 4. 拼接所有的n*n图像，最大8个
        attn_grid = np.zeros((h * 2, h * 4))  # 创建一个足够大的画布来拼接
        for idx, img in enumerate(attn_imgs):
            row = idx // 4  # 行
            col = idx % 4  # 列
            attn_grid[row * h:(row + 1) * h, col * h:(col + 1) * h] = img  # 填充拼接位置

        # 5. 保存拼接后的attn图像
        attn_path = os.path.join(attn_png_path, f'attn_{i}.png')
        cv2.imwrite(attn_path, attn_grid)
    print(f"Saved {m} attention images to {attn_png_path}")






# DMA vis

def generate_token_color(tokencolor, n_classes, n_nprototypes0, n_nprototypes, n_pprototypes):
    # n_ctoken = tokencolor.shape[1]//n_classes
    n_c0token = n_nprototypes
    down_time = n_nprototypes0 // n_nprototypes

    new_tokencolor = tokencolor * 1  # [B N 3]
    start_gray = 20
    inter_gray = (250 - start_gray) // n_c0token
    for i in range(n_c0token):
        new_tokencolor[:, i, 0] = (start_gray + i * inter_gray) / 255
        new_tokencolor[:, i, 1] = (start_gray + i * inter_gray) / 255
        new_tokencolor[:, i, 2] = (start_gray + i * inter_gray) / 255
    table = torch.tensor(
        [[134, 131, 0], [154, 150, 0], [168, 164, 0], [184, 180, 0], [200, 195, 0], [222, 217, 0], [234, 228, 0],
         [244, 238, 0],
         [176, 134, 0], [210, 160, 0], [234, 178, 0], [254, 194, 0], [255, 209, 63], [255, 222, 117], [255, 234, 167],
         [255, 243, 205],
         [230, 93, 0], [255, 106, 5], [255, 128, 41], [255, 143, 67], [255, 161, 97], [255, 178, 125], [255, 198, 159],
         [255, 216, 189],
         [168, 84, 0], [204, 102, 0], [238, 119, 0], [255, 148, 41], [255, 165, 75], [255, 182, 109], [255, 192, 129],
         [255, 213, 171],
         [176, 0, 0], [204, 0, 0], [230, 0, 0], [255, 0, 0], [255, 91, 91], [255, 147, 147], [255, 179, 179],
         [255, 201, 201],
         [204, 0, 83], [230, 0, 93], [255, 25, 118], [255, 75, 148], [255, 109, 168], [255, 147, 191], [255, 167, 203],
         [255, 197, 220],
         [102, 0, 51], [146, 0, 73], [180, 0, 90], [218, 0, 109], [250, 0, 125], [255, 67, 161], [255, 113, 184],
         [255, 147, 201],
         [155, 37, 116], [186, 44, 139], [210, 66, 162], [219, 103, 180], [227, 137, 197], [233, 165, 210],
         [241, 197, 226], [245, 215, 235]])
    table = table[None, :, :].cuda() / 255
    table = table.type_as(tokencolor)
    table = table.repeat(tokencolor.shape[0], 1, 1)
    down_time2 = n_nprototypes0 // n_pprototypes
    for i in range(tokencolor.shape[1] - n_c0token):
        new_tokencolor[:, i + n_c0token, 0] = table[:, i * down_time2, 2]
        new_tokencolor[:, i + n_c0token, 1] = table[:, i * down_time2, 1]
        new_tokencolor[:, i + n_c0token, 2] = table[:, i * down_time2, 0]
    return new_tokencolor


from models.components.tcformer_parts import map2token, token2map
from utils.imgname import read_img_name
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch.nn.functional as Fn
import math
import torch

def vis_cluster_results(img0, idx2cluster, img_name, token_num, id_layer, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens
    Return:
        vis_img (Tensor[B, 3, H, W]): visualize result.
    """
    # imgpath = read_img_name()
    # img0 = cv2.imread(imgpath, 1)

    img0 = cv2.resize(img0, (256, 256))
    img = F.to_tensor(img0)[None, :, :].cuda()

    color_map = Fn.avg_pool2d(img, kernel_size=8)
    B, C, H, W = color_map.shape

    n_sequence = token_num
    h, w = int(math.sqrt(n_sequence)), int(math.sqrt(n_sequence))
    # n_tokens = prototypes.shape[1]
    n_tokens = token_num
    idx_token = torch.arange(n_tokens)[None, :].repeat(B, 1)  # [b 4096], i.e., [0 1 2 3 4 ... 4095]
    agg_weight = torch.ones(B, n_sequence, 1)  # [B 4096 1]=[1 1 1 .... 1]

    out_dict = {}
    out_dict['x'] = 0
    out_dict['token_num'] = n_tokens
    out_dict['map_size'] = [h, w],
    out_dict['init_grid_size'] = [H, W]
    out_dict['idx_token'] = torch.tensor(idx2cluster).cuda()  # [B N]

    # idx_token = idx2cluster[:, n_tokens:]

    token_color = map2token(color_map, out_dict)  # [B N C]
    # token_color = generate_token_color(token_color, n_classes, n_nprototypes0, n_nprototypes, n_pprototypes)

    tmp_dict = out_dict.copy()
    tmp_dict['map_size'] = [H, W]
    tmp_dict['x'] = token_color
    vis_img = token2map(tmp_dict)

    # token_idx = torch.arange(n_tokens)[None, :, None].float().cuda() / n_tokens
    # tmp_dict['x'] = token_idx
    # idx_map = token2map(tmp_dict)  # [B, 1, H, W]
    idx_map = torch.tensor(idx2cluster, dtype=torch.float32).view(1,1,32,32).cuda()

    vis_img = Fn.interpolate(vis_img, [H * 8, W * 8], mode='nearest')
    idx_map = Fn.interpolate(idx_map, [H * 8, W * 8], mode='nearest')

    # cv2.imwrite("/home/wzh/code/DTMFormerv2/visual_final/ACDC/merges/npy/test_vis_img.png", (vis_img.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    # cv2.imwrite("/home/wzh/code/DTMFormerv2/visual_final/ACDC/merges/npy/test_idx_map.png", (idx_map.detach().squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8))


    kernel = idx_map.new_zeros([4, 1, 3, 3])
    kernel[:, :, 1, 1] = 1
    kernel[0, :, 0, 1] = -1
    kernel[1, :, 2, 1] = -1
    kernel[2, :, 1, 0] = -1
    kernel[3, :, 1, 2] = -1

    for i in range(edge_width):
        edge_map = Fn.conv2d(Fn.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
        edge_map = (edge_map != 0).max(dim=1, keepdim=True)[0]
        idx_map = idx_map * (~edge_map) + torch.rand(idx_map.shape).cuda() * edge_map
        # idx_map = idx_map * (~edge_map) + idx_map * edge_map


    edge_color = torch.tensor(edge_color)[None, :, None, None].cuda()

    # cv2.imwrite("/home/wzh/code/DTMFormerv2/visual_final/ACDC/merges/npy/test_111.png", ((vis_img * (~edge_map)).detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    # cv2.imwrite("/home/wzh/code/DTMFormerv2/visual_final/ACDC/merges/npy/test_222.png", ((edge_color * edge_map).detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


    vis_img = vis_img * (~edge_map) + edge_color * edge_map  # [B 3 H W]


    # print("depth:", id_layer, "min:", torch.min(vis_img), "max:", torch.max(vis_img))
    vis_img = vis_img[0, :, :, :].cpu()
    vis_img = vis_img.permute(1, 2, 0)
    vis_img = np.uint8(vis_img * 255)

    # cv2.imwrite("/home/wzh/code/DTMFormerv2/visual_final/ACDC/merges/npy/test.png", vis_img)



    fulldir = "/home/wzh/code/DTMFormerv2/visual_final/ISIC/merges_new/"
    # fulldir = "/home/wzh/code/DTMFormerv2/visual_final/ACDC/merges/vis/save/"

    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    image_filename = img_name
    cv2.imwrite(fulldir + image_filename + '_' + str(id_layer) + '.png', vis_img)
    return vis_img


def vis_grid_results(x, prototypes, idx2cluster, id_layer, n_classes, n_cprototypes, k_class0,
                     edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens
    Return:
        vis_img (Tensor[B, 3, H, W]): visualize result.
    """
    transf = transforms.ToTensor()
    imgpath = read_img_name()
    img0 = cv2.imread(imgpath, 1)
    img0 = cv2.resize(img0, (256, 256))
    img = F.to_tensor(img0)[None, :, :].cuda()

    vis_img = img
    vis_img[:, :, 0:256:8, :] = 1
    vis_img[:, :, :, 0:256:8] = 1
    vis_img = vis_img[0, :, :, :].cpu()
    vis_img = vis_img.permute(1, 2, 0)

    vis_img = np.uint8(vis_img * 255)

    fulldir = "./visualization/Merge/" + "instanceP8" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    image_filename = imgpath.split('/')[-1]
    image_filename = image_filename[:-4]
    cv2.imwrite(fulldir + image_filename + '_' + str(id_layer) + '.png', vis_img)
    return vis_img
