import os
from PIL import Image
from collections import defaultdict


def count_zero_pixel_groups(root_dir):
    group_stats = {
        'all_zero': 0,
        'three_zero': 0,
        'two_zero': 0,
        'one_zero': 0,
        'none_zero': 0
    }

    # 遍历root_dir下的每个子目录（train, val, test）
    for dataset in ['train', 'val', 'test']:
        gt_dir = os.path.join(root_dir, dataset, dataset, 'gt')
        print("Processing {} ...".format(dataset))
        if not os.path.exists(gt_dir):
            print(f"Warning: Directory {gt_dir} does not exist.")
            continue

        # 遍历gt_dir下的每个病人文件夹
        for patient_id in os.listdir(gt_dir):
            patient_gt_dir = os.path.join(gt_dir, patient_id)

            if not os.path.isdir(patient_gt_dir):
                continue

            # 使用字典存储每组图片及其像素和是否为零
            groups = defaultdict(list)

            # 遍历patient_gt_dir下的每个标签图像
            for label_image_name in os.listdir(patient_gt_dir):
                label_image_path = os.path.join(patient_gt_dir, label_image_name)

                # 提取前缀作为组标识
                prefix = '_'.join(label_image_name.split('_')[:-1])

                with Image.open(label_image_path) as img:
                    pixel_sum = sum(img.getdata())

                    # 记录该组的像素和信息
                    groups[prefix].append(pixel_sum == 0)

            # 统计每组的情况
            for group_images in groups.values():
                zero_count = sum(group_images)

                if zero_count == 4:
                    if group_stats['all_zero'] == 0:
                        print(f"Warning: Found all_zero group in {patient_gt_dir}")
                    group_stats['all_zero'] += 1
                elif zero_count == 3:
                    if group_stats['three_zero'] == 0:
                        print(f"Warning: Found three_zero group in {patient_gt_dir}")

                    group_stats['three_zero'] += 1
                elif zero_count == 2:
                    if group_stats['two_zero'] == 0:
                        print(f"Warning: Found two_zero group in {patient_gt_dir}")
                    group_stats['two_zero'] += 1
                elif zero_count == 1:
                    if group_stats['one_zero'] == 0:
                        print(f"Warning: Found one_zero group in {patient_gt_dir}")
                    group_stats['one_zero'] += 1
                else:
                    if group_stats['none_zero'] == 0:
                        print(f"Warning: Found none_zero group in {patient_gt_dir}")
                    group_stats['none_zero'] += 1

    return group_stats


# 数据集根目录
dataset_root = '/data/wzh/dataset/LIDC-nips2018'

# 统计组内像素和情况
group_stats = count_zero_pixel_groups(dataset_root)

print("Group statistics:")
for key, value in group_stats.items():
    print(f"{key.replace('_', ' ').capitalize()} Groups: {value}")



