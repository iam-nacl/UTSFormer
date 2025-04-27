import os
from PIL import Image
from collections import defaultdict


def calculate_consistency_score(root_dir):
    label_stats = {
        'l0': {'consistent_count': 0, 'total_count': 0},
        'l1': {'consistent_count': 0, 'total_count': 0},
        'l2': {'consistent_count': 0, 'total_count': 0},
        'l3': {'consistent_count': 0, 'total_count': 0}
    }

    # 遍历root_dir下的每个子目录（train, val, test）
    for dataset in ['train', 'val', 'test']:
        gt_dir = os.path.join(root_dir, dataset, dataset, 'gt')

        if not os.path.exists(gt_dir):
            print(f"Warning: Directory {gt_dir} does not exist.")
            continue

        # 遍历gt_dir下的每个病人文件夹
        for patient_id in os.listdir(gt_dir):
            patient_gt_dir = os.path.join(gt_dir, patient_id)

            if not os.path.isdir(patient_gt_dir):
                continue

            # 使用字典存储每组图片及其像素和是否为零
            groups = defaultdict(lambda: {'l0': False, 'l1': False, 'l2': False, 'l3': False})

            # 遍历patient_gt_dir下的每个标签图像
            for label_image_name in os.listdir(patient_gt_dir):
                label_image_path = os.path.join(patient_gt_dir, label_image_name)

                # 提取前缀作为组标识
                prefix = '_'.join(label_image_name.split('_')[:-1])
                label_type = label_image_name.split('_')[-1][:-4]  # 去掉'.png'后缀

                with Image.open(label_image_path) as img:
                    pixel_sum = sum(img.getdata())

                    # 记录该组的像素和信息
                    groups[prefix][label_type] = (pixel_sum == 0)

            # 统计每组的情况
            for group_images in groups.values():
                zero_counts = sum(group_images.values())

                if zero_counts == 4:
                    for label in ['l0', 'l1', 'l2', 'l3']:
                        label_stats[label]['consistent_count'] += 1
                        label_stats[label]['total_count'] += 1
                elif zero_counts == 3:
                    for label, is_zero in group_images.items():
                        if is_zero:
                            label_stats[label]['consistent_count'] += 1
                            label_stats[label]['total_count'] += 1
                        else:
                            label_stats[label]['total_count'] += 1
                elif zero_counts == 2:
                    for label, is_zero in group_images.items():
                        if is_zero:
                            label_stats[label]['consistent_count'] += 1
                            label_stats[label]['total_count'] += 1
                        else:
                            label_stats[label]['total_count'] += 1
                elif zero_counts == 1:
                    for label, is_zero in group_images.items():
                        if is_zero:
                            label_stats[label]['consistent_count'] += 1
                            label_stats[label]['total_count'] += 1
                        else:
                            label_stats[label]['total_count'] += 1
                else:
                    for label in ['l0', 'l1', 'l2', 'l3']:
                        label_stats[label]['total_count'] += 1

    # 计算并打印每种label的一致性分数
    print("Label consistency statistics:")
    for label, stats in label_stats.items():
        consistent_count = stats['consistent_count']
        total_count = stats['total_count']
        if total_count > 0:
            consistency_score = consistent_count / total_count * 100
        else:
            consistency_score = 0.0
        print(
            f"{label}: Consistent Images = {consistent_count}, Total Images = {total_count}, Consistency Score = {consistency_score:.2f}%")


# 数据集根目录
dataset_root = '/data/wzh/dataset/LIDC-nips2018'

# 统计每种label的一致性分数
calculate_consistency_score(dataset_root)



