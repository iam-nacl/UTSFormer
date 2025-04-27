
import os
from PIL import Image
from collections import defaultdict


def count_zero_pixel_labels(root_dir):
    label_stats = {
        'l0': {'zero_count': 0, 'total_count': 0},
        'l1': {'zero_count': 0, 'total_count': 0},
        'l2': {'zero_count': 0, 'total_count': 0},
        'l3': {'zero_count': 0, 'total_count': 0}
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

            # 遍历patient_gt_dir下的每个标签图像
            for label_image_name in os.listdir(patient_gt_dir):
                label_image_path = os.path.join(patient_gt_dir, label_image_name)

                # 提取标签类型（l0, l1, l2, l3）
                label_type = label_image_name.split('_')[-1][:-4]  # 去掉'.png'后缀

                if label_type in label_stats:
                    with Image.open(label_image_path) as img:
                        pixel_sum = sum(img.getdata())

                        label_stats[label_type]['total_count'] += 1

                        if pixel_sum == 0:
                            label_stats[label_type]['zero_count'] += 1

    # 计算并打印每种label的像素和为0的数目/总数的对比
    print("Label statistics:")
    for label, stats in label_stats.items():
        zero_count = stats['zero_count']
        total_count = stats['total_count']
        if total_count > 0:
            zero_ratio = zero_count / total_count * 100
        else:
            zero_ratio = 0.0
        print(f"{label}: Zero Pixel Images = {zero_count}, Total Images = {total_count}, Ratio = {zero_ratio:.2f}%")


# 数据集根目录
dataset_root = '/data/wzh/dataset/LIDC-nips2018'

# 统计每种label的像素和情况
count_zero_pixel_labels(dataset_root)



