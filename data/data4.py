import os
from PIL import Image
from collections import defaultdict


def process_dataset(root_dir, output_dir):
    # 定义输出路径
    label_dirs = {label: os.path.join(output_dir, label, 'label') for label in ['l0', 'l1', 'l2', 'l3']}
    img_dirs = {label: os.path.join(output_dir, label, 'img') for label in ['l0', 'l1', 'l2', 'l3']}
    main_patient_dirs = {label: os.path.join(output_dir, label, 'MainPatient') for label in ['l0', 'l1', 'l2', 'l3']}

    # 创建必要的目录
    for label in ['l0', 'l1', 'l2', 'l3']:
        os.makedirs(label_dirs[label], exist_ok=True)
        os.makedirs(img_dirs[label], exist_ok=True)
        os.makedirs(main_patient_dirs[label], exist_ok=True)

    # 初始化文件对象
    all_files = {label: open(os.path.join(main_patient_dirs[label], 'all.txt'), 'w') for label in
                 ['l0', 'l1', 'l2', 'l3']}
    train_files = {label: open(os.path.join(main_patient_dirs[label], 'train_ori.txt'), 'w') for label in
                   ['l0', 'l1', 'l2', 'l3']}
    val_files = {label: open(os.path.join(main_patient_dirs[label], 'val_ori.txt'), 'w') for label in
                 ['l0', 'l1', 'l2', 'l3']}
    test_files = {label: open(os.path.join(main_patient_dirs[label], 'test_ori.txt'), 'w') for label in
                  ['l0', 'l1', 'l2', 'l3']}
    delete_files = {label: open(os.path.join(main_patient_dirs[label], 'delete.txt'), 'w') for label in
                    ['l0', 'l1', 'l2', 'l3']}

    print("Directories and files initialized.")

    # 遍历root_dir下的每个子目录（train, val, test）
    for dataset in ['train', 'val', 'test']:
        gt_dir = os.path.join(root_dir, dataset, dataset, 'gt')
        images_dir = os.path.join(root_dir, dataset, dataset, 'images')

        if not os.path.exists(gt_dir) or not os.path.exists(images_dir):
            print(f"Warning: Directories {gt_dir} or {images_dir} do not exist.")
            continue

        print(f"Processing {dataset} dataset...")

        # 遍历gt_dir下的每个病人文件夹
        patient_ids = os.listdir(gt_dir)
        total_patients = len(patient_ids)
        processed_patients = 0

        for patient_id in patient_ids:
            global_index_l0 = 0
            global_index_l1 = 0
            global_index_l2 = 0
            global_index_l3 = 0
            patient_gt_dir = os.path.join(gt_dir, patient_id)
            patient_images_dir = os.path.join(images_dir, patient_id)

            if not os.path.isdir(patient_gt_dir) or not os.path.isdir(patient_images_dir):
                continue

            # 使用字典存储每组图片及其像素和是否为零
            groups = defaultdict(lambda: {'l0': None, 'l1': None, 'l2': None, 'l3': None})

            # 遍历patient_gt_dir下的每个标签图像
            for label_image_name in os.listdir(patient_gt_dir):
                label_image_path = os.path.join(patient_gt_dir, label_image_name)

                # 提取前缀作为组标识
                prefix = '_'.join(label_image_name.split('_')[:-1])
                label_type = label_image_name.split('_')[-1][:-4]  # 去掉'.png'后缀

                with Image.open(label_image_path) as img:
                    pixel_sum = sum(img.getdata())

                    # 记录该组的像素和信息
                    groups[prefix][label_type] = (img, pixel_sum == 0)

            # 统计每组的情况
            index_map = defaultdict(int)
            for prefix, group_images in groups.items():
                for label, (img, is_zero) in group_images.items():
                    if label == 'l0':
                        global_index = global_index_l0
                    elif label == 'l1':
                        global_index = global_index_l1
                    elif label == 'l2':
                        global_index = global_index_l2
                    elif label == 'l3':
                        global_index = global_index_l3
                    else:
                        print("Warning: unknown label.")

                    if not is_zero:
                        # Resize and save the label image
                        resized_img = img.resize((256, 256))
                        new_label_filename = f"{patient_id.replace('LIDC-IDRI-', '')}{global_index:03d}.png"
                        new_label_filepath = os.path.join(label_dirs[label], new_label_filename)
                        resized_img.save(new_label_filepath)

                        # Find corresponding image in images directory
                        image_prefix = f"{prefix}.png"  # 去掉多余的_c0
                        image_path = os.path.join(patient_images_dir, image_prefix)

                        if os.path.exists(image_path):
                            with Image.open(image_path) as original_img:
                                resized_original_img = original_img.resize((256, 256))
                                new_image_filename = f"{patient_id.replace('LIDC-IDRI-', '')}{global_index:03d}.png"
                                new_image_filepath = os.path.join(img_dirs[label], new_image_filename)
                                resized_original_img.save(new_image_filepath)

                            # Write to all.txt
                            all_files[label].write(f"{new_label_filename[:-4]}\n")

                            # Write to appropriate dataset file
                            if dataset == 'train':
                                train_files[label].write(f"{new_label_filename[:-4]}\n")
                            elif dataset == 'val':
                                val_files[label].write(f"{new_label_filename[:-4]}\n")
                            elif dataset == 'test':
                                test_files[label].write(f"{new_label_filename[:-4]}\n")

                            # Increment global index
                            if label == 'l0':
                                global_index_l0 += 1
                            elif label == 'l1':
                                global_index_l1 += 1
                            elif label == 'l2':
                                global_index_l2 += 1
                            elif label == 'l3':
                                global_index_l3 += 1
                            else:
                                print("Warning: unknown label.")
                    else:
                        # Write to delete.txt with additional information
                        delete_files[label].write(f"{prefix}_{label} from {patient_id} in {dataset}\n")

            processed_patients += 1
            if processed_patients % 10 == 0:
                print(f"Processed {processed_patients}/{total_patients} patients in {dataset} dataset.")

        print(f"Completed processing {dataset} dataset.")

    # 关闭所有文件
    for label in ['l0', 'l1', 'l2', 'l3']:
        all_files[label].close()
        train_files[label].close()
        val_files[label].close()
        test_files[label].close()
        delete_files[label].close()

    print("All datasets processed and files closed.")


# 数据集根目录
dataset_root = '/data/wzh/dataset/LIDC-nips2018'
output_dir = '/data/wzh/dataset/LIDC-IDRI-256'

# 处理数据集
process_dataset(dataset_root, output_dir)



