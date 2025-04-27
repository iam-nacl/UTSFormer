import os
from shutil import copyfile


def merge_train_val_and_copy_test(output_dir):
    # 定义标签列表
    labels = ['l0', 'l1', 'l2', 'l3']

    for label in labels:
        main_patient_dir = os.path.join(output_dir, label, 'MainPatient')

        # 定义文件路径
        train_ori_path = os.path.join(main_patient_dir, 'train_ori.txt')
        val_ori_path = os.path.join(main_patient_dir, 'val_ori.txt')
        test_ori_path = os.path.join(main_patient_dir, 'test_ori.txt')
        train_path = os.path.join(main_patient_dir, 'train.txt')
        test_path = os.path.join(main_patient_dir, 'test.txt')

        # 检查文件是否存在
        if not os.path.exists(train_ori_path):
            print(f"Warning: {train_ori_path} does not exist.")
            continue

        if not os.path.exists(val_ori_path):
            print(f"Warning: {val_ori_path} does not exist.")
            continue

        if not os.path.exists(test_ori_path):
            print(f"Warning: {test_ori_path} does not exist.")
            continue

        # 合并train_ori.txt和val_ori.txt到train.txt
        with open(train_path, 'w') as train_file:
            with open(train_ori_path, 'r') as train_ori_file:
                train_file.write(train_ori_file.read())

            with open(val_ori_path, 'r') as val_ori_file:
                train_file.write(val_ori_file.read())

        print(f"Merged {train_ori_path} and {val_ori_path} into {train_path}.")

        # 复制test_ori.txt为test.txt
        copyfile(test_ori_path, test_path)
        print(f"Copied {test_ori_path} to {test_path}.")


# 输出目录
output_dir = '/data/wzh/dataset/LIDC-IDRI-256'

# 执行合并和复制操作
merge_train_val_and_copy_test(output_dir)
