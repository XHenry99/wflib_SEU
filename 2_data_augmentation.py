import numpy as np
import os
from tqdm import tqdm

# 输入和输出目录
input_dir = r"E:\Henry\工作\暗网溯源\早期流代码\npz_1"
output_dir = r"E:\Henry\工作\暗网溯源\早期流代码\时间分布分析"

# 假设有效加载范围，这里简单示例，你可以根据实际情况修改
# 这里假设每个网站的有效加载范围是 30% - 60%
effective_ranges = {}
# 遍历所有 npz 文件，获取所有网站名
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.npz'):
            npz_file = os.path.join(root, file)
            # 修改：设置 allow_pickle=True
            data = np.load(npz_file, allow_pickle=True)
            y = data["y"]
            for website in np.unique(y):
                effective_ranges[website] = (30, 60)

def gen_augment(data, num_aug, effective_ranges, out_file):
    """
    生成增强数据并保存到文件
    :param data: 包含 'X' 和 'y' 的字典
    :param num_aug: 每个原始样本生成的增强样本数量
    :param effective_ranges: 每个类别的有效范围字典
    :param out_file: 输出文件路径
    """
    X = data["X"]
    y = data["y"]

    new_X = []
    new_y = []
    abs_X = np.absolute(X)

    # 检查 X 的维度
    if len(X.shape) == 1:
        # 如果 X 是一维数组，将其转换为二维数组
        X = X.reshape(1, -1)
        abs_X = abs_X.reshape(1, -1)
    elif len(X.shape) == 0:
        # 如果 X 是空数组，直接返回
        print("Warning: X is an empty array. Skipping augmentation.")
        return

    feat_length = X.shape[1]

    # 遍历每个样本
    for index in tqdm(range(abs_X.shape[0])):
        cur_abs_X = abs_X[index]
        # 检查 cur_abs_X 是否为 numpy 数组
        if not isinstance(cur_abs_X, np.ndarray):
            cur_abs_X = np.array(cur_abs_X)
        # 检查 cur_abs_X 是否为空
        if cur_abs_X.size == 0:
            continue
        cur_web = y[index]
        loading_time = cur_abs_X.max()

        # 为每个样本生成增强样本
        for ii in range(num_aug):
            p = np.random.randint(effective_ranges[cur_web][0], effective_ranges[cur_web][1])
            threshold = loading_time * p / 100
            valid_X = cur_abs_X[cur_abs_X > 0]
            valid_X = valid_X[valid_X <= threshold]
            valid_length = valid_X.shape[0]
            new_X.append(np.pad(X[index][:valid_length], (0, feat_length - valid_length), "constant", constant_values=(0, 0)))
            new_y.append(cur_web)

        # 添加原始样本
        new_X.append(X[index])
        new_y.append(cur_web)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    # 保存增强数据到指定文件
    np.savez(out_file, X=new_X, y=new_y)
    print(f"生成 {out_file} 完成。")

def process_all_npzs():
    """
    处理指定目录下的所有 npz 文件
    """
    num_aug = 5  # 每个原始样本生成的增强样本数量
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_file = os.path.join(root, file)
                # 生成对应的输出文件路径
                relative_path = os.path.relpath(npz_file, input_dir)
                out_file = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '_aug.npz')
                out_dir = os.path.dirname(out_file)
                # 创建输出文件夹
                os.makedirs(out_dir, exist_ok=True)
                # 修改：设置 allow_pickle=True
                data = np.load(npz_file, allow_pickle=True)
                # 调用生成增强数据的函数
                gen_augment(data, num_aug, effective_ranges, out_file)

if __name__ == "__main__":
    # 调用处理所有 npz 文件的函数
    process_all_npzs()