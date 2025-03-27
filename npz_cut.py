import numpy as np
import os
from tqdm import tqdm

# 输入和输出目录
input_dir = r"E:\Henry\工作\暗网溯源\早期流代码\npz_1"
output_dir = r"E:\Henry\工作\暗网溯源\早期流代码\npz_2"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 定义百分比列表
percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def extract_early_traffic(X, percentages):
    """
    提取早期区间的流量数据
    :param X: 输入的流量数据
    :param percentages: 百分比列表
    :return: 处理后的流量数据字典
    """
    # 检查 X 的维度
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)  # 将一维数组转换为二维数组
    feat_length = X.shape[1]
    early_traffic = {}
    abs_X = np.absolute(X)
    for p in percentages:
        cur_X = []
        for idx in tqdm(range(X.shape[0]), desc=f"Processing {p}%"):
            tmp_X = abs_X[idx]
            loading_time = tmp_X.max()
            threshold = loading_time * p / 100
            tmp_X = tmp_X[tmp_X > 0]
            tmp_X = tmp_X[tmp_X <= threshold]
            tmp_size = tmp_X.shape[0]
            cur_X.append(np.pad(X[idx][:tmp_size], (0, feat_length - tmp_size), "constant", constant_values=(0, 0)))
        cur_X = np.array(cur_X)
        early_traffic[p] = cur_X
    return early_traffic

# 遍历输入目录中的所有npz文件
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.npz'):
            npz_file = os.path.join(root, file)
            relative_path = os.path.relpath(npz_file, input_dir)
            # 加载npz文件，设置allow_pickle=True
            data = np.load(npz_file, allow_pickle=True)
            X = data["X"]
            y = data["y"]
            # 提取早期区间的流量数据
            early_traffic = extract_early_traffic(X, percentages)
            for p in percentages:
                cur_X = early_traffic[p]
                output_sub_dir = os.path.join(output_dir, os.path.splitext(relative_path)[0])
                os.makedirs(output_sub_dir, exist_ok=True)
                output_file = os.path.join(output_sub_dir, f"test_p{p}.npz")
                if not os.path.exists(output_file):
                    np.savez(output_file, X=cur_X, y=y)
                    print(f"Generated {output_file}")