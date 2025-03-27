import numpy as np
import os
from tqdm import tqdm

# 输入和输出目录
input_dir = r"E:\Henry\工作\暗网溯源\早期流代码\时间分布分析"
output_dir = r"E:\Henry\工作\暗网溯源\早期流代码\早期流量生成"

# 假设丢包比例范围，这里简单示例，你可以根据实际情况修改
# 这里假设每个样本的丢包比例是 10% - 30%
packet_loss_ranges = {}

def process_augmented_data(data):
    """
    处理增强后的数据，模拟网络丢包情况
    :param data: 包含 'X' 和 'y' 的字典
    :return: 处理后的数据字典
    """
    X = data["X"]
    y = data["y"]

    new_X = []
    new_y = []

    for index in tqdm(range(X.shape[0])):
        cur_X = X[index]
        cur_web = y[index]
        # 随机选择丢包比例
        loss_ratio = np.random.uniform(0.1, 0.3)
        # 计算要丢弃的数据包数量
        num_packets_to_drop = int(len(cur_X) * loss_ratio)
        # 随机选择要丢弃的数据包索引
        indices_to_drop = np.random.choice(len(cur_X), num_packets_to_drop, replace=False)
        # 丢弃数据包
        new_cur_X = np.delete(cur_X, indices_to_drop)
        # 补齐长度
        new_cur_X = np.pad(new_cur_X, (0, len(cur_X) - len(new_cur_X)), "constant", constant_values=(0, 0))

        new_X.append(new_cur_X)
        new_y.append(cur_web)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    return {"X": new_X, "y": new_y}

def process_all_augmented_npzs():
    """
    处理指定目录下的所有增强后的 npz 文件
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_file = os.path.join(root, file)
                # 生成对应的输出文件路径
                relative_path = os.path.relpath(npz_file, input_dir)
                out_file = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '_processed.npz')
                out_dir = os.path.dirname(out_file)
                # 创建输出文件夹
                os.makedirs(out_dir, exist_ok=True)
                # 加载增强后的数据
                data = np.load(npz_file, allow_pickle=True)
                # 处理增强后的数据
                processed_data = process_augmented_data(data)
                # 保存处理后的数据到指定文件
                np.savez(out_file, X=processed_data["X"], y=processed_data["y"])
                print(f"生成 {out_file} 完成。")

if __name__ == "__main__":
    # 调用处理所有增强后 npz 文件的函数
    process_all_augmented_npzs()