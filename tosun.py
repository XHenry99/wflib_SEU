import os
import numpy as np
from scapy.all import rdpcap
from sklearn.preprocessing import OneHotEncoder

def extract_features_and_labels(pcap_dir):
    all_features = []
    all_labels = []
    label_mapping = {}
    label_index = 0

    # 遍历pcap目录下的所有pcap文件
    for root, dirs, files in os.walk(pcap_dir):
        for file in files:
            if file.endswith('.pcap'):
                pcap_path = os.path.join(root, file)
                try:
                    # 假设文件名的一部分作为标签，例如文件名中包含网站名称
                    label = file.split('_')[0]  # 根据实际情况修改
                    if label not in label_mapping:
                        label_mapping[label] = label_index
                        label_index += 1
                    label_id = label_mapping[label]

                    # 读取pcap文件
                    packets = rdpcap(pcap_path)
                    for packet in packets:
                        # 提取特征，这里简单以数据包长度和时间戳为例
                        if 'IP' in packet:
                            length = packet['IP'].len
                            timestamp = packet.time
                            features = [length, timestamp]
                            all_features.append(features)
                            all_labels.append(label_id)
                except Exception as e:
                    print(f"Error processing {pcap_path}: {e}")

    # 将特征和标签转换为numpy数组
    X = np.array(all_features)
    y = np.array(all_labels)

    return X, y

def save_to_npz(X, y, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'processed_data.npz')
    np.savez(output_path, X=X, y=y)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    pcap_dir = r"E:\Henry\工作\暗网溯源\pcap\pcap"
    output_dir = r"E:\Henry\工作\暗网溯源\早期流代码\new"

    # 提取特征和标签
    X, y = extract_features_and_labels(pcap_dir)

    # 保存为npz文件
    save_to_npz(X, y, output_dir)