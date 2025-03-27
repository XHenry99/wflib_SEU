import scapy.all as scapy
import numpy as np
import os

# 直接指定输入和输出目录
pcap_dir = r"E:\Henry\工作\暗网溯源\pcap\pcap"
output_dir = r"E:\Henry\工作\暗网溯源\早期流代码\npz_1"


def pcap_to_npz(pcap_file, npz_file):
    """
    此函数用于将PCAP文件转换为NPZ文件
    :param pcap_file: 输入的PCAP文件路径
    :param npz_file: 输出的NPZ文件路径
    """
    try:
        # 从PCAP文件路径中提取网站名作为标签
        folder_name = os.path.basename(os.path.dirname(pcap_file))
        website_name = folder_name.split('_')[-1]

        # 读取PCAP文件
        packets = scapy.rdpcap(pcap_file)
        X = []
        y = []
        for packet in packets:
            if packet.haslayer(scapy.IP):
                # 提取IP层的时间戳
                timestamp = packet.time
                # 简单示例方向判断，假设源IP为192.168.1.1为正向
                direction = 1 if packet[scapy.IP].src == '192.168.1.1' else -1
                value = direction * timestamp
                X.append(value)
                y.append(website_name)

        # 将列表转换为NumPy数组
        X = np.array(X)
        y = np.array(y)
        # 保存为NPZ文件
        np.savez(npz_file, X=X, y=y)
        print(f"成功将 {pcap_file} 转换为 {npz_file}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")


def process_all_pcaps():
    """
    此函数用于处理指定目录下的所有PCAP文件
    """
    for root, dirs, files in os.walk(pcap_dir):
        for file in files:
            if file.endswith('.pcap'):
                pcap_file = os.path.join(root, file)
                # 生成对应的NPZ文件路径
                relative_path = os.path.relpath(pcap_file, pcap_dir)
                npz_file = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.npz')
                npz_dir = os.path.dirname(npz_file)
                # 创建输出文件夹
                os.makedirs(npz_dir, exist_ok=True)
                # 调用转换函数
                pcap_to_npz(pcap_file, npz_file)


if __name__ == "__main__":
    # 调用处理所有PCAP文件的函数
    process_all_pcaps()