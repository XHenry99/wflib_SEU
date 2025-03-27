import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import math

class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)

class Encoder2d(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num_layers):
        super().__init__()
        layers = []
        cur_channels = 32
        
        # 初始层（通道转换）
        layers.append(ConvBlock2d(in_channels, cur_channels, (3, 3)))
        layers.append(nn.AdaptiveAvgPool2d((32, 1)))  # 统一输入高度为32
        layers.append(nn.Dropout(0.1))

        # 中间层（高度方向降采样）
        for i in range(conv_num_layers-1):
            layers.append(ConvBlock2d(cur_channels, cur_channels*2, (3, 3)))
            layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))  # 仅降低高度
            layers.append(nn.Dropout(0.1))
            cur_channels *= 2

        # 最终调整层
        layers.append(ConvBlock2d(cur_channels, out_channels, (3, 3)))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # 统一输出尺寸
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size-1)*(dilation-1) - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                     dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)

class Encoder1d(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num_layers):
        super().__init__()
        layers = []
        cur_channels = in_channels
        
        # 添加自适应池化确保足够长度
        layers.append(nn.AdaptiveAvgPool1d(64))
        
        for _ in range(conv_num_layers):
            layers.append(ConvBlock1d(cur_channels, cur_channels*2, 3))
            layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
            layers.append(nn.Dropout(0.3))
            cur_channels *= 2
        
        layers.append(nn.Conv1d(cur_channels, out_channels, 1))
        layers.append(nn.AdaptiveAvgPool1d(1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Holmes(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder2d = Encoder2d(in_channels=3, out_channels=64, conv_num_layers=3)
        self.encoder1d = Encoder1d(in_channels=64, out_channels=128, conv_num_layers=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder2d(x)  # [batch, 64, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [batch, 64]
        x = x.unsqueeze(-1)    # [batch, 64, 1]
        x = self.encoder1d(x)  # [batch, 128, 1]
        return self.classifier(x)

def mad(data):
    median = np.median(data, axis=0)
    return np.median(np.abs(data - median), axis=0)

def calculate_spatial_distribution(embs, labels):
    unique_labels = np.unique(labels)
    centroids = []
    radii = []
    
    for label in unique_labels:
        label_embs = embs[labels == label]
        centroid = np.mean(label_embs, axis=0)
        distances = 1 - cosine_similarity(label_embs, [centroid])
        radii.append(mad(distances))
        centroids.append(centroid)
    
    return np.array(centroids), np.array(radii)

def process_all_npzs(input_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Holmes(num_classes=100).to(device)
    
    # 特征工程参数
    TARGET_FEATURES = 3 * 32  # 3通道 x 32高度
    MIN_FEATURES = 10  # 最小有效特征数

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith('.npz'):
                npz_path = os.path.join(root, file)
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    X = data["X"].astype(np.float32)
                    y = data["y"]
                    
                    # ========== 新增数据预处理 ==========
                    # 检查有效样本
                    if X.shape[0] == 0 or X.ndim != 2:
                        raise ValueError(f"无效数据形状: {X.shape}")
                        
                    # 检查特征维度
                    original_features = X.shape[1]
                    
                    # 特征填充/截断逻辑
                    if original_features < TARGET_FEATURES:
                        # 填充0到目标维度
                        pad_width = TARGET_FEATURES - original_features
                        X = np.pad(X, ((0,0), (0,pad_width)), mode='constant')
                        print(f"已填充 {pad_width} 个特征: {file}")
                    elif original_features > TARGET_FEATURES:
                        # 截断到目标维度
                        X = X[:, :TARGET_FEATURES]
                        print(f"已截断 {original_features-TARGET_FEATURES} 个特征: {file}")
                    
                    # 重塑为4D输入 [batch, channels, height, width]
                    X = X.reshape(-1, 3, 32, 1)  # 3通道 x 32高度 x 1宽度
                    
                    # 跳过无效数据
                    if X.shape[0] == 0:
                        raise ValueError("零样本数据")
                        
                    X_tensor = torch.tensor(X, device=device)
                    
                    with torch.no_grad():
                        embeddings = model(X_tensor).cpu().numpy()
                        
                    # 保存结果
                    output_path = os.path.join(
                        output_dir,
                        os.path.relpath(npz_path, input_dir).replace(".npz", "_spatial.npz")
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    centroids, radii = calculate_spatial_distribution(embeddings, y)
                    np.savez(output_path, centroid=centroids, radius=radii)
                    
                except Exception as e:
                    print(f"处理文件 {file} 出错: {str(e)}")
                    continue

if __name__ == "__main__":
    input_dir = r"E:\Henry\工作\暗网溯源\早期流代码\时间分布分析"
    output_dir = r"E:\Henry\工作\暗网溯源\早期流代码\空间对比分析"
    process_all_npzs(input_dir, output_dir)