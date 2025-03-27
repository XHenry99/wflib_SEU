import numpy as np
import torch
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json

class Holmes(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 保持与预处理阶段相同的模型结构
        self.encoder2d = Encoder2d(in_channels=3, out_channels=64, conv_num_layers=3)
        self.encoder1d = Encoder1d(in_channels=64, out_channels=128, conv_num_layers=2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder2d(x)
        x = x.squeeze(-1).squeeze(-1)
        x = x.unsqueeze(-1)
        x = self.encoder1d(x)
        return self.classifier(x)

class Encoder2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_num_layers):
        super().__init__()
        # 保持与预处理阶段相同的编码器结构
        layers = []
        cur_channels = 32
        layers.append(ConvBlock2d(in_channels, cur_channels, (3, 3)))
        layers.append(torch.nn.AdaptiveAvgPool2d((32, 1)))
        layers.append(torch.nn.Dropout(0.1))
        
        for _ in range(conv_num_layers-1):
            layers.append(ConvBlock2d(cur_channels, cur_channels*2, (3, 3)))
            layers.append(torch.nn.MaxPool2d((2, 1), (2, 1)))
            layers.append(torch.nn.Dropout(0.1))
            cur_channels *= 2
        
        layers.append(ConvBlock2d(cur_channels, out_channels, (3, 3)))
        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.last_relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)

class Encoder1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_num_layers):
        super().__init__()
        layers = []
        layers.append(torch.nn.AdaptiveAvgPool1d(64))
        
        cur_channels = in_channels
        for _ in range(conv_num_layers):
            layers.append(ConvBlock1d(cur_channels, cur_channels*2, 3))
            layers.append(torch.nn.MaxPool1d(3, 2, padding=1))
            layers.append(torch.nn.Dropout(0.3))
            cur_channels *= 2
        
        layers.append(torch.nn.Conv1d(cur_channels, out_channels, 1))
        layers.append(torch.nn.AdaptiveAvgPool1d(1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size-1)*(dilation-1) - 1) // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                          dilation=dilation, padding=padding),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size,
                          dilation=dilation, padding=padding),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU()
        )
        self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.last_relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)

def process_identification(input_dir, output_dir, ckp_path, num_classes=100, scenario="Closed-world"):
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = Holmes(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(ckp_path, "holmes_model.pth")))
    model.eval()
    
    # 加载空间分布数据
    spatial_file = os.path.join(ckp_path, "spatial_distribution.npz")
    spatial_data = np.load(spatial_file)
    webs_centroid = spatial_data["centroid"]
    webs_radius = spatial_data["radius"]
    
    # 配置参数
    open_threshold = 1e-2  # 开放世界阈值
    target_features = 3 * 32  # 必须与预处理阶段一致
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith(".npz"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(
                    output_dir,
                    os.path.relpath(input_path, input_dir).replace(".npz", "_result.json")
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    # 加载数据
                    data = np.load(input_path)
                    X = data["X"].astype(np.float32)
                    y = data["y"] if "y" in data else None
                    
                    # 数据预处理
                    current_features = X.shape[1]
                    if current_features < target_features:
                        pad_width = target_features - current_features
                        X = np.pad(X, ((0,0), (0,pad_width)), mode="constant")
                    elif current_features > target_features:
                        X = X[:, :target_features]
                    
                    # 重塑为模型输入形状
                    X = X.reshape(-1, 3, 32, 1)
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    
                    # 特征提取
                    with torch.no_grad():
                        embeddings = model(X_tensor).cpu().numpy()
                    
                    # 网站识别
                    all_sims = 1 - cosine_similarity(embeddings, webs_centroid)
                    all_sims -= webs_radius
                    predictions = np.argmin(all_sims, axis=1)
                    
                    # 概念漂移检测
                    drift_indices = []
                    if scenario == "Open-world":
                        min_distances = np.min(all_sims, axis=1)
                        drift_indices = np.where(min_distances > open_threshold)[0].tolist()
                        predictions[drift_indices] = num_classes - 1  # 标记为未知
                    
                    # 保存结果
                    result = {
                        "file": file,
                        "predictions": predictions.tolist(),
                        "drift_samples": drift_indices
                    }
                    if y is not None:
                        result["true_labels"] = y.tolist()
                    
                    with open(output_path, "w") as f:
                        json.dump(result, f, indent=2)
                        
                except Exception as e:
                    print(f"处理文件 {file} 时发生错误: {str(e)}")
                    continue

if __name__ == "__main__":
    # 配置路径参数
    input_dir = r"E:\Henry\工作\暗网溯源\早期流代码\空间对比分析"
    output_dir = r"E:\Henry\工作\暗网溯源\早期流代码\早期网站识别"
    checkpoint_dir = r"E:\Henry\工作\暗网溯源\模型检查点"
    
    # 执行识别流程
    process_identification(
        input_dir=input_dir,
        output_dir=output_dir,
        ckp_path=checkpoint_dir,
        num_classes=100,
        scenario="Open-world"
    )