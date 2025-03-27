import numpy as np
import torch
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json
import sys
sys.path.append("E:\Henry\工作\暗网溯源\早期流代码\holmes.py")  # 如果文件不在同一目录
from holmes import Holmes  # 确保与重命后的文件名一致

# 模型定义保持不变...

def train_model(train_data_dir, checkpoint_dir, num_classes=100, epochs=50):
    """
    模型训练函数
    :param train_data_dir: 训练数据目录
    :param checkpoint_dir: 检查点保存目录
    :param num_classes: 类别数
    :param epochs: 训练轮数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = Holmes(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 创建数据加载器（需根据实际数据格式调整）
    def load_train_data(data_dir):
        X_list, y_list = [], []
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                for npz_file in os.listdir(class_path):
                    if npz_file.endswith(".npz"):
                        data = np.load(os.path.join(class_path, npz_file))
                        X_list.append(data["X"])
                        y_list.extend([int(class_dir)] * len(data["X"]))
        return np.concatenate(X_list), np.array(y_list)

    # 加载训练数据
    print("正在加载训练数据...")
    X_train, y_train = load_train_data(train_data_dir)
    
    # 数据预处理
    target_features = 3 * 32
    current_features = X_train.shape[1]
    if current_features < target_features:
        pad_width = target_features - current_features
        X_train = np.pad(X_train, ((0,0), (0,pad_width)), mode="constant")
    elif current_features > target_features:
        X_train = X_train[:, :target_features]
    
    X_train = X_train.reshape(-1, 3, 32, 1)
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 训练循环
    print("开始模型训练...")
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存最佳模型
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "holmes_model.pth"))
            print(f"保存最佳模型到 {checkpoint_dir}")

    # 生成空间分布数据
    print("生成空间分布数据...")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            emb = model.encoder1d(model.encoder2d(inputs).squeeze(-1).unsqueeze(-1))
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.concatenate(embeddings).reshape(-1, 128)
    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings)
    
    # 计算每个类的半径
    centroid_radius = []
    for i in range(num_classes):
        cluster_points = embeddings[kmeans.labels_ == i]
        distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
        centroid_radius.append(np.percentile(distances, 95))  # 取95%分位数作为半径
    
    np.savez(os.path.join(checkpoint_dir, "spatial_distribution.npz"),
             centroid=kmeans.cluster_centers_,
             radius=np.array(centroid_radius))

def process_identification(input_dir, output_dir, ckp_path, num_classes=100, scenario="Closed-world"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查模型文件是否存在
    model_path = os.path.join(ckp_path, "holmes_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型或提供正确路径")

    # 加载模型
    model = Holmes(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 检查空间分布文件
    spatial_file = os.path.join(ckp_path, "spatial_distribution.npz")
    if not os.path.exists(spatial_file):
        raise FileNotFoundError(f"空间分布文件 {spatial_file} 不存在")

    spatial_data = np.load(spatial_file)
    webs_centroid = spatial_data["centroid"]
    webs_radius = spatial_data["radius"]

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith(".npz"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path.replace(".npz", "_result.json"))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    data = np.load(input_path)
                    X = data["X"].astype(np.float32)
                    y = data["y"] if "y" in data else None
                    
                    # 数据标准化处理
                    target_features = 3 * 32
                    current_features = X.shape[1]
                    if current_features != target_features:
                        if current_features < target_features:
                            pad_width = target_features - current_features
                            X = np.pad(X, ((0,0), (0,pad_width)), mode="constant")
                        else:
                            X = X[:, :target_features]
                    
                    X = X.reshape(-1, 3, 32, 1)
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        embeddings = model(X_tensor).cpu().numpy()
                    
                    # 相似度计算
                    all_sims = 1 - cosine_similarity(embeddings, webs_centroid)
                    all_sims -= webs_radius
                    predictions = np.argmin(all_sims, axis=1)
                    
                    drift_indices = []
                    if scenario == "Open-world":
                        open_threshold = 1e-2
                        min_distances = np.min(all_sims, axis=1)
                        drift_indices = np.where(min_distances > open_threshold)[0].tolist()
                        predictions[drift_indices] = num_classes - 1
                    
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
    # 配置参数
    config = {
        "train_data_dir": r"E:\Henry\工作\暗网溯源\训练数据",  # 训练数据目录
        "input_dir": r"E:\Henry\工作\暗网溯源\早期流代码\空间对比分析",
        "output_dir": r"E:\Henry\工作\暗网溯源\早期流代码\早期网站识别",
        "checkpoint_dir": r"E:\Henry\工作\暗网溯源\模型检查点",
        "num_classes": 100,
        "scenario": "Open-world",
        "need_train": True  # 设置为False跳过训练
    }

    # 自动训练检测
    if config["need_train"] or not os.path.exists(os.path.join(config["checkpoint_dir"], "holmes_model.pth")):
        print("检测到需要模型训练...")
        train_model(
            train_data_dir=config["train_data_dir"],
            checkpoint_dir=config["checkpoint_dir"],
            num_classes=config["num_classes"],
            epochs=50
        )

    # 执行识别流程
    process_identification(
        input_dir=config["input_dir"],
        output_dir=config["output_dir"],
        ckp_path=config["checkpoint_dir"],
        num_classes=config["num_classes"],
        scenario=config["scenario"]
    )