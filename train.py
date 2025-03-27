import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from WFlib.tools.data_processor import load_data, load_iter
from WFlib.tools.evaluator import measurement

# 定义模型训练和评估函数
def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, eval_metrics):
    model.train()
    for epoch in range(30):  # 假设训练30个epoch
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    results = measurement(np.array(all_targets), np.array(all_preds), eval_metrics)
    return results

# 输入和输出目录
input_dir = r"E:\Henry\工作\暗网溯源\早期流代码\npz_2"
# 假设使用的数据集名称
dataset = "your_dataset_name"
# 假设使用的模型，这里以ARES为例
from WFlib.models.ARES import ARES
model = ARES(num_classes=100)  # 假设类别数为100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
eval_metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

# 定义百分比列表
percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

best_accuracy = 0
best_percentage = None

# 遍历不同的早期区间百分比
for p in percentages:
    # 加载训练和测试数据
    train_file = os.path.join(input_dir, f"your_train_folder/test_p{p}.npz")
    test_file = os.path.join(input_dir, f"your_test_folder/test_p{p}.npz")

    # 加载数据
    X_train, y_train = load_data(train_file, "DIR", seq_len=5000)  # 假设特征类型为DIR，序列长度为5000
    X_test, y_test = load_data(test_file, "DIR", seq_len=5000)

    # 创建数据加载器
    train_loader = load_iter(X_train, y_train, batch_size=256, is_train=True)
    test_loader = load_iter(X_test, y_test, batch_size=256, is_train=False)

    # 训练和评估模型
    results = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, eval_metrics)

    accuracy = results["Accuracy"]
    print(f"Percentage: {p}%, Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_percentage = p

print(f"Best percentage: {best_percentage}%, Best accuracy: {best_accuracy}")