import os
import ast
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ---------------------------------- 数据预处理 ----------------------------------

# functions for data preprocessing
def get_label_from_filename(filename):
    if 'drinkWater' in filename:
        return 'drinkWater'
    elif 'reachOut' in filename:
        return 'reachOut'
    elif 'getPhone' in filename:
        return 'getPhone'
    else:
        return None

def parse_entry(entry):
    return np.array(ast.literal_eval(entry))

# 将字符串转换为np.array并计算差分
def compute_difference(df):
    df = df[0].apply(lambda x: np.array(ast.literal_eval(x)))
    diff = df.diff().dropna()
    return diff

# 基于差分结果，计算每一行的模
def compute_magnitude(diff):
    magnitude = diff.apply(lambda x: np.linalg.norm(x))
    return magnitude

# 根据平均序列长度设置一个阈值，并标记显著性差异
def significant_changes(magnitude, average_length):
    threshold = 1 / average_length
    significant = magnitude > threshold
    return significant

# 根据显著的差异来选择关键帧
def get_keyframes(df, target_length):
    diff = compute_difference(df)
    magnitude = compute_magnitude(diff)
    significant = significant_changes(magnitude, target_length)
    
    return df.iloc[significant.nlargest(target_length).index]

# 加载**训练**数据并进行预处理
def preprocess_data(directory):
    # Preprocess the data
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

    # Read the data and get each file's length
    lengths = []
    for file in file_paths:
        df = pd.read_csv(file, header=None)
        lengths.append(len(df))
    average_length = int(np.mean(lengths))

    processed_data = []
    processed_labels = []

    for file in file_paths:
        df = pd.read_csv(file, header=None)
        current_length = len(df)

        if current_length > average_length:
            df = get_keyframes(df, average_length)  # Assuming get_keyframes is a function you've defined elsewhere
        processed_data.append(df)
        
        # Get labels
        label = os.path.basename(file).split('.')[0]
        label = label.split("_")[-1]
        processed_labels.append(label)


    for idx, df in enumerate(processed_data):
        # Convert df to a nested list
        df_list = df.values.tolist()
        new_df_list = []  # Will contain the filtered rows

        for j in range(len(df_list)):
            delete_row = False  # flag to decide whether to delete the row or not

            for k in range(len(df_list[j])):
                value = df_list[j][k]
                if isinstance(value, str) and value.isdigit():  # if it's a string representation of an integer
                    delete_row = True
                    break  # exit the inner loop early
                else:
                    df_list[j][k] = ast.literal_eval(value)  # conversion as before

            if not delete_row:  # if the flag is still False, keep the row
                new_df_list.append(df_list[j])

        # Pad the data which is shorter than the average length
        while len(new_df_list) < average_length:
            new_df_list.append(new_df_list[-1])

        processed_data[idx] = new_df_list

    return processed_data, processed_labels, average_length

# 加载**测试**数据并进行预处理
def preprocess_data_test(directory, keyframe):
    # Preprocess the data
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

    # Read the data and get each file's length
    for file in file_paths:
        df = pd.read_csv(file, header=None)

    processed_data = []
    processed_labels = []

    for file in file_paths:
        df = pd.read_csv(file, header=None)
        current_length = len(df)

        if current_length > keyframe:
            df = get_keyframes(df, keyframe)  # Assuming get_keyframes is a function you've defined elsewhere
        processed_data.append(df)
        
        # Get labels
        label = os.path.basename(file).split('.')[0]
        label = label.split("_")[-1]
        processed_labels.append(label)


    for idx, df in enumerate(processed_data):
        # Convert df to a nested list
        df_list = df.values.tolist()
        new_df_list = []  # Will contain the filtered rows

        for j in range(len(df_list)):
            delete_row = False  # flag to decide whether to delete the row or not

            for k in range(len(df_list[j])):
                value = df_list[j][k]
                if isinstance(value, str) and value.isdigit():  # if it's a string representation of an integer
                    delete_row = True
                    break  # exit the inner loop early
                else:
                    df_list[j][k] = ast.literal_eval(value)  # conversion as before

            if not delete_row:  # if the flag is still False, keep the row
                new_df_list.append(df_list[j])

        # Pad the data which is shorter than the average length
        while len(new_df_list) < keyframe:
            new_df_list.append(new_df_list[-1])

        processed_data[idx] = new_df_list


    return processed_data, processed_labels

# ---------------------------------- 数据增强 ----------------------------------

# functions for data augmentation
def add_noise(points, sigma=0.01):
    points_np = np.array(points)
    noise = np.random.normal(0, sigma, points_np.shape)
    return points_np + noise

def scale(points, scale_factor=None):
    points_np = np.array(points)
    if scale_factor is None:
        scale_factor = np.random.uniform(0.9, 1.1)
    return points_np * scale_factor

def rotate(points, degree_range=10):
    points_np = np.array(points)
    
    if points_np.shape[-1] != 3:  # 只对三维数据执行旋转操作
        return points_np
    
    angle_x = np.radians(np.random.uniform(-degree_range, degree_range))
    angle_y = np.radians(np.random.uniform(-degree_range, degree_range))
    angle_z = np.radians(np.random.uniform(-degree_range, degree_range))
    
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    rotation_matrix_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    rotation_matrix_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
    return np.dot(points_np, rotation_matrix.T)

def translate(points, max_translation=0.1):
    points_np = np.array(points)
    
    if points_np.shape[-1] != 3:  # 对非三维数据返回原始数据
        return points_np
    
    dx, dy, dz = np.random.uniform(-max_translation, max_translation, 3)
    return points_np + np.array([dx, dy, dz])

def augment_single_action(action, times=5):
    """
    对单一动作数据进行多次增强。
    
    参数:
    - action: 原始的动作数据
    - times: 增强的次数
    
    返回值:
    - 一个增强后的动作数据列表
    """
    augmented_actions = [action]  # 包括原始数据
    
    for _ in range(times):
        augmented_action = []
        for keyframe in action:
            keyframe = add_noise(keyframe)
            keyframe = scale(keyframe)
            keyframe = rotate(keyframe)
            keyframe = translate(keyframe)
            augmented_action.append(keyframe)
        augmented_actions.append(augmented_action)
    
    return augmented_actions

def augment_data_and_labels(data, labels, times=5):
    """
    对整个数据集和标签进行多次增强。
    
    参数:
    - data: 原始的动作数据列表
    - labels: 对应的标签列表
    - times: 每个动作增强的次数
    
    返回值:
    - 增强后的数据和标签列表
    """
    augmented_data = []
    augmented_labels = []

    for action, label in zip(data, labels):
        new_actions = augment_single_action(action, times)
        augmented_data.extend(new_actions)
        augmented_labels.extend([label] * len(new_actions))

    return augmented_data, augmented_labels


# ---------------------------- 模型定义及训练工具函数 ----------------------------

# 定义LSTM模型：LSTM架构的浅层RNN
class ActionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ActionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 检测输入数据的结构是否符合预设
def check_shape(data, desired_shape):
    # 每个动作的关键帧长度是否相同
    for i, action in enumerate(data):
        if len(action) != desired_shape[0]:
            print(f"Action at index {i} has {len(action)} keyframes instead of {desired_shape[0]}.")

        # 每个关键帧的关键点数量是否相同
        for j, keyframe in enumerate(action):
            if len(keyframe) != desired_shape[1]:
                print(f"Keyframe {j} in action at index {i} has {len(keyframe)} keypoints instead of {desired_shape[1]}.")

            # 每个关键点的数据输入是否是三维的
            for k, keypoint in enumerate(keyframe):
                        try:
                            if len(keypoint) != desired_shape[2]:
                                print(f"Keypoint {k} in keyframe {j} of action at index {i} has a shape of {len(keypoint)} instead of {desired_shape[2]}.")
                        except:
                            print(f"Keypoint {k} in keyframe {j} of action at index {i} is {keyframe} instead of list of length {desired_shape[2]}.")


# ----------------------------- 模型训练与评估函数 ------------------------------

# 检测错误分类的结果
def get_wrongly_classified_info(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    wrong_indices = (predicted != labels).nonzero(as_tuple=True)[0]
    wrong_predictions = predicted[wrong_indices]
    return wrong_indices.tolist(), wrong_predictions.tolist()

# 分割训练集和测试集
def split_data_for_training(data, labels, test_size=0.1):
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=test_size)
    return train_data, val_data, train_labels, val_labels

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, early_stop_patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{num_epochs - 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == early_stop_patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    model.load_state_dict(best_model)
    return model



# body part code
# 1. data preprocessing
train_directory = "excel"
data, labels, keyframe = preprocess_data(train_directory)

# data augmentation
augmented_data, augmented_labels = augment_data_and_labels(data, labels, times=5)

# ckeck shape of input training
desired_shape = (keyframe, 21, 3)
check_shape(data,desired_shape)

# 数据处理：转换为 [batch, seq_len, input_size] 的格式
data = [[[coord for keypoint in frame for coord in keypoint] for frame in action] for action in data]

# 创建label到整数的映射
label_to_int = {label: idx for idx, label in enumerate(set(labels))}
int_to_label = {idx: label for label, idx in label_to_int.items()}

# 打印编码情况
print(label_to_int)

# 将字符串标签编码为整数
encoded_labels = [label_to_int[label] for label in labels]

# 将嵌套的列表结构转换为torch tensor
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)

dataset = TensorDataset(data_tensor, labels_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# 2. 定义LSTM模型
input_dim = 63  # 展平后的关键点维度
hidden_dim = 128
output_dim = len(label_to_int)
num_layers = 2

model = ActionClassifier(input_dim, hidden_dim, output_dim, num_layers)


# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 4. 训练模型，并记录训练误差, 同时也记录错分类的数据索引和预测值
num_epochs = 10
train_errors = []

wrongly_classified_train_info = []

for epoch in range(num_epochs):
    epoch_error = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_error += loss.item()

        # Collect wrongly classified information
        wrong_indices, wrong_predictions = get_wrongly_classified_info(outputs, labels)
        wrongly_classified_train_info.extend(zip(wrong_indices, wrong_predictions))

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_error /= len(train_loader)
    train_errors.append(epoch_error)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_error:.4f}')

# 打印错分类的训练数据信息
print("Error Information for Training samples:")
for idx, prediction in wrongly_classified_train_info:
    print(f"Index: {idx}, Original Label is: {labels[idx]} Predicted Label: {int_to_label[prediction]}")

with open('train_errors.txt', 'w') as f:
    for error in train_errors:
        f.write(f"{error}\n")
print("\n\n")


# 5. 测试函数
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# 6. 为新的CSV文件测试模型
test_directory = 'data_test'
test_data, test_labels = preprocess_data_test(test_directory,keyframe)
encoded_test_labels = [label_to_int[label] for label in test_labels]
test_data = [[[coord for keypoint in frame for coord in keypoint] for frame in action] for action in test_data]

test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_labels_tensor = torch.tensor(encoded_test_labels, dtype=torch.long)

test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

accuracy = evaluate_model(model, test_loader)

# Collect wrongly classified test data information
wrongly_classified_test_info = []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        wrong_indices, wrong_predictions = get_wrongly_classified_info(outputs, labels)
        wrongly_classified_test_info.extend(zip(wrong_indices, wrong_predictions))

# 打印错分类的测试数据信息
print("Error Information for Testing Samples:")
for idx, prediction in wrongly_classified_train_info:
    print(f"Index: {idx}, Original Label is: {test_labels[idx]} Predicted Label: {int_to_label[prediction]}")

for idx, prediction in wrongly_classified_test_info:
    print(f"Index: {idx}, Predicted Label: {int_to_label[prediction]}")

print(f'Accuracy on the test data: {accuracy:.2f}%')