import os
import ast
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from models import ActionClassifier,TrajectoryModel
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------------- 数据预处理 ----------------------------------

"""
data 格式：
* $data$: dataset of actions，共15个action
* $data[i]$: $i^{th}$ action，有230个关键帧
* $data[i][j]$: $j^{th}$ keyframe of $i^{th}$ action，有21个关键点
* $data[i][j][k]$: the $k^{th}$ key point's information for the $j^{th}$ keyframe of $i^{th}$ action，三维坐标
"""

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

# save and load encoded labels
def save_label_dicts(label_to_encoded, encoded_to_label, save_dir="saved_dicts"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'label_to_encoded.pkl'), 'wb') as f:
        pickle.dump(label_to_encoded, f)
    with open(os.path.join(save_dir, 'encoded_to_label.pkl'), 'wb') as f:
        pickle.dump(encoded_to_label, f)

def load_label_dicts(save_dir="saved_dicts"):
    with open(os.path.join(save_dir, 'label_to_encoded.pkl'), 'rb') as f:
        label_to_encoded = pickle.load(f)
    with open(os.path.join(save_dir, 'encoded_to_label.pkl'), 'rb') as f:
        encoded_to_label = pickle.load(f)
    return label_to_encoded, encoded_to_label

# ---------------------------------- 数据增强 ----------------------------------

# 加噪声
def add_noise(points, sigma=0.01):
    points_np = np.array(points)
    noise = np.random.normal(0, sigma, points_np.shape)
    return points_np + noise

# 放大缩小
def scale(points, scale_factor=None):
    points_np = np.array(points)
    if scale_factor is None:
        scale_factor = np.random.uniform(0.9, 1.1)
    return points_np * scale_factor

# 旋转
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

# 移动
def translate(points, max_translation=0.1):
    points_np = np.array(points)
    
    if points_np.shape[-1] != 3:  # 对非三维数据返回原始数据
        return points_np
    
    dx, dy, dz = np.random.uniform(-max_translation, max_translation, 3)
    return points_np + np.array([dx, dy, dz])

# 增强某个动作
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

# 增强数据集
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

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ---------------------------- 分类器训练 & 测试函数 -----------------------------

def train_classifier(data, labels, classify_model_path):
    # train-val split
    train_data, val_data, train_labels, val_labels = split_data_for_training(data, labels)

    # Convert data to tensor
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Create dataloaders
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize classify_model, criterion, and optimizer
    classify_model = ActionClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classify_model.parameters(), lr=LEARNING_RATE)

    # Train the classify_model
    classify_model = train_model(classify_model, criterion, optimizer, train_loader, val_loader, NUM_EPOCHS, device, EARLY_STOP_PATIENCE)

    # After training: save classify_model
    torch.save(classify_model.state_dict(), classify_model_path)

    return classify_model, keyframe

def test_classifier(data, labels, keyframe, classify_model, device):
    test_data_tensor = torch.tensor(data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(labels, dtype=torch.long)

    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    accuracy = evaluate_model(classify_model, test_loader, device)
    print(f'Accuracy on the test data: {accuracy:.2f}%')

# ---------------------------- 解码器训练 & 测试函数 -----------------------------

# save and load dictionary of models
def save_models(models_dict, save_dir="saved_models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for label, model in models_dict.items():
        # 为了确保文件名不出现非法字符，我们使用repr函数
        safe_label = repr(label)
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{safe_label}.pt"))

def load_models(unique_labels, save_dir="saved_models"):
    models_dict = {}
    for label in unique_labels:
        # 使用之前的方式保持标签安全
        safe_label = repr(label)
        model = TrajectoryModel().to(device)
        model.load_state_dict(torch.load(os.path.join(save_dir, f"model_{safe_label}.pt")))
        models_dict[label] = model
    return models_dict

# generate predicted trajectory of given class and initial point
def generate_trajectory(models_dict, label, start_point, device):
    model = models_dict[label]
    inputs = torch.tensor(start_point, dtype=torch.float32).unsqueeze(0).to(device)
    predicted_trajectory = [start_point]
    
    for _ in range(229):
        output = model(inputs)
        predicted_trajectory.append(output.squeeze().cpu().detach().numpy())
        inputs = output
    
    return np.array(predicted_trajectory).reshape(-1, 21, 3)

def save_models(models_dict, save_dir="saved_models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for label, model in models_dict.items():
        # 为了确保文件名不出现非法字符，我们使用repr函数
        safe_label = repr(label)
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{safe_label}.pt"))

def load_models(unique_labels, save_dir="saved_models"):
    models_dict = {}
    for label in unique_labels:
        # 使用之前的方式保持标签安全
        safe_label = repr(label)
        model = TrajectoryModel().to(device)
        model.load_state_dict(torch.load(os.path.join(save_dir, f"model_{safe_label}.pt")))
        models_dict[label] = model
    return models_dict

def decoder(unique_labels, device, data_transformed):
    models_dict = {label: TrajectoryModel().to(device) for label in unique_labels}
    optimizers_dict = {label: optim.Adam(models_dict[label].parameters(), lr=0.001) for label in unique_labels}
    criterion = nn.MSELoss().to(device)

    for epoch in range(100):
        for label in unique_labels:
            loss_list = []  # 初始化每个label的loss列表
            for i, data_label in enumerate(encoded_labels):
                if data_label == label:
                    model = models_dict[label]
                    optimizer = optimizers_dict[label]
                    
                    inputs = torch.tensor(data_transformed[i][:-1], dtype=torch.float32).unsqueeze(0).to(device)
                    targets = torch.tensor(data_transformed[i][1:], dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    loss_list.append(loss.item())

            avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
            print(f"Epoch {epoch + 1} - Label {label}, Avg Loss: {avg_loss:.4f}")

    save_models(models_dict)
    return models_dict

# ---------------------------- 主函数 & 运行模型代码 -----------------------------

if __name__ == "__main__":
    # Constants
    INPUT_DIM = 21 * 3  # Assuming 17 keypoints and 2D coordinates
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    OUTPUT_DIM = None
    EARLY_STOP_PATIENCE = 10
    LEARNING_RATE = 0.001
    
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load and preprocess data
    train_directory = 'excel'
    train_data, labels, keyframe = preprocess_data(train_directory)
    # use augmented train_data
    train_data, labels = augment_data_and_labels(train_data, labels, times=15)
    # training data for decoder. 230 is # key frame, 63 = # keypoint * 3 (3-dimensional information)
    data_transformed = np.array(train_data).reshape(len(train_data), 230, 63)

    # Encoding labels and save mapping directory
    unique_labels = set(labels)
    label_to_encoded = {label: i for i, label in enumerate(unique_labels)}
    encoded_to_label = {i: label for label, i in label_to_encoded.items()}
    OUTPUT_DIM = len(label_to_encoded)  # Number of classes
    print(label_to_encoded)
    encoded_labels = [label_to_encoded[l] for l in labels]
    save_label_dicts(label_to_encoded, encoded_to_label)
    # loaded_label_to_encoded, loaded_encoded_to_label = load_label_dicts()

    classify_model_path = "model_checkpoint.pth"
    classify_model, keyframe = train_classifier(train_data, encoded_labels, classify_model_path)

    # Test the classify_model
    test_directory = 'data_test'
    test_data, test_labels = preprocess_data_test(test_directory, keyframe)
    encoded_test_labels = [label_to_encoded[label] for label in test_labels]

    test_classifier(test_data, encoded_test_labels, keyframe, classify_model, device)

    # train encoder
    models_dict = decoder(unique_labels, device, data_transformed)
    
    # save model
    # loaded_models_dict = load_models(unique_labels)

    # predict trajectory
    trajectory_label = 0  # 根据需要更改为其他标签
    trajectory = generate_trajectory(models_dict, encoded_to_label[trajectory_label], data_transformed[0][0], device)
    print(trajectory)