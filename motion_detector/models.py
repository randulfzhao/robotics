import os
import ast
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# model for classifier
class ActionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ActionClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor 
        x = x.view(x.size(0), x.size(1), -1)
        
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# model for decoder
class TrajectoryModel(nn.Module):
    def __init__(self):
        super(TrajectoryModel, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 63)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
