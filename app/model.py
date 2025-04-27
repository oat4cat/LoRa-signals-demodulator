import numpy as np
import os
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
import copy
class CNN2D(nn.Module):
    def __init__(self, output_size):
        super(CNN2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(32 * 4 * 4, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
