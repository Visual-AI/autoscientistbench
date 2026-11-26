"""CNN used for the controlled MNIST experiments of the FedQueue paper.

Architecture as described in Appendix C (Table 9 / "Model" paragraph):
two 3x3 convolutions with channel sizes (32, 64) and padding 1, each followed
by 2x2 max pooling, a fully connected layer 64*7*7 -> 128, dropout p=0.5, and
a 10-way linear classifier, ReLU activations throughout.

NOTE: this file is an addition (outside the APPFL repository) because the
released ``examples/resources/model/cnn.py`` in APPFL implements a different
CNN (5x5 convolutions, no padding, FC 64*4*4 -> 512).
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.5)
        reduced = num_pixel // 2 // 2  # 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * reduced * reduced, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
