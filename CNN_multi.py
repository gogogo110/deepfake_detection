from __future__ import print_function, division
import cv2
import os
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import time as t

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import datasets
from torchvision import transforms, utils

# 数据集加载
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net1_32(nn.Module):
    def __init__(self):
        super(Net1_32, self).__init__()  # 输入为 12*100
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.fc1 = nn.Linear(30720, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class Net1_64(nn.Module):
    def __init__(self):
        super(Net1_64, self).__init__()  # 输入为 12*100
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.fc1 = nn.Linear(30720, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class Net1_128(nn.Module):
    def __init__(self):
        super(Net1_128, self).__init__()  # 输入为 12*100
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.fc1 = nn.Linear(61440, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class Net2_64(nn.Module):
    def __init__(self):
        super(Net2_64, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.fc1 = nn.Linear(28672, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class Net2_128(nn.Module):
    def __init__(self):
        super(Net2_128, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 5), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 5), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 5), padding=1)

        self.fc1 = nn.Linear(46592, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class Net2_32(nn.Module):
    def __init__(self):
        super(Net2_32, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.fc1 = nn.Linear(43008, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out


class Net3_64(nn.Module):
    def __init__(self):
        super(Net3_64, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv10 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(1024, 1024, (3, 3), padding=1)

        self.fc1 = nn.Linear(12288, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out
class Net3_32(nn.Module):
    def __init__(self):
        super(Net3_32, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv10 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(512, 1024, (3, 3), padding=1)

        self.fc1 = nn.Linear(12288, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class Net3_128(nn.Module):
    def __init__(self):
        super(Net3_128, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv10 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(1024, 1024, (3, 3), padding=1)

        self.fc1 = nn.Linear(24576, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class Net3_128_22(nn.Module):
    def __init__(self):
        super(Net3_128_22, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv10 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(1024, 1024, (3, 3), padding=1)

        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out
class Net4_64(nn.Module):
    def __init__(self):
        super(Net4_64, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv10 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(1024, 1024, (3, 3), padding=1)

        self.conv13 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv14 = nn.Conv2d(1024, 2048, (3, 3), padding=1)
        self.conv15 = nn.Conv2d(2048, 2048, (3, 3), padding=1)

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv15(F.relu(self.conv14(F.relu(self.conv13(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out
class Net4_32(nn.Module):
    def __init__(self):
        super(Net4_32, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv10 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(1024, 1024, (3, 3), padding=1)

        self.conv13 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv14 = nn.Conv2d(1024, 2048, (3, 3), padding=1)
        self.conv15 = nn.Conv2d(2048, 2048, (3, 3), padding=1)

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv15(F.relu(self.conv14(F.relu(self.conv13(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class Net4_128(nn.Module):
    def __init__(self):
        super(Net4_128, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv7 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv8 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv10 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(1024, 1024, (3, 3), padding=1)

        self.conv13 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv14 = nn.Conv2d(1024, 2048, (3, 3), padding=1)
        self.conv15 = nn.Conv2d(2048, 2048, (3, 3), padding=1)

        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(out1)))))), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv15(F.relu(self.conv14(F.relu(self.conv13(out1)))))), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

class my_vgg19(nn.Module):
    def __init__(self):
        super(my_vgg19, self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            # conv2
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            # conv3
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv4 = nn.Sequential(
            # conv4
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv5 = nn.Sequential(
            # conv5
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.adaptivepooling = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=2, bias=True))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.adaptivepooling(self.conv5(conv4))
        conv5 = conv5.view(conv5.size(0), -1)
        # print(conv5.shape)
        score = self.classifier(conv5)
        return score


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        correct = 0
        data = torch.tensor(data, dtype=torch.float32)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        # print(output)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        # print(target)

        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # if (batch_idx + 1) % 20 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc:{:.2f}'
        #           .format(epoch, batch_idx * len(data), len(train_loader.dataset),
        #                   100. * batch_idx / len(train_loader), loss.item(),
        #                   100. * correct / len(data)))
        total_loss += loss.item()
    return total_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.tensor(data, dtype=torch.float32)
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output)

            test_loss += criterion(output, target).item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
    acc = 100. * correct / len(test_loader.dataset)
    return test_loss, acc


def list_of_groups(list_info, per_list_len):
    list_of_group = zip(*(iter(list_info),) * per_list_len)
    end_list = [list(i) for i in list_of_group]  # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count != 0 else end_list
    return end_list


def list_of_vote(res, l):
    ret = []
    for i in range(int(len(res) / l)):
        temp = res[i * l:(i + 1) * l]
        ret.append(max(temp, key=temp.count))
    return ret


def sum_of_secs(score, sec):
    res = []
    for i in range(1, len(sec)):
        sec[i] = sec[i] + sec[i - 1]

    res.append(np.mean(score[0:sec[0]]))
    for i in range(len(sec) - 1):
        res.append(np.mean(score[sec[i]:sec[i + 1]]))
    return res


def softmax(l):
    l_exp = np.exp(l)
    l_exp_sum = l_exp.sum(axis=1).reshape(-1, 1)
    return l_exp / l_exp_sum


def calc_eer_thre(fpr, tpr, thre):
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    th = interp1d(fpr, thre)(eer)
    return eer, th



loss_train_seq = []
for num in range(10):
    root_train = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake\train'
    root_test = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake\test'

    train_dataset = ImageFolder(root_train, transform=data_transform)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_dataset = ImageFolder(root_test, transform=data_transform)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    test_loader1 = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    net = Net3_128()
    # net = torchvision.models.vgg19(pretrained=True)
    # net = my_vgg19()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    # lambda1 = lambda epoch: 0.98 ** epoch
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    EPOCHS = 50
    loss_train_seq = []
    loss_test_seq = []
    max_acc = 0
    net = net.to(device)

    for epoch in range(1, EPOCHS + 1):
        train_l = train(net, train_loader, optimizer, epoch)
        torch.save(net, r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake\model3_' + str(
            num) + '.pkl')
        # scheduler.step()

    #
    # loss_train_seq.append(train_l)
    #     loss_test_seq.append(test_l)
    #
    # plt.plot(loss_train_seq, 'r')
    # plt.show()

    # CNN网络训练之后的代码
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda")
    net1 = torch.load(
        r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake\model3_' + str(num) + '.pkl')
    net1 = net1.to(device_cpu)
    net1.eval()

    test_res_0 = []
    test_res_1 = []
    with torch.no_grad():
        for data, target in test_loader1:
            data = torch.tensor(data, dtype=torch.float32)
            # data, target = data.to(device), target.to(device)
            # print(target)
            output = net1(data)
            output = softmax(output).numpy().squeeze()
            # print(output)

            # pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            test_res_0.append(output[0])
            test_res_1.append(output[1])

    # print(test_res_0)

    secs_0 = np.load(r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake_fake\test\0\secs.npy')
    secs_1 = np.load(r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake_fake\test\1\secs.npy')
    secs0 = np.concatenate((secs_0, secs_1))
    secs1 = np.concatenate((secs_0, secs_1))
    test_result_0 = sum_of_secs(test_res_0, secs0)
    test_result_1 = sum_of_secs(test_res_1, secs1)
    test_label = [0] * len(secs_0) + [1] * len(secs_1)

    fpr2, tpr2, thresholds = roc_curve(test_label, test_result_1, pos_label=1)
    AUC2=auc(fpr2, tpr2)
    print("AUC:", AUC2)


    pred_label = []
    for xxx in test_result_0:
        if xxx > 0.5:
            pred_label.append(0)
        else:
            pred_label.append(1)
    # for i in range(len(pred_label)):
    #     print(str(int(pred_label[i])) + '\t\t', end='')
    # print()


    p = precision_score(test_label, pred_label, average='binary')
    r = recall_score(test_label, pred_label, average='binary')
    f1score = f1_score(test_label, pred_label, average='binary')
    print('acc:', sum(pred_label) / len(test_label))
    print('precision:', p)
    print('recall:', r)
    print('fiscore:', f1score)


# plt.figure(figsize=(6, 6))
# plt.title('ROC')
# plt.plot(fpr1, tpr1, 'b', label='single-scale AUC = %0.3f' % AUC1)
# plt.plot(fpr2,tpr2, 'r', label='multi-scale AUC = %0.3f' % AUC2)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
