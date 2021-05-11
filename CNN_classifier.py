from __future__ import print_function, division
import cv2
import os
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import time as t
import math

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
import torch

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision import transforms, utils

# 数据集加载
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()  # 输入为 12*100
        # self.conv1 = nn.Conv2d(1, 10, (7, 21))
        self.fc1 = nn.Linear(2200, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        # out1 = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        out1 = x.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()  # 输入为 12*100
        self.conv1 = nn.Conv2d(1, 10, (7, 21))
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.fc2(out)
        return out


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()  # 输入为 12*100
        self.conv1 = nn.Conv2d(1, 10, (7, 21))
        self.conv2 = nn.Conv2d(10, 20, (7, 21))
        self.fc1 = nn.Linear(200, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv2(out1)), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.fc1(out1)
        return out


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()  # 输入为 12*100
        self.conv1 = nn.Conv2d(1, 10, (7, 21))
        self.conv2 = nn.Conv2d(10, 20, (5, 15))
        self.fc1 = nn.Linear(520, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.size(0)
        out1 = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        out1 = F.max_pool2d(F.relu(self.conv2(out1)), 2, 2)
        out1 = out1.view(in_size, -1)
        # print(out1.shape)
        out = self.dropout1(F.relu(self.fc1(out1)))
        out = self.fc2(out)
        return out

class VGG19(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # # Block 4
            # nn.Conv2d(256, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2),
            # # Block 5
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256*7*7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        # 因为前面可以用预训练模型参数，所以单独把最后一层提取出来
        self.classifier2 = nn.Linear(2048, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        # torch.flatten 推平操作
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        correct = 0
        data = torch.tensor(data, dtype=torch.float32)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标

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
    '''
    :param list_info:   列表
     :param per_list_len:  每个小列表的长度
    :return:
    '''
    list_of_group = zip(*(iter(list_info),) * per_list_len)
    end_list = [list(i) for i in list_of_group]  # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count != 0 else end_list
    return end_list


def softmax(l):
    l_exp = np.exp(l)
    l_exp_sum = l_exp.sum(axis=1).reshape(-1, 1)
    return l_exp / l_exp_sum


def calc_eer_thre(fpr, tpr, thre):
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    th = interp1d(fpr, thre)(eer)
    return eer, th


loss_train_seq = []

for num in range(10,20):
    root_train = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\二分类data\original_deepfake_face2face_faceswap_997\train'
    root_test = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\二分类data\original_deepfake_face2face_faceswap_997\test'

    train_dataset = ImageFolder(root_train, transform=data_transform)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_dataset = ImageFolder(root_test, transform=data_transform)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # net = VGG19(2)
    net = Net3()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # lambda1 = lambda epoch: 0.98 ** epoch
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=30, step_size_down=30, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

    EPOCHS = 50
    loss_train_seq = []
    loss_test_seq = []
    max_acc = 0
    # net = torch.load('model.pkl')
    net = net.to(device)

    for epoch in range(1, EPOCHS + 1):
        train_l = train(net, train_loader, optimizer, epoch)
        # test_l, acc_test = test(net, test_loader)
        torch.save(net, r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\二分类data\original_deepfake_face2face_faceswap_997\model3_' + str(num) + '.pkl')
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
        r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\二分类data\original_deepfake_face2face_faceswap_997\model3_' + str(num) + '.pkl')
    net1 = net1.to(device_cpu)
    net1.eval()

    test_res = []
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.tensor(data, dtype=torch.float32)
            # data, target = data.to(device), target.to(device)
            # print(target)
            output = net1(data)
            output = softmax(output).numpy()

            # pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            test_res.append(output[0][0])

    test_result = list_of_groups(test_res, 21)
    test_score = [sum(i) / 21 for i in test_result]
    test_label = [1] * 299 + [0] * 299

    res = 0
    for i in range(598):
        if test_score[i] >= 0.5 and test_label[i] == 1:
            res += 1
        if test_score[i] < 0.5 and test_label[i] == 0:
            res += 1
    print("mean:",res / 598)

    # test_score_maj = [sum([1 if x > 0.5 else 0 for x in i]) / 21 for i in test_result]
    # res = 0
    # for i in range(598):
    #     if test_score_maj[i] >= 0.5 and test_label[i] == 1:
    #         res += 1
    #     if test_score_maj[i] < 0.5 and test_label[i] == 0:
    #         res += 1
    # print("maj:", res / 598)
    #
    # test_score_log = [sum([math.log(x/(1-x)) for x in i]) / 21 for i in test_result]
    # res = 0
    # for i in range(598):
    #     if test_score_log[i] >= 0.5 and test_label[i] == 1:
    #         res += 1
    #     if test_score_log[i] < 0.5 and test_label[i] == 0:
    #         res += 1
    # print("log:", res / 598)
