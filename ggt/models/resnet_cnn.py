import torch
import torch.nn as nn
from torchvision import models
from ggt.models import *

'''
A discrete Convolutional Neural Network implemented into the GaMPEN framework.
To use, use in conjunction with train_cnn.py or train_optuna_cnn.py.
'''
class anomaly_cnn(nn.Module):
    def __init__(
            self,
            channels,
            cutout_size=267,
            n_classes=1,
            n_out=1,
            pretrained=True,
            dropout=False,
            dropout_rate=0.05
    ):
        super(anomaly_cnn, self).__init__()
        self.channels = channels
        self.cutout_size = cutout_size

        self.n_out = n_out # Specifying the number of outputs for the network. If binary, only uses 1.
        self.n_classes = n_classes # n_classes specified as a string array for multiple
        self.pretrained = pretrained

        self.model = models.resnet50(pretrained=self.pretrained)
        self.model.conv1 = nn.Conv2d(self.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=2048, eps=1e-05, momentum=0.1),
            nn.Dropout(p=dropout_rate if dropout else 0),
            nn.Linear(2048, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_classes)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0, posinf=1)
        output = self.model(x)

        if self.n_out < 2:
            output = torch.flatten(output)

        return output
