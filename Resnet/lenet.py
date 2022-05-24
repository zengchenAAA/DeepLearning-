import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    """
    for CIFAR10 dataset
    """
    def __init__(self):
        super(Lenet5, self).__init__()
        self.model = nn.Sequential(
            # x:[b,3,32,32] => [b,6,30,30]  30 = 32+2-5+1
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1),
            # x:[b,6,30,30] => [b,6,15,15]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # x:[b,6,15,15] => [b,16,13,13]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1),
            # x:[b,16,13,13] => [b,16,6,6]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # x:[b,16*6*6]
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        # use Cross Entropy Loss
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        :param x: [b,3,32,32]
        :return:
        """
        # [b, 3, 32, 32] => [b, 10]
        logits = self.model(x)
        return logits

