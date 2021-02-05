import torch
import torch.nn as nn

class Loss():

    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, prediction, gt):
        """
        prediction : [N, C, D1, ... DK], K-dim loss
        gt : [N, C]
        """
        loss = self.criterion(prediction, gt)

        return loss
