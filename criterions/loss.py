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

class CosineLoss():

    def __init__(self):
        self.criterion = nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, x1, x2, label):
        loss = self.criterion(x1, x2, torch.tensor(label).cuda())
        return loss
