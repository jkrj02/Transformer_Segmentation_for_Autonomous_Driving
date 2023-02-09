import torch
import torch.nn as nn
import torch.nn.functional as F


class Entropy_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target):
        target = torch.squeeze(target, dim=1).long()
        loss_fnc = nn.CrossEntropyLoss()
        loss = loss_fnc(inputs, target)

        return loss
