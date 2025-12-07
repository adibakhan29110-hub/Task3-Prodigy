import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def __call__(self, pred, target_is_real):
        # target_is_real: True (1) or False (0)
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)