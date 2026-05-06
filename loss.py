import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# loss.py (请添加或替换)
import torch
import torch.nn as nn
import torch.nn.functional as F

class NaNMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,lossmse):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        loss = torch.sqrt(lossmse(y_true, y_pred))
        return loss
