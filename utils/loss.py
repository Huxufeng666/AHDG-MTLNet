
import torch
import torch.nn as nn
# def dice_loss(pred_logits, targets, smooth=1e-6):
#     probs = torch.sigmoid(pred_logits)
#     B = probs.shape[0]
#     probs  = probs.view(B, -1)
#     targets = targets.view(B, -1)
#     inter = (probs * targets).sum(dim=1)
#     union = probs.sum(dim=1) + targets.sum(dim=1)
#     dice = (2. * inter + smooth) / (union + smooth)
#     return 1 - dice.mean()

import torch
import torch.nn.functional as F


def boundary_loss(pred, target):
    """
    pred: logits [B,1,H,W]
    target: mask [B,1,H,W]
    """

    pred = torch.sigmoid(pred)

    # ===== x方向梯度 =====
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

    # ===== y方向梯度 =====
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)

    return loss_x + loss_y

def dice_loss(pred_logits, targets, smooth=1e-6):


    if pred_logits.shape[1] == 1:
        probs = torch.sigmoid(pred_logits)
    else:
        probs = torch.softmax(pred_logits, dim=1)[:,1:2]


    targets = (targets > 0.5).float()

    B = probs.shape[0]

    probs = probs.reshape(B, -1)
    targets = targets.reshape(B, -1)

    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * inter + smooth) / (union + smooth)

    return 1 - dice.mean()




class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        smooth = 1e-5

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        loss = (1 - tversky) ** self.gamma

        return loss