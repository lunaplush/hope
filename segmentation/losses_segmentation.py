import torch

def bce_loss(y_real, y_pred):
    loss = torch.mean(y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred)))
    return loss

def dice_loss(y_real, y_pred):
    SMOOTH = 1e-8
    outputs = y_pred.sigmoid().squeeze(1)
    labels = y_real.squeeze(1)
    num = (outputs * labels).sum().float()
    den = (outputs + labels).sum().float()
    res = 1 - ((2 * (num + SMOOTH))/(den + SMOOTH))
    return res

def focal_loss(y_real, y_pred, eps = 1e-6, gamma = 2):
    probs = torch.clamp(torch.sigmoid(y_pred), min=eps, max=1-eps)
    return -torch.mean((1 - probs) ** gamma * y_real * torch.log(probs) + (1 - y_real) * torch.log(1 - probs))

