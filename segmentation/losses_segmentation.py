import torch
from scipy.ndimage import distance_transform_edt as distance_transform

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





def boundary_loss(y_real, y_pred, alpha=0.01):
    y_real = y_real.squeeze(1)
    y_pred = y_pred.squeeze(1)
    probs = torch.sigmoid(y_pred)
    preds = (probs > 0.5).cpu()

    foreground_map = - distance_transform(y_real.cpu())
    background_map = distance_transform((1 - y_real).cpu())
    dist_map = torch.tensor(background_map + foreground_map).cpu()

    boundary_loss = ((dist_map * preds).sum((1, 2)) / 256 / 256).mean()

    smooth = 1e-8
    num = (2. * y_real * probs).sum((1, 2))
    den = (y_real + probs).sum((1, 2))
    dice = (num + smooth) / (den + smooth)
    dice = 1 - dice.sum() / y_real.size(0)

    return (1 - alpha) * dice + alpha * boundary_loss

