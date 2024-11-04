import torch
import torch.nn.functional as F


def cosine_similarity_loss(output_de_st_list):
    loss = 0
    for instance in output_de_st_list:
        _, _, h, w = instance.shape
        loss += torch.sum(instance) / (h * w)
    return loss

def contrast_loss(output_de_st_list):
    loss = 0
    current_batchsize = output_de_st_list[0][0].shape[0]

    target = -torch.ones(current_batchsize).to('cuda')
    contrast = torch.nn.CosineEmbeddingLoss(margin=0.5)
    loss = contrast(output_de_st_list[0][0].view(output_de_st_list[0][0].shape[0], -1), output_de_st_list[1][0].view(output_de_st_list[1][0].shape[0], -1), target = target) + \
        contrast(output_de_st_list[0][1].view(output_de_st_list[0][1].shape[0], -1), output_de_st_list[1][1].view(output_de_st_list[1][1].shape[0], -1), target = target)+ \
        contrast(output_de_st_list[0][2].view(output_de_st_list[0][2].shape[0], -1), output_de_st_list[1][2].view(output_de_st_list[1][2].shape[0], -1), target = target)

    return loss

def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs, targets, reduction=reduction)
