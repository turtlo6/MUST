import torch
import torch.nn.functional as F

device = torch.device('cuda')


def get_reg_loss(beta, gnd_list, pred_adj_list, theta, lambda_reg=0.0005):
    """
    获取适应稀疏情况的不加权重构误差，交叉熵损失和1范数正则化
    """
    loss = 0.0
    win_size = len(pred_adj_list)
    for i in range(win_size):
        gnd = gnd_list[i]
        pred_adj = pred_adj_list[i]
        decay = (1 - theta) ** (win_size - i - 1)  # Decaying factor

        # 计算原始的加权重构误差
        weight = gnd * (beta - 1) + 1
        reconstruction_loss = torch.mean(torch.sum(weight * torch.square(gnd - pred_adj), dim=1), dim=-1)

        # 计算1范数正则化项
        l1_regularization = lambda_reg * torch.sum(torch.abs(pred_adj))

        # 总损失
        total_loss = decay * (reconstruction_loss + l1_regularization)

        loss += total_loss
    return loss
