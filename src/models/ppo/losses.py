import torch


def GAE(rewards, values, gamma, lambda_, device):
    delta = rewards + gamma * values[:, 1:] - values[:, :-1]
    mask = torch.triu(torch.ones((delta.shape[-1], delta.shape[-1]))).unsqueeze(0)
    discounting = (
        torch.pow(gamma * lambda_, (mask.cumsum(-1) - 1.0).clamp(min=0.0)) * mask
    )
    advantage = (discounting.to(device) * delta.unsqueeze(1)).sum(-1)
    # adjustment_factor = 1.0 - torch.pow(lambda_, mask.sum(-1))
    # adjustment_factor = adjustment_factor.unsqueeze(1) * mask
    return advantage
