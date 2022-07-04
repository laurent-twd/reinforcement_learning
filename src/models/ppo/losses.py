import torch


def GAE(rewards, values, gamma, lambda_):
    delta = rewards + gamma * values[:, 1:] - values[:, :-1]
    mask = torch.triu(torch.ones((delta.shape[-1], delta.shape[-1]))).unsqueeze(0)
    discounting = torch.pow(gamma * lambda_, mask.cumsum(-1) - 1.0) * mask
    advantage = (discounting * delta.unsqueeze(1)).sum(-1)
    # adjustment_factor = 1.0 - torch.pow(lambda_, mask.sum(-1))
    # adjustment_factor = adjustment_factor.unsqueeze(1) * mask
    return advantage


def clipped_loss(rewards, values, ratios, gamma, lambda_, epsilon, c_1):

    advantage = GAE(rewards, values, gamma, lambda_)

    # Actor Loss
    p1 = ratios[:, :-1] * advantage
    p2 = torch.where(
        advantage >= 0, (1.0 + epsilon) * advantage, (1.0 - epsilon) * advantage
    )
    actor_loss = torch.minimum(p1, p2)
    # Critic Loss
    mask = torch.triu(torch.ones((rewards.shape[-1], rewards.shape[-1]))).unsqueeze(0)
    forward_rewards = rewards.unsqueeze(-1) * mask
    discounting = torch.pow(gamma, mask.cumsum(-1) - 1.0) * mask
    target_values = (forward_rewards * discounting).sum(-1)
    critic_loss = (values - target_values).square().mean()
    return -(actor_loss.mean() - c_1 * critic_loss)


def easier_loss(rewards, values, ratios, gamma, lambda_, epsilon, c_1):
    return -rewards.mean()
