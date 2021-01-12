import random
import torch
from torch import nn


def train_batch(replay_buffer, model, optimizer, batch_size, discount_factor):
    batch = random.sample(replay_buffer, batch_size)

    observations = torch.tensor([x[0] for x in batch]).float()
    actions = torch.tensor([x[1] for x in batch])
    rewards = torch.tensor([x[2] for x in batch])
    not_done_mask = torch.tensor([not x[3] for x in batch])
    post_observation = torch.tensor([x[4] for x in batch]).float()

    post_max_q = model(post_observation).max(dim=1).values

    y = model(observations).gather(1, actions.reshape((-1, 1))).reshape((-1,))
    bellman = rewards + (not_done_mask * post_max_q * discount_factor)

    optimizer.zero_grad()
    loss = nn.MSELoss()(y, bellman)
    loss.backward()
    optimizer.step()

    return loss
