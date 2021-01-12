import random

import gym
import torch

from model import Model


def validate(model, episodes):
    env = gym.make('CartPole-v0')

    scores = []

    for episode in range(episodes):
        observation = env.reset()
        reward = 0
        done = False
        info = None

        t = 0

        while not done:
            q = model(torch.from_numpy(observation).float())
            action = q.argmax().item()

            observation, reward, done, info = env.step(action)

            t += 1

        scores.append(t)

    return sum(scores) / len(scores)


if __name__ == "__main__":
    model = Model(input_size=4, hidden_layers=2,
                  hidden_layer_size=8, output_size=2)

    model.load_state_dict(torch.load('models/pretrained'))

    print('Avg. score:', validate(model, 100))
