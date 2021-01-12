import random
from collections import deque
import os

import gym
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from misc import train_batch
from model import Model

random.seed(1)

# Hyperparameters/constants

REPLAY_BUFFER_SIZE = 10000
TRAINING_START_STEP = 2500

TRAINING_INTERVAL = 4
BATCH_SIZE = 16
LEARNING_RATE = 0.001
LEARNING_RATE_STEP = 95000
DISCOUNT_FACTOR = 0.95
# EPOCHS = repetitions of the same data = BATCH_SIZE / TRAINING_INTERVAL

EPSILON_START = 1
EPSILON_DECAY_PERIOD = 10000
EPSILON_MIN = 0.025
EPSILON_REPEAT_PERIOD = 35000

SAVING_INTERVAL = 5000


def get_epsilon(n, start, period, min_value, repeat_period=None):
    if(repeat_period is not None):
        n = n % repeat_period

    return max(start * (1 - (n / period)), min_value)


def train_loop(model):
    writer = SummaryWriter()
    writer.add_scalar('Learning Rate', LEARNING_RATE, 0)

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    env = gym.make('CartPole-v0')

    n = 0

    rolling_avg_score_deque = deque(maxlen=100)

    for episode in range(10000):
        observation = env.reset()
        reward = 0
        done = False
        info = None

        t = 0

        while not done:
            epsilon = get_epsilon(
                n, EPSILON_START, EPSILON_DECAY_PERIOD, EPSILON_MIN, EPSILON_REPEAT_PERIOD)
            if(random.random() > epsilon):
                q = model(torch.from_numpy(
                    observation).float())
                action = q.argmax().item()
            else:
                action = random.randrange(2)

            post_observation, reward, done, info = env.step(action)
            replay_buffer.append(
                (observation, action, reward, done, post_observation)
            )
            observation = post_observation

            t += 1
            n += 1

            # NON-GAME LOGIC
            if(n >= TRAINING_START_STEP):
                writer.add_scalar('Epsilon', epsilon, n)

                # PER EPISODE
                if(done):
                    writer.add_scalar('Score', t, n)
                    rolling_avg_score_deque.append(t)
                    writer.add_scalar(
                        'Rolling Avg. Score',
                        sum(rolling_avg_score_deque) /
                        len(rolling_avg_score_deque),
                        n
                    )

                # PER 100 EPISODES
                if(episode % 100 == 0):
                    env.render()

                # PER [LEARNING_RATE_STEP] STEPS
                if(n % LEARNING_RATE_STEP == 0):
                    scheduler.step()
                    writer.add_scalar(
                        'Learning Rate', LEARNING_RATE * (0.1 ** (n / LEARNING_RATE_STEP)), n)

                # PER [TRAINING_INTERVAL] STEPS
                if(n % TRAINING_INTERVAL == 0):
                    loss = train_batch(replay_buffer, model,
                                       optimizer, BATCH_SIZE, DISCOUNT_FACTOR)

                    writer.add_scalar('Loss', loss, n)

                # PER [SAVING_INTERVAL] STEPS
                if(n % SAVING_INTERVAL == 0):
                    if(not os.path.isdir('models/')):
                        os.mkdir('models/')
                    torch.save(model.state_dict(), 'models/' + str(n // SAVING_INTERVAL) +
                               '-avg' + str(int(sum(rolling_avg_score_deque) / len(rolling_avg_score_deque))))

    env.close()


if __name__ == "__main__":
    model = Model(input_size=4, hidden_layers=2,
                  hidden_layer_size=8, output_size=2)

    train_loop(model)
