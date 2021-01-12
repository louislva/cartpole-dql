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


class Hyperparameters:
    REPLAY_BUFFER_SIZE = 10000
    TRAINING_START_STEP = 2500

    TRAINING_INTERVAL = 4
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    LEARNING_RATE_STEP = 95000
    DISCOUNT_FACTOR = 0.95

    EPSILON_START = 1
    EPSILON_DECAY_PERIOD = 10000
    EPSILON_MIN = 0.025
    EPSILON_REPEAT_PERIOD = 35000

    SAVING_INTERVAL = 5000


class Trainer:
    def __init__(self, model, env, hyperparameters):
        # hyperparameters
        self.hyper = hyperparameters

        # objects
        self.model = model
        self.env = env
        self.summary_writer = SummaryWriter()

        # buffers
        self.last_100_scores = deque(maxlen=100)
        self.replay_buffer = deque(
            maxlen=self.hyper.REPLAY_BUFFER_SIZE)

        # optimizers
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.hyper.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.1)
        print('Learning rate', self.optimizer.param_groups[0]['lr'])

        pass

    @property
    def epsilon(self):
        start = self.hyper.EPSILON_START
        period = self.hyper.EPSILON_DECAY_PERIOD
        min_value = self.hyper.EPSILON_MIN
        repeat_period = self.hyper.EPSILON_REPEAT_PERIOD
        n = self.n

        if(repeat_period is not None):
            n = n % repeat_period

        return max(start * (1 - (n / period)), min_value)

    def act(self, observation):
        if(random.random() > self.epsilon):
            q = self.model(torch.from_numpy(observation).float())
            return q.argmax().item()
        else:
            return self.env.action_space.sample()

    def train(self):
        self.n = 0
        self.episode = 0

        self.summary_writer.add_scalar(
            'Learning Rate', self.hyper.LEARNING_RATE, self.n)

        while self.n < 250000:
            # EPISODE
            t = 0

            # first frame
            observation = self.env.reset()
            done = False

            while not done:
                # STEP
                action = self.act(observation)
                post_observation, reward, done, info = env.step(
                    action
                )
                self.replay_buffer.append(
                    (observation, action, reward, done, post_observation)
                )
                observation = post_observation

                t += 1
                self.n += 1
                self.post_step(self.episode, self.n, t)

            self.episode += 1
            self.post_episode(self.episode, self.n, t)

    def post_step(self, episode, n, t):
        self.summary_writer.add_scalar('Epsilon', self.epsilon, n)

        # Only start doing these things when we start training
        if(n >= self.hyper.TRAINING_START_STEP):
            # PER [LEARNING_RATE_STEP] STEPS
            if(n % self.hyper.LEARNING_RATE_STEP == 0):
                self.scheduler.step()
                self.summary_writer.add_scalar(
                    'Learning Rate', self.hyper.LEARNING_RATE * (0.1 ** (n / self.hyper.LEARNING_RATE_STEP)), n)

            # PER [TRAINING_INTERVAL] STEPS
            if(n % self.hyper.TRAINING_INTERVAL == 0):
                loss = self.train_batch()

                self.summary_writer.add_scalar('Loss', loss, n)

            # PER [SAVING_INTERVAL] STEPS
            if(n % self.hyper.SAVING_INTERVAL == 0):
                if(not os.path.isdir('models/')):
                    os.mkdir('models/')
                torch.save(self.model.state_dict(), 'models/' + str(n // self.hyper.SAVING_INTERVAL) +
                           '-avg' + str(int(sum(self.last_100_scores) / len(self.last_100_scores))))

    def post_episode(self, episode, n, t):
        self.summary_writer.add_scalar('Score', t, n)

        self.last_100_scores.append(t)
        self.summary_writer.add_scalar(
            'Rolling Avg. Score',
            sum(self.last_100_scores) /
            len(self.last_100_scores),
            n
        )

        # PER 100 EPISODES
        if(episode % 100 == 0):
            env.render()

    def train_batch(self):
        replay_buffer = self.replay_buffer
        model = self.model
        optimizer = self.optimizer
        batch_size = self.hyper.BATCH_SIZE
        discount_factor = self.hyper.DISCOUNT_FACTOR

        batch = random.sample(replay_buffer, batch_size)

        observations = torch.tensor([x[0] for x in batch]).float()
        actions = torch.tensor([x[1] for x in batch])
        rewards = torch.tensor([x[2] for x in batch])
        not_done_mask = torch.tensor([not x[3] for x in batch])
        post_observation = torch.tensor([x[4] for x in batch]).float()

        post_max_q = model(post_observation).max(dim=1).values

        y = model(observations).gather(
            1, actions.reshape((-1, 1))).reshape((-1,))
        bellman = rewards + (not_done_mask * post_max_q * discount_factor)

        optimizer.zero_grad()
        loss = nn.MSELoss()(y, bellman)
        loss.backward()
        optimizer.step()

        return loss


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    model = Model(input_size=4, hidden_layers=2,
                  hidden_layer_size=8, output_size=2)
    trainer = Trainer(model, env, Hyperparameters())

    trainer.train()

    env.close()
