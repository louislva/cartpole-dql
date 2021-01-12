import random

import gym

from model import Model
from trainer import Trainer

random.seed(1)


class Hyperparameters:
    REPLAY_BUFFER_SIZE = 10000
    TRAINING_START_STEP = 2500
    TRAIN_FOR_STEPS = 250000

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


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    model = Model(input_size=4, hidden_layers=2,
                  hidden_layer_size=8, output_size=2)

    # Trainer is responsible for sampling environment data (trying things)
    # and updating the weights of the model. It will log to the runs/ directory.
    # Use Tensorboard to read it.
    trainer = Trainer(model, env, Hyperparameters())

    trainer.train()

    env.close()
