# Deep Q Learning

This is my first Deep Q Learning implementation. It learns to play Cartpole in 2 minutes on my MacBook Pro 2017.

To validate my pretrained model:

`python3 validate.py`

To train your own model:

`python3 train.py`

While training, you can monitor it by running `tensorboard --logdir runs/` (assuming you have [Tensorboard](https://www.tensorflow.org/tensorboard) installed).

## Architecture

Here's a quick overview:

- `model.py`: Class `Model` which can be used to make a dense model of any size desired.
- `trainer.py`: Class `Trainer` which takes a model and an OpenAI Gym environment, samples from the environment, and trains the model on that.
- `train.py`: Entry-point if you want to train a cartpole model.
- `validate.py`: Entry-point if you want to validate a saved model.
