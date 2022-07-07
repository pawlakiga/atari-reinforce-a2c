import numpy as np
from random import random
import config
import tensorflow as tf


def random_policy(action_space, state, model):
    return action_space.sample()


def network_model_policy(action_space, state, model):
    action_probs = model(tf.expand_dims(state,0))
    if config.DEBUG_PRINT:
        print(f"Action probs: {action_probs}")
    return np.random.choice(action_space.n, p=np.squeeze(action_probs))

