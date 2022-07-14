import gym
import tensorflow as tf
import numpy as np

env = gym.make("Breakout-v4")

state = env.reset()
print(state)
grayscale = tf.image.rgb_to_grayscale(state)
print(grayscale)