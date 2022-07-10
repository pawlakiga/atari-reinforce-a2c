import gym
from sympy.core import function

DEBUG_PRINT = False
REPLAY_BUFFER_LEN = 10000
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
NUM_BATCHES = 12
LOG_PATH = 'logs/classic_baseline'
STEPS_LIMIT = 10000

CLASSIC_ENVIRONMENT_NAME = 'CartPole-v1'
ATARI_ENVIRONMENT_NAME = 'DemonAttack-v4'

policy_learning_rate = 1e-3
value_learning_rate = 1e-3
episodes_no = 500

"""
 functions_dict = {
    1 : reinforce_classic, 
    2 : reinforce_baseline_classic, 
    3 : reinforce_classic_no_batch, 
    4 : reinforce_baseline_classic_no_batch
    5 : reinforce_baseline_atari
    6 : actor_critic_classic_no_batch
}
"""
chosen_fun = 3


