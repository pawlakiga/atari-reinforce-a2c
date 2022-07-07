from sympy.core import function

DEBUG_PRINT = False
REPLAY_BUFFER_LEN = 1000
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 1
LOG_PATH = 'logs/batches'

CLASSIC_ENVIRONMENT_NAME = 'CartPole-v1'
ATARI_ENVIRONMENT_NAME = 'DemonAttack-v4'

policy_learning_rate = 1e-5
value_learning_rate = 1e-5
episodes_no = 500

"""
 functions_dict = {
    1 : reinforce_classic, 
    2 : reinforce_baseline_classic, 
    3 : reinforce_classic_no_batch, 
    4 : reinforce_baseline_classic_no_batch
    5 : reinforce_baseline_atari
}
"""
chosen_fun = 5


