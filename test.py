from operator import le
from learning_functions import *
import config

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

functions_dict = {
    1 : reinforce_classic, 
    2 : reinforce_baseline_classic, 
    3 : reinforce_classic_no_batch, 
    4 : reinforce_baseline_classic_no_batch, 
    5 : reinforce_baseline_atari,
    6 : actor_critic_classic_no_batch


}


# fun = functions_dict[config.chosen_fun]
# fun()
# reinforce_baseline_atari(episodes_no=episodes_no, policy_learning_rate=policy_learning_rate, value_learning_rate=value_learning_rate)

learning_rates = [1e-5, 1e-3]
batch_sizes = [8, 16, 32, 64, 128]
discount_factors = [0.8, 0.9, 0.95, 0.99, 1]

for fun_no in [2,3,4,6]:
    for df in discount_factors:
        for bs in batch_sizes:
            for policy_lr in learning_rates:
                for value_lr in learning_rates:
                    if value_lr > 1e-5 and fun_no == 1 or fun_no == 3:
                        continue
                    else:
                        test_fun = functions_dict[fun_no]
                        print("-------------------------------------------------------------------------------------------------------------------")
                        print(f"Testing {test_fun.__name__}, with gamma = {df}, batch = {bs}, lrs = {policy_lr, value_lr}")
                        test_fun(episodes_no = episodes_no, 
                                value_learning_rate = value_lr, 
                                policy_learning_rate = policy_lr, 
                                discount_factor = df, 
                                batch_size = bs)