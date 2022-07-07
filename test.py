from learning_functions import *
import config

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

functions_dict = {
    1 : reinforce_classic, 
    2 : reinforce_baseline_classic, 
    3 : reinforce_classic_no_batch, 
    4 : reinforce_baseline_classic_no_batch, 
    5 : reinforce_baseline_atari

}


fun = functions_dict[config.chosen_fun]
fun()
# reinforce_baseline_atari(episodes_no=episodes_no, policy_learning_rate=policy_learning_rate, value_learning_rate=value_learning_rate)