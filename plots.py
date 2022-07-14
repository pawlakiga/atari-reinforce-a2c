import matplotlib.pyplot as plt
from util import *

root_path = 'C:\\Users\\Iga\\Documents\\atari-reinforce-a2c'
folder_path ='C:\\Users\\Iga\\Documents\\atari-reinforce-a2c\\rewards\\a2c'
file_names  = [
    ###########################################################################
    # REINFORCE 
    # gamma
    # 'reinforce_test_classic_lr0.001_gamma0.8_batch128_10-07_11-35.csv', 
    # 'reinforce_test_classic_lr0.001_gamma0.95_batch128_10-07_12-39.csv',
    # 'reinforce_test_classic_lr0.001_gamma1_batch128_10-07_14-17.csv'  
    
    # # batch size
    # 'reinforce_test_classic_lr0.01_gamma1_batch8_10-07_13-36.csv',
    # 'reinforce_test_classic_lr0.01_gamma1_batch32_10-07_13-53.csv'  ,
    # 'reinforce_test_classic_lr0.01_gamma1_batch128_10-07_14-23.csv',

    # learning rates
    # 'reinforce_test_classic_lr1e-05_gamma1_batch128_10-07_14-12.csv',
    # 'reinforce_test_classic_lr0.01_gamma1_batch64_10-07_14-06.csv' , 
    # 'reinforce_test_classic_lr0.01_gamma1_batch128_10-07_14-23.csv'
    ############################################################################
    # BATCH BASELINE
    # gamma
    # 'reinforce_baseline_test_classic_lr[0.001, 0.001]_gamma0.8_batch128_10-07_18-05.csv', 
    # 'reinforce_baseline_test_classic_lr[0.001, 0.001]_gamma0.95_batch128_10-07_19-14.csv',
    # 'reinforce_baseline_test_classic_lr[0.001, 0.001]_gamma1_batch128_10-07_20-54.csv'  
    
    # batch size
    # 'reinforce_baseline_test_classic_lr[0.001, 0.001]_gamma1_batch8_10-07_20-09.csv',
    # 'reinforce_baseline_test_classic_lr[0.001, 0.001]_gamma0.99_batch64_10-07_19-46.csv'  ,
    # 'reinforce_baseline_test_classic_lr[0.001, 0.001]_gamma0.99_batch128_10-07_20-03.csv',

    # learning rates
    # 'reinforce_baseline_test_classic_lr[1e-05, 1e-05]_gamma1_batch128_14-07_01-27.csv',
    # 'reinforce_baseline_test_classic_lr[1e-05, 0.001]_gamma1_batch128_14-07_01-34.csv', 
    # 'reinforce_baseline_test_classic_lr[0.001, 0.001]_gamma1_batch128_14-07_01-42.csv'

    # NO BATCH BASELINE
    # learning rates 
    # 'reinforce_baseline_test_classic_nb_lr[1e-05, 1e-05]_gamma1_batch1_14-07_06-55.csv',
    # 'reinforce_baseline_test_classic_nb_lr[1e-05, 0.001]_gamma1_batch1_14-07_07-30.csv', 
    # 'reinforce_baseline_test_classic_nb_lr[0.001, 0.001]_gamma1_batch1_14-07_08-55.csv'

    #gamma
    # 'reinforce_baseline_test_classic_nb_lr[0.001, 0.001]_gamma0.8_batch1_10-07_23-18.csv',
    # 'reinforce_baseline_test_classic_nb_lr[0.001, 0.001]_gamma0.95_batch1_14-07_02-52.csv', 
    # 'reinforce_baseline_test_classic_nb_lr[0.001, 0.001]_gamma1_batch1_14-07_08-55.csv'

    ########################
    # comparison baseline
    # 'reinforce_baseline_test_classic_lr[0.001, 0.001]_gamma1_batch128_10-07_20-54.csv',
    # 'reinforce_baseline_test_classic_nb_lr[0.001, 0.001]_gamma1_batch1_14-07_08-55.csv'

  #########################
    # 'actor_critic_nb_lr[0.001, 0.001]_gamma1_batch1_14-07_14-32.csv',
    # 'actor_critic_nb_lr[0.001, 1e-05]_gamma1_batch1_14-07_14-27.csv',
    # 'actor_critic_nb_lr[0.001, 0.001]_gamma0.99_batch1_14-07_14-13'
#########################

    ]

# colors = ['#DC143C', 'b', 'm', '#ff7e55']
colors = [ '#ff7e55', '#ff007f',  '#3333ff',  '#00cc00']

# labels = ['gamma = 0.8','gamma = 0.95' , 'gamma = 1']
# labels = ['batch size = 8', 'batch size = 64', 'batch size = 128']
labels = ['learning rates = [0.00001, 0.00001]', 'learning rates = [0.001, 0.00001]', 'learning rates = [0.001, 0.001]']
# labels = ['learning rate = 0.00001','learning rate = 0.001', 'learning rate = 0.01' ]
# labels = ['With experience replay', 'Without experience replay']

# title = 'Reinforce with experience replay - discount factor'
# title = 'Reinforce with experience replay - learning rates'
# title = 'Reinforce with experience replay - batch size'

# title = 'Reinforce with baseline and experience replay - discount factor'
# title = 'Reinforce with baseline and experience replay - batch size'
# title = 'Reinforce with baseline and experience replay - learning rates'
# title = 'Reinforce with baseline  - learning rates'
# title = 'Reinforce with baseline  - discount factor'

# title = 'Reinforce with baseline  - experience replay comparison'
title = 'Actor critic - learning rates'

episodes_mean = 20


for file_name in file_names:
    data = load_from_file(file_path= folder_path + '\\' + file_name)
    means = [np.mean(data[i:i+episodes_mean]) for i in range(len(data)-episodes_mean)]
    plt.plot(range(1,len(data)+1,1), data, color = colors[file_names.index(file_name)], alpha = 0.15)
    plt.plot(range(1,len(means)+1,1), means, color = colors[file_names.index(file_name)], label = labels[file_names.index(file_name)])

plt.title(title)
plt.xlabel('Episode')
plt.ylabel(f'Cumulative reward (mean for every {episodes_mean} episodes)')    
plt.legend()
plt.grid(True)
plt.savefig(f'{root_path}\\figures\\{title}.png')
plt.show()
