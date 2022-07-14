from numpy import expand_dims
from agent import Agent
from config import *
from policies import *
from network_builders import *
from network_training_functions import *
from util import *
import gym

def reinforce_classic(episodes_no = episodes_no,
    policy_learning_rate = policy_learning_rate,
    value_learning_rate = value_learning_rate,
    discount_factor = DISCOUNT_FACTOR, 
    batch_size = BATCH_SIZE):

    """
    Function to perform the training of the agent with the REINFORCE algorithm in the classic atari environment with experience replay
    """

    # Creation of file writer for tensorboard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{LOG_PATH}/reinforce_classic/lr{[value_learning_rate, policy_learning_rate]}_batch{BATCH_SIZE}_' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Environment initialisation
    env = gym.make(CLASSIC_ENVIRONMENT_NAME)
    # Agent initialisation
    agent = Agent(environment=env, start_state=env.reset())
    # Policy network creation
    policy_model = dense_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)

    rewards_mem = []
    steps_mem = []

    
    for episode in range(episodes_no):
        if DEBUG_PRINT:
            print(f"================== Episode {episode} =====================")
        step = agent.play_episode(policy = network_model_policy, policy_model= policy_model, experience_replay=True, discount_factor = discount_factor)
        # After some episodes when we have enough samples in the buffer - start training
        if episode >= 20:
            total_loss = 0
            states, actions, targets, _ , _ = agent.replay_buffer.get_random_experience_batch(batch_size = batch_size)
            total_loss = policy_training_step(model = policy_model, states = states, actions = actions, targets = targets, optimizer=policy_optimizer)
            # Write data to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('policy mean loss', total_loss/batch_size, step = episode)
                tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        if not DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    # make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce - {CLASSIC_ENVIRONMENT_NAME}')
    # np.savetxt(file_name('reinforce_classic\\reinforce_test_classic', policy_learning_rate, 'rewards', discount_factor=discount_factor, batch_size=batch_size), rewards_mem, delimiter=',')



def reinforce_baseline_classic(episodes_no = episodes_no,
                                policy_learning_rate = policy_learning_rate,
                                value_learning_rate = value_learning_rate, 
                                discount_factor = DISCOUNT_FACTOR, 
                                batch_size = BATCH_SIZE):
    """
    Function to perform the training of the agent with the REINFORCE algorithm in the classic atari environment with baseline and experience replay
    """

    # Creation of file writer for tensorboard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{LOG_PATH}/baseline_classic/lr{[value_learning_rate, policy_learning_rate]}_batch{BATCH_SIZE}_' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Environment initialisation
    env = gym.make(CLASSIC_ENVIRONMENT_NAME)
    # Agent initialisation
    agent = Agent(environment=env, start_state=env.reset())
    # Policy and value netowrks creation
    value_model = dense_value_network(input_shape=env.observation_space.shape)
    policy_model = dense_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)
    # Optimizers and metrics initialisation
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate = value_learning_rate)
    value_metrics = tf.keras.metrics.MeanSquaredError()

    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        step = agent.play_episode(policy = network_model_policy, policy_model= policy_model, experience_replay=True, discount_factor = discount_factor)
        if episode >= 20:
            states, actions, targets, _ , _ = agent.replay_buffer.get_random_experience_batch(batch_size = batch_size)
            # Value network training step
            total_value_loss = value_training_step(model = value_model, states = states, targets = targets, optimizer=value_optimizer, metrics=value_metrics)
            # Calculation of baselines
            baselines = value_model(states)
            # delta = G_t - v(S,w)
            deltas = tf.math.subtract(targets, baselines)
            # Policy network training step
            total_policy_loss = policy_training_step(model = policy_model, states = states, actions = actions, targets = deltas, optimizer=policy_optimizer)
            # Writing data to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('policy mean loss', total_policy_loss/batch_size, step = episode)
                tf.summary.scalar('value mean loss', total_value_loss/batch_size, step = episode)
                tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
                tf.summary.scalar('value mse', value_metrics.result(), step=episode)

        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        if not DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    # make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {CLASSIC_ENVIRONMENT_NAME}')
    # np.savetxt(file_name('baseline_classic\\reinforce_baseline_test_classic', [value_learning_rate, policy_learning_rate], 'rewards', discount_factor=discount_factor, batch_size=batch_size), rewards_mem, delimiter=',')


def reinforce_baseline_classic_no_batch(episodes_no = episodes_no, 
                                        value_learning_rate = value_learning_rate, 
                                        policy_learning_rate = policy_learning_rate, 
                                        discount_factor = DISCOUNT_FACTOR, 
                                        batch_size = BATCH_SIZE):

    # Creation of file writer for tensorboard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{LOG_PATH}/baseline_classic_nb/lr{[value_learning_rate, policy_learning_rate]}_batch{BATCH_SIZE}_' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # Environment and agent initialisation
    env = gym.make(CLASSIC_ENVIRONMENT_NAME)
    agent = Agent(environment=env, start_state=env.reset())

    # building networks
    value_model = dense_value_network(input_shape=env.observation_space.shape)
    policy_model = dense_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)

    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate = value_learning_rate)
    value_metrics = tf.keras.metrics.MeanSquaredError()

    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        if DEBUG_PRINT:
            print(f"================== Episode {episode} =====================")
        step = agent.play_episode(policy = network_model_policy, policy_model= policy_model, experience_replay=True, discount_factor = discount_factor)
        total_loss = 0
        value_loss = 0
        G_t_mem = []
        # learning 
        for t in range(len(agent.state_history)-1):
            state_t = agent.state_history[t]
            action_t = agent.action_history[t]
            # calculate G_t
            G_t = tf.math.reduce_sum(
                    [agent.rewards_history[tt] * discount_factor ** tt for tt in range(t + 1, len(agent.state_history), 1)])
            G_t_mem.append(G_t)
            if DEBUG_PRINT:
                print(f"G_t: {G_t}")
            # value approximator training step
            value_loss += value_training_step(model = value_model, 
                                            states = tf.expand_dims(state_t,0), 
                                            targets=(tf.expand_dims(G_t,0)), 
                                            optimizer= value_optimizer, 
                                            metrics = value_metrics)

            # calculate baseline as current approximated state value
            baseline = value_model(tf.expand_dims(state_t,0))
            if DEBUG_PRINT:
                print(f"Value after: {baseline}")
            # delta = G_t - v(S,w)
            delta_t = G_t - baseline
            total_loss += policy_training_step(model = policy_model, states = tf.expand_dims(state_t,0), actions=tf.expand_dims(action_t,0), targets=tf.expand_dims(delta_t,0), optimizer=policy_optimizer )
            if DEBUG_PRINT:
                print(f"Action probs after: {policy_model(tf.expand_dims(state_t,0))} and action was: {action_t}")

        mean_loss = total_loss / len(agent.state_history)
        value_mean_loss = value_loss/len(agent.state_history)
        # Writing data to tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('policy mean loss', mean_loss, step = episode)
            tf.summary.scalar('value mean loss', value_mean_loss, step = episode)
            tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
            tf.summary.scalar('value mse', value_metrics.result(), step=episode)
        
        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        if not DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    # make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {CLASSIC_ENVIRONMENT_NAME}')
    # np.savetxt(file_name('baseline_classic\\reinforce_baseline_test_classic_nb', [value_learning_rate, policy_learning_rate], 'rewards', discount_factor=discount_factor, batch_size=1), rewards_mem, delimiter=',')

def reinforce_classic_no_batch(episodes_no = episodes_no, 
                                policy_learning_rate = policy_learning_rate,
                                value_learning_rate = value_learning_rate, 
                                discount_factor = DISCOUNT_FACTOR, 
                                batch_size = BATCH_SIZE):

    # Creation of file writer for tensorboard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{LOG_PATH}/classic_nb/lr{[value_learning_rate, policy_learning_rate]}_batch{BATCH_SIZE}_' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    env = gym.make(CLASSIC_ENVIRONMENT_NAME)
    agent = Agent(environment=env, start_state=env.reset())

    policy_model = dense_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)

    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        step = agent.play_episode(policy = network_model_policy, policy_model= policy_model, experience_replay=False)
        total_loss = 0
        G_t_mem = []

        for t in range(len(agent.state_history)-1):
            state_t = agent.state_history[t]
            action_t = agent.action_history[t]
            G_t = tf.math.reduce_sum(
                    [agent.rewards_history[tt] * discount_factor ** tt for tt in range(t + 1, len(agent.state_history), 1)])

            G_t_mem.append(G_t)
            total_loss += policy_training_step(model = policy_model, states = tf.expand_dims(state_t,0), actions=tf.expand_dims(action_t,0), targets=tf.expand_dims(G_t,0), optimizer=policy_optimizer )
            if DEBUG_PRINT:
                print(f"Action probs after: {policy_model(tf.expand_dims(state_t,0))} and action was: {action_t}, total_loss {total_loss}")

        mean_loss = total_loss / np.sum(G_t_mem)
        if DEBUG_PRINT:
            print(f"And mean loss: {mean_loss}")

        with train_summary_writer.as_default():
            tf.summary.scalar('policy mean loss', mean_loss, step = episode)
            tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)

        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        if not DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    # make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {CLASSIC_ENVIRONMENT_NAME}')
    np.savetxt(file_name('reinforce_classic\\reinforce_classic_nb', policy_learning_rate, 'rewards', discount_factor=discount_factor, batch_size=1), rewards_mem, delimiter=',')

def reinforce_baseline_atari(episodes_no = episodes_no, 
                            policy_learning_rate = policy_learning_rate, 
                            value_learning_rate = value_learning_rate, 
                            discount_factor = DISCOUNT_FACTOR, 
                            batch_size = BATCH_SIZE):
   
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{LOG_PATH}/baseline_atari/lr{[value_learning_rate, policy_learning_rate]}_batch{BATCH_SIZE}_' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    env = gym.make(ATARI_ENVIRONMENT_NAME, full_action_space = False)
    env = gym.wrappers.ResizeObservation(env, (80, 80))
    env = gym.wrappers.FrameStack(env, 4)    
    agent = Agent(environment=env, start_state=env.reset())

    value_model = conv_value_network(input_shape=env.observation_space.shape)
    policy_model = conv_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)

    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate = value_learning_rate)
    value_metrics = tf.keras.metrics.MeanSquaredError()

    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        step = agent.play_episode(policy = network_model_policy, policy_model= policy_model, experience_replay=True, discount_factor = discount_factor)
        print(f"Episode {episode+1}/{episodes_no} finished in {step} steps with reward {tf.reduce_sum(agent.rewards_history)}")
        if episode >= 0.05 * episodes_no:
            states, actions, targets, _ , _ = agent.replay_buffer.get_random_experience_batch(batch_size = batch_size)
            value_mean_loss = value_training_step(model = value_model, states = states, targets = targets, optimizer=value_optimizer, metrics=value_metrics)
            baselines = value_model(states)
            deltas = tf.math.subtract(targets, baselines)
            policy_mean_loss = policy_training_step(model = policy_model, states = states, actions = actions, targets = deltas, optimizer=policy_optimizer)
            with train_summary_writer.as_default():
                tf.summary.scalar('policy mean loss', tf.reduce_mean(policy_mean_loss), step = episode)
                tf.summary.scalar('value mean loss', tf.reduce_mean(value_mean_loss), step = episode)
                tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
                tf.summary.scalar('value mse', value_metrics.result(), step=episode)

        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        # if not DEBUG_PRINT:
        #     progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {CLASSIC_ENVIRONMENT_NAME}')
    np.savetxt(file_name('reinforce_baseline_test_classic', policy_learning_rate, 'rewards'), rewards_mem, delimiter=',')
    value_model.save_weights(f'value_{value_learning_rate}')
    policy_model.save_weights(f'policy_{policy_learning_rate}')

def reinforce_baseline_atari_nb(episodes_no = episodes_no,
                                value_learning_rate = value_learning_rate, 
                                policy_learning_rate = policy_learning_rate, 
                                discount_factor = DISCOUNT_FACTOR, 
                                batch_size = BATCH_SIZE):

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{LOG_PATH}/reinforce_baseline_atari' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    
    env = gym.make(ATARI_ENVIRONMENT_NAME, full_action_space = False)
    env = gym.wrappers.ResizeObservation(env, (80, 80))
    env = gym.wrappers.FrameStack(env, 4)    
    agent = Agent(environment=env, start_state=env.reset())

    value_model = conv_value_network(input_shape=env.observation_space.shape)
    policy_model = conv_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)

    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate = value_learning_rate)
    value_metrics = tf.keras.metrics.MeanSquaredError()

    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        step = agent.play_episode(policy = network_model_policy, policy_model= policy_model, experience_replay=False, discount_factor = discount_factor)
        print(f"Episode {episode} finished in {step} steps with reward {tf.reduce_sum(agent.rewards_history)}")
        total_loss = 0
        value_loss = 0
        G_t_mem = []
        for t in range(len(agent.state_history)-1):
            state_t = agent.state_history[t]
            action_t = agent.action_history[t]
            G_t = tf.math.reduce_sum(
                    [agent.rewards_history[tt] * discount_factor ** tt for tt in range(t + 1, len(agent.state_history), 1)])
            G_t_mem.append(G_t)
            value_loss+= value_training_step(model = value_model, 
                                            states = tf.expand_dims(state_t,0), 
                                            targets=(tf.expand_dims(G_t,0)), 
                                            optimizer= value_optimizer, 
                                            metrics = value_metrics)

            baseline = value_model(tf.expand_dims(state_t,0))
            if DEBUG_PRINT:
                print(f"Value after: {baseline}")
            delta_t = G_t - baseline
            total_loss += policy_training_step(model = policy_model, 
                                               states = tf.expand_dims(state_t,0), 
                                               actions=tf.expand_dims(action_t,0), 
                                               targets=tf.expand_dims(delta_t,0), 
                                               optimizer=policy_optimizer )
            if DEBUG_PRINT:
                print(f"Action probs after: {policy_model(tf.expand_dims(state_t,0))} and action was: {action_t}")

        mean_loss = total_loss / np.sum(G_t_mem)
        value_mean_loss = value_loss/ np.sum(G_t_mem)

        with train_summary_writer.as_default():
            tf.summary.scalar('policy mean loss', tf.reduce_mean(mean_loss), step = episode)
            tf.summary.scalar('value mean loss', tf.reduce_mean(value_mean_loss), step = episode)
            tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
            tf.summary.scalar('value mse', value_metrics.result(), step=episode)

        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        if not DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {ATARI_ENVIRONMENT_NAME}')
    np.savetxt(file_name('reinforce_baseline_test_atari', policy_learning_rate, 'rewards'), rewards_mem, delimiter=',')
    value_model.save_weights(f'value_{value_learning_rate}')
    policy_model.save_weights(f'policy_{policy_learning_rate}')

def actor_critic_classic_no_batch(episodes_no = episodes_no, 
                                    value_learning_rate = value_learning_rate, 
                                    policy_learning_rate = policy_learning_rate, 
                                    discount_factor = DISCOUNT_FACTOR, 
                                    batch_size = BATCH_SIZE):


    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{LOG_PATH}/actor_critic_nb' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    env = gym.make(CLASSIC_ENVIRONMENT_NAME)
    agent = Agent(environment=env, start_state=env.reset())

    # building networks
    value_model = dense_value_network(input_shape=env.observation_space.shape)
    policy_model = dense_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)

    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate = value_learning_rate)
    value_metrics = tf.keras.metrics.MeanSquaredError()

    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        if DEBUG_PRINT:
            print(f"================== Episode {episode} =====================")
        total_loss = 0
        total_value_loss = 0
        # play an episode while learning
        for step in range(STEPS_LIMIT):

            done = agent.take_action(policy=network_model_policy, policy_model=policy_model)
            state = agent.state_history[-1] # S
            action = agent.action_history[-1] # A taken in S
            reward = agent.rewards_history[-1] # R received after taking action A in S
             # if S' is not terminal
            if not done:
                # S' next state
                next_state = agent.state 
                # target = R + gamma * v(S',w)
                target = reward + discount_factor * value_model(tf.expand_dims(next_state,0))[0] 
            else:
                target = reward
            if DEBUG_PRINT:
                print(f"State: {state}, action {action}, reward {reward}")
                print(f"State value before: {value_model(tf.expand_dims(state, 0))}, action probs before: {policy_model(tf.expand_dims(state, 0))}")
            total_value_loss += value_training_step(model = value_model, 
                states = tf.expand_dims(state,0), 
                targets= tf.expand_dims(target,0), 
                optimizer = value_optimizer, 
                metrics = value_metrics
                )
            # delta = R + gamma * v(S',w) - v(S,w')
            delta = target - value_model(tf.expand_dims(state, 0))
            total_loss += policy_training_step(model = policy_model,
                states = tf.expand_dims(state,0),
                actions = tf.expand_dims(action,0),
                targets = tf.expand_dims(delta,0),
                optimizer = policy_optimizer   
                )
            if DEBUG_PRINT:
                print(f"State value after: {value_model(tf.expand_dims(state, 0))}, action probs before: {policy_model(tf.expand_dims(state, 0))}")
            if done: 
                break


        steps_mem.append(step)
        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        mean_loss = total_loss / np.sum(agent.rewards_history)
        value_mean_loss = total_value_loss/np.sum(agent.rewards_history)

        with train_summary_writer.as_default():
            tf.summary.scalar('policy mean loss', mean_loss, step = episode)
            tf.summary.scalar('value mean loss', value_mean_loss, step = episode)
            tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
            tf.summary.scalar('value mse', value_metrics.result(), step=episode)

        agent.reset_episode(state = env.reset())
        if not DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)
            
    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for actor critic- {CLASSIC_ENVIRONMENT_NAME}')
    np.savetxt(file_name('a2c\\actor_critic_nb', [policy_learning_rate, value_learning_rate], 'rewards', discount_factor=discount_factor, batch_size=1), rewards_mem, delimiter=',')


def actor_critic_classic(episodes_no = episodes_no, 
                        value_learning_rate = value_learning_rate, 
                        policy_learning_rate = policy_learning_rate, 
                        discount_factor = DISCOUNT_FACTOR, 
                        batch_size = BATCH_SIZE):


    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{LOG_PATH}/actor_critic' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    env = gym.make(CLASSIC_ENVIRONMENT_NAME)
    agent = Agent(environment=env, start_state=env.reset())

    # building networks
    value_model = dense_value_network(input_shape=env.observation_space.shape)
    policy_model = dense_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)

    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate = value_learning_rate)
    value_metrics = tf.keras.metrics.MeanSquaredError()

    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        if DEBUG_PRINT:
            print(f"================== Episode {episode} =====================")
        total_loss = 0
        total_value_loss = 0
        # play an episode while learning
        for step in range(STEPS_LIMIT):

            done = agent.take_action(policy=network_model_policy, policy_model=policy_model)
            state = agent.state_history[-1] # S
            action = agent.action_history[-1] # A taken in S
            reward = agent.rewards_history[-1] # R received after taking action A in S
            next_state = agent.state 
            # Adding to experience replay
            experience = (state, action, reward, next_state, done)
            agent.replay_buffer.add_experience(experience)

            if episode > 50:
                states, actions, rewards, next_states, dones = agent.replay_buffer.get_random_experience_batch(batch_size = batch_size)
                total_value_loss = a2c_value_training_step(model = value_model, 
                                                           states=states, 
                                                           rewards=rewards, 
                                                           next_states=next_states, 
                                                           dones = dones, 
                                                           optimizer= value_optimizer, 
                                                           metrics=value_metrics, 
                                                           discount_factor=discount_factor)
                if DEBUG_PRINT:
                    print(f"State: {state}, action {action}, reward {reward}")
                    print(f"State value before: {value_model(tf.expand_dims(states[0], 0))}, action probs before: {policy_model(tf.expand_dims(states[0], 0))}")
                targets = tf.add(tf.constant(rewards, dtype = 'float32'), tf.multiply(tf.multiply(tf.constant(discount_factor, dtype = 'float32'), value_model(next_states)),tf.constant(~dones,dtype='float32')))
                # delta = R + gamma * v(S',w) - v(S,w')
                deltas = tf.subtract(targets, value_model(states))
                total_loss += policy_training_step(model = policy_model,
                    states = states,
                    actions = actions,
                    targets = deltas,
                    optimizer = policy_optimizer   
                    )
                if DEBUG_PRINT:
                    print(f"Action probs after: {value_model(tf.expand_dims(states[0], 0))}, action probs before: {policy_model(tf.expand_dims(states[0], 0))}")
            if done: 
                break


        steps_mem.append(step)
        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        mean_loss = total_loss / (batch_size)*len(agent.rewards_history)
        value_mean_loss = total_value_loss/ batch_size*len(agent.rewards_history)

        with train_summary_writer.as_default():
            tf.summary.scalar('policy mean loss', mean_loss, step = episode)
            tf.summary.scalar('value mean loss', value_mean_loss, step = episode)
            tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
            tf.summary.scalar('value mse', value_metrics.result(), step=episode)

        agent.reset_episode(state = env.reset())
        if not DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)
            
    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for actor critic- {CLASSIC_ENVIRONMENT_NAME}')
    np.savetxt(file_name('a2c\\actor_critic', [policy_learning_rate, value_learning_rate], 'rewards', discount_factor=discount_factor, batch_size=batch_size), rewards_mem, delimiter=',')








