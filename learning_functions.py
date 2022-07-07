from agent import Agent
import config
from policies import *
from network_builders import *
from network_training_functions import *
from util import *
import gym

def reinforce_classic(episodes_no = config.episodes_no,
    policy_learning_rate = config.policy_learning_rate,
    discount_factor = config.DISCOUNT_FACTOR, 
    batch_size = config.BATCH_SIZE):

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{config.LOG_PATH}/reinforce_classic' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    env = gym.make(config.CLASSIC_ENVIRONMENT_NAME)
    agent = Agent(environment=env, start_state=env.reset())
    policy_model = dense_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)
    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        if config.DEBUG_PRINT:
            print(f"================== Episode {episode} =====================")
        step = agent.play_episode(policy = network_model_policy, policy_model= policy_model, experience_replay=True, discount_factor = discount_factor)
        if episode >= 0 * episodes_no:
            # states, actions, targets, _ , _ = agent.replay_buffer.get_random_experience_batch(batch_size = batch_size)
            states, actions, targets, _, _ = agent.replay_buffer.get_ordered_experience_batch(batch_size = step)
            agent.replay_buffer.clear()
            if config.DEBUG_PRINT:
                print(f"Action probs before: {policy_model(tf.expand_dims(states[0],0))} and action was: {actions[0]}")
            policy_mean_loss = policy_batch_training_step(model = policy_model, states = states, actions = actions, targets = targets, optimizer=policy_optimizer)
            if config.DEBUG_PRINT:
                print(f"Action probs after: {policy_model(tf.expand_dims(states[0],0))} and action was: {actions[0]}, mean loss {policy_mean_loss}")
            with train_summary_writer.as_default():
                tf.summary.scalar('policy mean loss', tf.reduce_mean(policy_mean_loss), step = episode)
                tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        if not config.DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce - {config.CLASSIC_ENVIRONMENT_NAME}')
    np.savetxt(file_name('reinforce_test_classic', policy_learning_rate, 'rewards'), rewards_mem, delimiter=',')



def reinforce_baseline_classic(episodes_no = config.episodes_no, policy_learning_rate = config.policy_learning_rate, value_learning_rate = config.value_learning_rate, discount_factor = config.DISCOUNT_FACTOR, batch_size = config.BATCH_SIZE):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/local/reinforce_baseline_classic' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    env = gym.make(config.CLASSIC_ENVIRONMENT_NAME)
    agent = Agent(environment=env, start_state=env.reset())

    value_model = dense_value_network(input_shape=env.observation_space.shape)
    policy_model = dense_policy_network(input_shape = env.observation_space.shape, outputs_no = env.action_space.n)

    policy_optimizer = tf.keras.optimizers.Adam(learning_rate = policy_learning_rate)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate = value_learning_rate)
    value_metrics = tf.keras.metrics.MeanSquaredError()

    rewards_mem = []
    steps_mem = []

    for episode in range(episodes_no):
        step = agent.play_episode(policy = network_model_policy, policy_model= policy_model, experience_replay=True, discount_factor = discount_factor)
        if episode > 0.1 * episodes_no:
            states, actions, targets, _ , _ = agent.replay_buffer.get_ordered_experience_batch(batch_size = step)
            agent.replay_buffer.clear()
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
        if not config.DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {config.CLASSIC_ENVIRONMENT_NAME}')
    np.savetxt(file_name('reinforce_baseline_test_classic', policy_learning_rate, 'rewards'), rewards_mem, delimiter=',')


def reinforce_baseline_classic_no_batch(episodes_no = config.episodes_no, 
                                        value_learning_rate = config.value_learning_rate, 
                                        policy_learning_rate = config.policy_learning_rate, 
                                        discount_factor = config.DISCOUNT_FACTOR, 
                                        batch_size = config.BATCH_SIZE):

    # For tensorflow
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{config.LOG_PATH}/reinforce_baseline_classic_nb' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    env = gym.make(config.CLASSIC_ENVIRONMENT_NAME)
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
        if config.DEBUG_PRINT:
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
            if config.DEBUG_PRINT:
                print(f"G_t: {G_t}")
            # value approximator training step
            value_loss += value_training_step(model = value_model, 
                                            states = tf.expand_dims(state_t,0), 
                                            targets=(tf.expand_dims(G_t,0)), 
                                            optimizer= value_optimizer, 
                                            metrics = value_metrics)

            # calculate baseline as current approximated state value
            baseline = value_model(tf.expand_dims(state_t,0))
            if config.DEBUG_PRINT:
                print(f"Value after: {baseline}")
            
            delta_t = G_t - baseline
            total_loss += policy_training_step(model = policy_model, states = tf.expand_dims(state_t,0), actions=tf.expand_dims(action_t,0), targets=tf.expand_dims(delta_t,0), optimizer=policy_optimizer )
            if config.DEBUG_PRINT:
                print(f"Action probs after: {policy_model(tf.expand_dims(state_t,0))} and action was: {action_t}")

        mean_loss = total_loss / np.sum(G_t_mem)
        value_mean_loss = value_loss/np.sum(G_t_mem)

        with train_summary_writer.as_default():
            tf.summary.scalar('policy mean loss', mean_loss, step = episode)
            tf.summary.scalar('value mean loss', value_mean_loss, step = episode)
            tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)
            tf.summary.scalar('value mse', value_metrics.result(), step=episode)
        
        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        if not config.DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {config.CLASSIC_ENVIRONMENT_NAME}')
    np.savetxt(file_name('reinforce_baseline_test_classic_nb', policy_learning_rate, 'rewards'), rewards_mem, delimiter=',')

def reinforce_classic_no_batch(episodes_no = config.episodes_no, 
                                policy_learning_rate = config.policy_learning_rate, 
                                discount_factor = config.DISCOUNT_FACTOR, 
                                batch_size = config.BATCH_SIZE):

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{config.LOG_PATH}/reinforce_classic_nb' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    env = gym.make(config.CLASSIC_ENVIRONMENT_NAME)
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
            if config.DEBUG_PRINT:
                print(f"Action probs after: {policy_model(tf.expand_dims(state_t,0))} and action was: {action_t}, total_loss {total_loss}")

        mean_loss = total_loss / np.sum(G_t_mem)
        if config.DEBUG_PRINT:
            print(f"And mean loss: {mean_loss}")

        with train_summary_writer.as_default():
            tf.summary.scalar('policy mean loss', mean_loss, step = episode)
            tf.summary.scalar('cumulative reward', tf.reduce_sum(agent.rewards_history), step = episode)

        rewards_mem.append(tf.reduce_sum(agent.rewards_history))
        steps_mem.append(step)
        agent.reset_episode(state = env.reset())
        if not config.DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {config.CLASSIC_ENVIRONMENT_NAME}')
    np.savetxt(file_name('reinforce_classic_nb', policy_learning_rate, 'rewards'), rewards_mem, delimiter=',')


def reinforce_baseline_atari(episodes_no = config.episodes_no,
                            value_learning_rate = config.value_learning_rate, 
                            policy_learning_rate = config.policy_learning_rate, 
                            discount_factor = config.DISCOUNT_FACTOR, 
                            batch_size = config.BATCH_SIZE):

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{config.LOG_PATH}/reinforce_baseline_atari' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    
    env = gym.make(config.ATARI_ENVIRONMENT_NAME, full_action_space = False)
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
            if config.DEBUG_PRINT:
                print(f"Value after: {baseline}")
            delta_t = G_t - baseline
            total_loss += policy_training_step(model = policy_model, 
                                               states = tf.expand_dims(state_t,0), 
                                               actions=tf.expand_dims(action_t,0), 
                                               targets=tf.expand_dims(delta_t,0), 
                                               optimizer=policy_optimizer )
            if config.DEBUG_PRINT:
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
        if not config.DEBUG_PRINT:
            progress_bar(episode, episodes_no, rewards_mem[-1], 80)

    env.close() 
    make_plot(range(1, episodes_no + 1, 1), rewards_mem, 'Episode', 'Total reward', f'Cumulative reward for reinforce baseline - {config.ATARI_ENVIRONMENT_NAME}')
    np.savetxt(file_name('reinforce_baseline_test_atari', policy_learning_rate, 'rewards'), rewards_mem, delimiter=',')

