import tensorflow as tf
import numpy as np
import config
from replay_buffer import ReplayBuffer
from util import *

class Agent:

    def __init__(self, environment, start_state):
        self.state = start_state
        self.environment = environment
        self.state_history = []
        self.action_history = []
        self.rewards_history = []
        self.replay_buffer = ReplayBuffer(max_length=config.REPLAY_BUFFER_LEN)

    def take_action(self, policy, policy_model):
        action = policy(action_space=self.environment.action_space, model = policy_model, state=self.state)
        next_state, reward, done, info = self.environment.step(action)
        if config.DEBUG_PRINT:
            print(f"Selected action: {action}, reward {reward}")
        self.state_history.append(self.state)
        self.state = next_state
        self.action_history.append(action)
        self.rewards_history.append(reward)
        return done

    def play_episode(self, policy, policy_model, steps_limit=100000, experience_replay = False, discount_factor = 1):
        done = False
        for step in range(steps_limit):
            done = self.take_action(policy=policy, policy_model=policy_model)
            if done:
                break
        if experience_replay:
            for t in range(len(self.state_history)):
                G_t = tf.math.reduce_sum([self.rewards_history[tt] * discount_factor ** tt for tt in range(t + 1,
                                                                                  len(self.state_history), 1)])
                if t < len(self.state_history) - 1: # state is not terminal
                    experience = (self.state_history[t], self.action_history[t], G_t, self.state_history[t+1], False)
                else: 
                    experience = (self.state_history[t], self.action_history[t], G_t, [], True)
                if config.DEBUG_PRINT:
                    print(f"Adding experience: {experience}")
                self.replay_buffer.add_experience(experience)
            
        return step
    
    def reset_episode(self, state):
        self.state = state
        self.state_history = []
        self.action_history = []
        self.rewards_history = []

