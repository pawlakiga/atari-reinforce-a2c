import time
import gym 
import flappy_bird_gym
import numpy as np
import tensorflow as tf

env = gym.make("Breakout-v4")

state = env.reset()
print(state.shape)
grayscale = tf.image.rgb_to_grayscale(state)
print(grayscale.shape)

# for ep in range(5):
#     env.reset()
#     for step in range(1000):
#         action = np.random.choice(env.action_space.n, p=np.squeeze([0.93 , 0.07]))

#         print(f"Action {action} ")
#         next_state, reward, done, info = env.step(action)
#         if done :
#             break
#         env.render()
#         time.sleep(1/30)
        
# env.close()