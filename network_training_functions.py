from cv2 import REDUCE_SUM
import tensorflow as tf
from losses import *
import config
import numpy as np

def policy_training_step(model, states, actions, targets, optimizer):
    """
    Training step for policy network, involving calculationa and application of gradients
    """
    with tf.GradientTape() as tape:
        output_values = model(states)
        # Select output for the performed action
        action_outputs = [output_values[i][actions[i]] for i in range(len(states))]
        loss = policy_loss(targets, action_outputs)
        if config.DEBUG_PRINT:
            print(f" |||| Policy |||| Targets: {targets}, outputs: {np.array(action_outputs)} and loss: {loss}")

    policy_grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(policy_grads, model.trainable_variables))
    return tf.reduce_sum(loss)

def value_training_step(model, states, targets, optimizer, metrics):
    """
    training step for value network for reinforce with baseline
    """
    with tf.GradientTape() as tape:
        output_values = model(states)
        # delta = G_t - v(S,w)
        deltas = targets - output_values
        loss = value_loss(deltas, output_values)
        if config.DEBUG_PRINT:
            print(f" |||| Value |||| Targets: {targets}, outputs: {output_values}, deltas: {deltas} and loss: {loss}")
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics.update_state(model(states), tf.constant(deltas))
    return tf.reduce_sum(loss)

def a2c_value_training_step(model, states, rewards, next_states, dones,  optimizer, metrics, discount_factor):
    """
    training step for value network in actor critic
    """
    with tf.GradientTape() as tape:
        # target = R + gamma * v(S',w) OR target = R if S' is terminal
        targets = tf.add(tf.constant(rewards, dtype = 'float32'), tf.multiply(tf.multiply(tf.constant(discount_factor, dtype = 'float32'), model(next_states)),tf.constant(~dones,dtype='float32')))
        output_values = model(states)
        # delta = target - v(S,w)
        deltas = targets - output_values
        loss = value_loss(deltas, output_values)
        if config.DEBUG_PRINT:
            print(f" |||| Value |||| Targets: {targets}, outputs: {output_values}, deltas: {deltas} and loss: {loss}")
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics.update_state(model(states), tf.constant(deltas))
    return tf.reduce_sum(loss)
