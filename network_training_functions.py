from cv2 import REDUCE_SUM
import tensorflow as tf
from losses import *
import config
import numpy as np

def policy_training_step(model, states, actions, targets, optimizer):
    with tf.GradientTape() as tape:
        output_values = model(states)
        action_outputs = [output_values[i][actions[i]] for i in range(len(states))]
        loss = policy_loss(targets, action_outputs)
        if config.DEBUG_PRINT:
            print(f" |||| Policy |||| Targets: {targets}, outputs: {np.array(action_outputs)} and loss: {loss}")

    policy_grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(policy_grads, model.trainable_variables))
    return tf.reduce_mean(loss)

def value_training_step(model, states, targets, optimizer, metrics):
    with tf.GradientTape() as tape:
        output_values = model(states)
        deltas = targets - output_values
        loss = value_loss(deltas, output_values)
        if config.DEBUG_PRINT:
            print(f" |||| Value |||| Targets: {targets}, outputs: {output_values}, deltas: {deltas} and loss: {loss}")
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics.update_state(model(states), tf.constant(deltas))
    return tf.reduce_mean(loss)

def policy_batch_training_step(model, states, actions, targets, optimizer):
    with tf.GradientTape() as tape:
        output_values = model(states)
        action_outputs = [output_values[i][actions[i]] for i in range(len(states))]
        loss = policy_loss(targets, action_outputs)
        if config.DEBUG_PRINT:
            print(f" |||| Policy |||| Targets: {targets}, outputs: {np.array(action_outputs)} and loss: {loss}")

    policy_grads = tape.gradient(loss, model.trainable_variables)

    total_loss = 0
    for i in range(len(states)):
        with tf.GradientTape() as tape2:
            
            output_values = model(states)
            action_outputs = [output_values[i][actions[i]] for i in range(len(states))]
            loss_single = policy_loss(tf.expand_dims(targets[i],0), tf.expand_dims(action_outputs[i],0))
            total_loss += loss_single
            if config.DEBUG_PRINT:
                print(f" |||| Policy |||| Targets: {targets[i]}, outputs: {action_outputs[i]} and loss: {loss_single}")
        # if i ==0:
        #     policy_grads_single = tape.gradient(loss_single, model.trainable_variables)
        # else:
        #     policy_grads_single += tape.gradient(loss_single, model.trainable_variables)
    single_grads = tape2.gradient(total_loss, model.trainable_variables)

    # optimizer.apply_gradients(zip(policy_grads, model.trainable_variables))
    return tf.reduce_mean(loss)