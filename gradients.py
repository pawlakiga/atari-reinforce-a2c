import tensorflow as tf
from sympy.core import function
from keras.models import Model
import numpy as np
import config
import keras.backend as K

def policy_grads(model: Model, loss_function: function, compute_gradient_arguments):
    state = compute_gradient_arguments[0][0]
    action = compute_gradient_arguments[0][1]
    target = compute_gradient_arguments[1]
    with tf.GradientTape() as tape:
        output_values = model(tf.expand_dims(state,0))
        action_output = output_values[0][action]
        loss = loss_function(target, action_output)
        if config.DEBUG_PRINT:
            print("--- In policy gradient ---")
            # print(f"Input: state {state} and action {action}, action prob: {action_output}, target: {target}, loss: {loss}")
            print(f"Action {action}, action prob: {action_output}, target: {target}, loss: {loss}")

            debug_output_fun = K.function([model.input], [model.output])
            print(f"Debug output values: {debug_output_fun(tf.expand_dims(state,0))}")
            debug_dense_fun = K.function([model.input], [model.layers[-2].output])
            print(f"Debug dense output value: {debug_dense_fun(tf.expand_dims(state,0))}")

    grads = tape.gradient(loss, model.trainable_variables)
    # print(f"Grads: {grads}")
    return grads, loss


def value_function_grads(model: Model, loss_function: function, compute_gradient_arguments):
    state = compute_gradient_arguments[0]
    G_t = compute_gradient_arguments[1]
    with tf.GradientTape() as tape:
        output_value = model(tf.expand_dims(state,0))
        delta = G_t - output_value[0]
        loss = loss_function(delta, output_value)
        if config.DEBUG_PRINT:
            print("--- In value gradient ---")
            # print(f"Input: {state},output: {output_value}, target: {G_t}, delta: {delta}, loss: {loss}")
            print(f"Input: {state},output: {output_value}, target: {G_t}, delta: {delta}, loss: {loss}")
            
    grads = tape.gradient(loss, model.trainable_variables)
    # grads_cpy = [grad[0].numpy() for grad in grads[0]]
    # print(f"Grads: {grads_cpy}, {grads[1]}")
    return grads, loss


def batch_policy_grads(model: Model, loss_function: function, compute_gradient_arguments):
    states = compute_gradient_arguments[0][0]
    actions = compute_gradient_arguments[0][1]
    targets = compute_gradient_arguments[1]
    with tf.GradientTape() as tape:
        output_values = model(states)
        action_outputs = [output_values[0][action] for action in actions]
        loss = loss_function(targets, action_outputs)
        if config.DEBUG_PRINT:
            print("--- In policy gradient ---")
            print(f"Loss: {model.losses}")
            # print(f"Input: state {state} and action {action}, action prob: {action_output}, target: {target}, loss: {loss}")
            # debug_output_fun = K.function([model.input], [model.output])
            # print(f"Debug output values: {debug_output_fun(state[np.newaxis])}")
            # debug_dense_fun = K.function([model.input], [model.layers[-2].output])
            # print(f"Debug dense output value: {debug_dense_fun(state[np.newaxis])}")
            
    grads = tape.gradient(loss, model.trainable_variables)

    # print(f"Grads: {grads}")
    return grads, loss

# def value_function_grads(model: Model, loss_function: function, compute_gradient_arguments):
#     state = compute_gradient_arguments[0]
#     G_t = compute_gradient_arguments[1]
#     with tf.GradientTape() as tape:
#         output_value = model(state)
#         delta = G_t - output_value[0]
#         loss = loss_function(delta, output_value)
#         if config.DEBUG_PRINT:
#             print("--- In value gradient ---")
#             print(f"Input: {state},output: {output_value}, target: {G_t}, delta: {delta}, loss: {loss}")
#     grads = tape.gradient(loss, model.trainable_variables)
#     return grads, loss