import tensorflow as tf


def policy_loss(target, output_value):
    return tf.multiply(-target, tf.math.log(output_value))


def value_loss(target, output_value):
    # return tf.multiply(-target, output_value)
    return tf.keras.losses.mean_squared_error(target, output_value)

def value_mse(target, output_value):
    return tf.math.squared_difference(target, output_value)
