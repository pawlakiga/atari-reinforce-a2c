import numpy as np
import tensorflow as tf
# from tensorflow.keras import layers
from keras.models import Model


def dense_policy_network(input_shape, outputs_no):
    x = tf.keras.layers.Input(shape = input_shape)
    h = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(x)
    # h = layers.Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(h)
    # h = layers.Dense(32, activation='relu')(h)
    h = tf.keras.layers.Dense(outputs_no, kernel_initializer='random_normal', bias_initializer='zeros')(h)
    y = tf.keras.layers.Softmax()(h)
    model = Model(inputs=x,outputs=y)
    return model


def conv_policy_network(input_shape, outputs_no):
    x = tf.keras.layers.Input(shape = input_shape)
    h = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32)/255.)(x) 
    h = tf.keras.layers.Conv2D(filters = 64, kernel_size = (4,4), padding='same', activation = 'relu')(h)
    h = tf.keras.layers.Conv2D(filters = 64, kernel_size = (8,8), padding='same',  activation = 'relu')(h)
    # h = layers.Conv2D(filters = 64, kernel_size = (3,3), padding='same', activation='relu')(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(128, activation='relu')(h)
    # h = layers.Dense(32, activation='relu')(h)
    h = tf.keras.layers.Dense(outputs_no)(h)
    y = tf.keras.layers.Softmax()(h)
    model = Model(inputs = x, outputs = y)
    return model


def dense_value_network(input_shape):
    x = tf.keras.layers.Input(shape = input_shape)
    h = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='random_normal')(x)
    # h = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(h)
    # h = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(h)
    y = tf.keras.layers.Dense(1)(h)
    model = Model(inputs = x, outputs = y)
    return model


def conv_value_network(input_shape):
    x = tf.keras.layers.Input(shape = input_shape)
    h = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32)/255.)(x)
    h = tf.keras.layers.Conv2D(filters = 64, kernel_size = (4,4), padding='same')(h)
    h = tf.keras.layers.Conv2D(filters = 64, kernel_size = (8,8), padding='same')(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(128, activation='relu')(h)
    # h = tf.keras.layers.Dense(32, activation='relu')(h)
    # h = tf.keras.layers.Dense(32, activation='relu')(h)
    y = tf.keras.layers.Dense(1)(h)
    model = Model(inputs = x, outputs = y)
    return model
