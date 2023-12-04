import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Activation, MaxPooling2D, Flatten, Conv2D, Dense, Multiply
from tensorflow.keras.models import Model

from .interpolation import Interpolation

class TransformerNetwork:

    def __init__(self) -> None:
        pass

    def _transform(self, input, output_shape, theta_init=np.eye(3), theta_const=False, loc_downsample=3, dense_units=20, filters=16, kernel_size=(3,3), activation=tf.nn.relu, dense_reg=0.0):

        theta_init = theta_init.flatten().astype(np.float32)

        if not theta_const:

            t = input

            # initialize transform to identity
            init_weights = [np.zeros((dense_units, 9), dtype=np.float32), theta_init]

            # localization network
            for d in range(loc_downsample):
                t = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(t)
                t = MaxPooling2D(pool_size=(2,2), padding="same")(t)
            t = Flatten()(t)
            t = Dense(dense_units)(t)

            k_reg = tf.keras.regularizers.l2(dense_reg) if dense_reg > 0 else None
            b_reg = tf.keras.regularizers.l2(dense_reg) if dense_reg > 0 else None
            theta = Dense(9, weights=init_weights, kernel_regularizer=k_reg, bias_regularizer=b_reg)(t) # transformation parameters

        else:

            theta = tf.tile(theta_init, tf.shape(input)[0:1])

        # transform feature map
        output = Interpolation(output_shape)([input, theta])

        return output

    def transform(self, input_shape, output_shape, theta_init=np.eye(3), theta_const=False, loc_downsample=3, dense_units=20, filters=16, kernel_size=(3,3), activation=tf.nn.relu, dense_reg=0.0):
        input = Input(input_shape)
        output = self._transform(input, output_shape[0:2], theta_init, theta_const, loc_downsample, dense_units, filters, kernel_size, activation, dense_reg)

        return Model(input, output)
