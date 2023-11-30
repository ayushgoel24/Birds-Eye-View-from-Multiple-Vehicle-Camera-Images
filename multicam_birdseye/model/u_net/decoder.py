import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Dropout, Conv2D, BatchNormalization

class UNetDecoder:
    def __init__(self, depth=3, initial_filters=8, kernel_size=(3, 3), activation=tf.nn.relu, use_batch_norm=True, dropout_rate=0.1):
        """
        Initializes the U-Net Decoder.

        Args:
        depth (int): Number of layers in the decoder.
        initial_filters (int): Number of filters in the first convolutional layer of the decoder.
        kernel_size (tuple): Size of the convolutional kernel.
        activation (callable): Activation function used in convolution layers.
        use_batch_norm (bool): Flag to determine the use of batch normalization.
        dropout_rate (float): Dropout rate applied in the decoder layers.
        """
        self.depth = depth
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

    def build(self, encoder_layers):
        """
        Builds the decoder part of the U-Net architecture.

        Args:
        encoder_layers (list): List of layers from the encoder part of the U-Net.

        Returns:
        tf.Tensor: The final tensor output of the decoder.
        """
        t = encoder_layers[self.depth - 1]

        for d in reversed(range(self.depth - 1)):
            filters = (2 ** d) * self.initial_filters
            t = Conv2DTranspose(filters=filters, kernel_size=self.kernel_size, strides=(2, 2), padding="same")(t)
            t = Concatenate()([encoder_layers[d], t])

            if self.dropout_rate > 0:
                t = Dropout(rate=self.dropout_rate)(t)

            t = Conv2D(filters=filters, kernel_size=self.kernel_size, padding="same", activation=self.activation)(t)

            if self.use_batch_norm:
                t = BatchNormalization()(t)

            t = Conv2D(filters=filters, kernel_size=self.kernel_size, padding="same", activation=self.activation)(t)

            if self.use_batch_norm:
                t = BatchNormalization()(t)

        return t