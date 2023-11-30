import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout

class UNetEncoder:
    def __init__(self, depth=3, initial_filters=8, kernel_size=(3, 3), activation=tf.nn.relu, use_batch_norm=True, dropout_rate=0.1):
        """
        Initializes the U-Net Encoder.

        Args:
        depth (int): Number of layers in the encoder.
        initial_filters (int): Number of filters in the first convolution layer.
        kernel_size (tuple): Size of the convolution kernel.
        activation (callable): Activation function used in convolution layers.
        use_batch_norm (bool): Flag to determine the use of batch normalization.
        dropout_rate (float): Dropout rate applied after pooling.
        """
        self.depth = depth
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

    def build(self, input_tensor):
        """
        Builds the encoder part of the U-Net architecture.

        Args:
        input_tensor (tf.Tensor): Input tensor for the encoder.

        Returns:
        list: List of output tensors from each encoder layer.
        """
        t = input_tensor
        encoder_layers = []

        for d in range(self.depth):
            filters = (2 ** d) * self.initial_filters
            t = Conv2D(filters=filters, kernel_size=self.kernel_size, padding="same", activation=self.activation)(t)
            if self.use_batch_norm:
                t = BatchNormalization()(t)
            t = Conv2D(filters=filters, kernel_size=self.kernel_size, padding="same", activation=self.activation)(t)
            if self.use_batch_norm:
                t = BatchNormalization()(t)
            
            encoder_layers.append(t)

            if d < (self.depth - 1):
                t = MaxPooling2D(pool_size=(2, 2), padding="same")(t)
                if self.dropout_rate > 0:
                    t = Dropout(rate=self.dropout_rate)(t)

        return encoder_layers
