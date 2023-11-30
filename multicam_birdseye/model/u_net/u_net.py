import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.models import Model

from .decoder import UNetDecoder
from .encoder import UNetEncoder
from .joiner import UNetJoiner

# Assuming Encoder, Joiner, and Decoder classes are defined as discussed earlier

class UNetNetwork:
    def __init__(self, input_shape, n_output_channels, n_inputs, thetas, 
                 udepth=5, filters1=16, kernel_size=(3,3), 
                 activation=tf.nn.relu, batch_norm=True, 
                 dropout=0.1, double_skip_connection=False):
        """
        Initializes the U-Net-like Network.

        Args:
        input_shape (tuple): Shape of the input images.
        n_output_channels (int): Number of output channels.
        n_inputs (int): Number of input images.
        thetas (list): List of theta values for spatial transformation.
        udepth (int): Depth of the U-Net encoder and decoder.
        filters1 (int): Number of filters in the first convolution layer.
        kernel_size (tuple): Size of the convolutional kernel.
        activation (callable): Activation function used in convolution layers.
        batch_norm (bool): Flag to determine the use of batch normalization.
        dropout (float): Dropout rate.
        double_skip_connection (bool): Whether to use double skip connections.
        """
        self.input_shape = input_shape
        self.n_output_channels = n_output_channels
        self.n_inputs = n_inputs
        self.thetas = thetas
        self.udepth = udepth
        self.filters1 = filters1
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.double_skip_connection = double_skip_connection

    def build(self):
        """
        Builds the U-Net-like network.

        Returns:
        tf.keras.Model: The complete U-Net-like model.
        """
        inputs = [Input(self.input_shape) for _ in range(self.n_inputs)]

        # Create encoder instances and build encoder layers for each input
        list_of_encoder_layers = []
        for input_tensor in inputs:
            encoder = UNetEncoder(self.udepth, self.filters1, self.kernel_size, self.activation, self.batch_norm, self.dropout)
            encoder_layers = encoder.build(input_tensor)
            list_of_encoder_layers.append(encoder_layers)

        # Create joiner instance and fuse encodings of all inputs at all layers
        joiner = UNetJoiner(self.filters1, self.kernel_size, self.activation, self.batch_norm, self.double_skip_connection)
        encoder_layers = joiner.build(list_of_encoder_layers, self.thetas)

        # Create decoder instance and decode from bottom to top layer
        decoder = UNetDecoder(self.udepth, self.filters1, self.kernel_size, self.activation, self.batch_norm, self.dropout)
        reconstruction = decoder.build(encoder_layers)

        # Build final prediction layer
        prediction = Conv2D(filters=self.n_output_channels, kernel_size=self.kernel_size, padding="same", activation=self.activation)(reconstruction)
        prediction = Activation("softmax")(prediction)

        return Model(inputs, prediction)
