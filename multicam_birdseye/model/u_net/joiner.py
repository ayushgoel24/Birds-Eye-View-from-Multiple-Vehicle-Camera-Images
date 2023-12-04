import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, BatchNormalization

from transformer import TransformerNetwork

class UNetJoiner:
    def __init__(self, filters1=8, kernel_size=(3, 3), activation=tf.nn.relu, use_batch_norm=True, double_skip_connection=False):
        """
        Initializes the U-Net Joiner.

        Args:
        filters1 (int): Number of filters in the first convolution layer.
        kernel_size (tuple): Size of the convolutional kernel.
        activation (callable): Activation function used in convolution layers.
        use_batch_norm (bool): Flag to determine the use of batch normalization.
        double_skip_connection (bool): Whether to use double skip connections.
        """
        self.filters1 = filters1
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.double_skip_connection = double_skip_connection

    def build(self, list_of_encoder_layers, thetas):
        """
        Builds the joiner layers of the U-Net architecture.

        Args:
        list_of_encoder_layers (list): List of lists of encoder layers from multiple inputs.
        thetas (list): List of theta values for spatial transformation.

        Returns:
        list: Joined layers of the U-Net architecture.
        """
        n_inputs = len(list_of_encoder_layers)
        udepth = len(list_of_encoder_layers[0])
        joined_layers = []

        for d in range(udepth):
            filters = (2**d) * self.filters1
            warped_maps, nonwarped_maps = [], []

            for i in range(n_inputs):
                # Assuming SpatialTransformer is a defined layer or function
                t = TransformerNetwork().transform()(list_of_encoder_layers[i][d], thetas[i])
                warped_maps.append(t)
                if self.double_skip_connection:
                    nonwarped_maps.append(list_of_encoder_layers[i][d])

            t = Concatenate()(warped_maps) if n_inputs > 1 else warped_maps[0]
            t = self._add_conv_block(t, filters)

            if self.double_skip_connection:
                t_nonwarped = Concatenate()(nonwarped_maps) if n_inputs > 1 else nonwarped_maps[0]
                t_nonwarped = self._add_conv_block(t_nonwarped, filters)
                t = Concatenate()([t, t_nonwarped])
                t = self._add_conv_block(t, filters)

            joined_layers.append(t)

        return joined_layers

    def _add_conv_block(self, tensor, filters):
        """
        Adds a convolutional block with optional batch normalization to the tensor.

        Args:
        tensor (tf.Tensor): Input tensor.
        filters (int): Number of filters for the convolutional layers.

        Returns:
        tf.Tensor: Output tensor after applying the convolutional block.
        """
        t = Conv2D(filters=filters, kernel_size=self.kernel_size, padding="same", activation=self.activation)(tensor)
        if self.use_batch_norm:
            t = BatchNormalization()(t)
        t = Conv2D(filters=filters, kernel_size=self.kernel_size, padding="same", activation=self.activation)(t)
        return BatchNormalization()(t) if self.use_batch_norm else t