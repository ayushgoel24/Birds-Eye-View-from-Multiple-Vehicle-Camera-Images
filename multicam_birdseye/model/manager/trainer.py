from datetime import datetime
import numpy as np
import os
import tensorflow

from .model_manager import ModelManager
from ...utils import ImageOperationsUtil, MetricUtil, PathUtil
from .common.callbacks import TrainerCallbacks
from .common.data_loader import DataLoader
class Trainer(ModelManager):

    def __init__(self, training_input_paths, training_label_path, max_training_samples, 
                 validation_input_paths, validation_label_path, max_validation_samples, 
                 target_image_shape, input_one_hot_palette_path, label_one_hot_palette_path, model_path, 
                 homography_path, training_epochs, batch_size, learning_rate, class_loss_weights, 
                 early_stopping_patience, model_save_interval, output_directory, pretrained_model_weights):
        """
        Initialize the NeuralNetworkTrainer class with training and validation data paths, model details, and training configurations.
        """

        super(Trainer, self).__init__(model_path=model_path)

        # Initialize all the parameters
        self.training_input_paths = [PathUtil.get_absolute_path(path) for path in training_input_paths] # directory/directories of input samples for training
        self.training_label_path = PathUtil.get_absolute_path(training_label_path) # directory of label samples for training
        self.max_training_samples = max_training_samples # maximum number of training samples
        self.validation_input_paths = [PathUtil.get_absolute_path(path) for path in validation_input_paths] # directory/directories of input samples for validation
        self.validation_label_path = PathUtil.get_absolute_path(validation_label_path) # directory of label samples for validation
        self.max_validation_samples = max_validation_samples # maximum number of validation samples
        self.target_image_shape = target_image_shape # image dimensions (HxW) of inputs and labels for network
        self.input_one_hot_palette_path = PathUtil.get_abs_path(input_one_hot_palette_path) # xml-file for one-hot-conversion of input images
        self.label_one_hot_palette_path = PathUtil.get_abs_path(label_one_hot_palette_path) # xml-file for one-hot-conversion of label images
        # self.model_path = PathUtil.get_abs_path(model_path) # Python file defining the neural network
        self.homography_path = PathUtil.get_abs_path(homography_path) if homography_path else None # Python file defining a list H of homographies to be used in model
        self.training_epochs = training_epochs # number of epochs for training
        self.batch_size = batch_size # batch size for training
        self.learning_rate = learning_rate # learning rate of Adam optimizer for training
        self.class_loss_weights = class_loss_weights # factors for weighting classes differently in loss function
        self.early_stopping_patience = early_stopping_patience # patience for early-stopping due to converged validation mIoU
        self.model_save_interval = model_save_interval # epoch interval between exports of the model
        self.output_directory = PathUtil.get_abs_path(output_directory) # output dir for TensorBoard and models
        self.pretrained_model_weights = PathUtil.get_abs_path(pretrained_model_weights) if pretrained_model_weights else None # weights file of trained model for training continuation

        
        self.training_input_files, self.training_label_files = DataLoader.prepare_training_data()
        self.validation_input_files, self.validation_label_files = DataLoader.prepare_validation_data()
        self._parse_one_hot_conversion()

        self._build_training_pipeline(self.training_input_files, self.training_label_files)
        self._build_validation_pipeline(self.validation_input_files, self.validation_label_files)

        self._build_model()
        self._create_output_directories()
        self._create_callbacks()
    
    # TODO: remove this method once the ModelManager._parse_sample is corrected
    def _parse_sample(self, input_files, label_file):
        """
        Parses and processes input and label images.

        Args:
        input_files (list): List of file paths for the input images.
        label_file (str): File path for the label image.

        Returns:
        Tuple: Processed input images and label image.
        """
        inputs = []
        for inp_file in input_files:
            inp = ImageOperationsUtil.load_image_op(inp_file)
            inp = ImageOperationsUtil.resize_image_op(inp, self.image_shape_original_input, self.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            inp = ImageOperationsUtil.one_hot_encode_image_op(inp, self.one_hot_palette_input)
            inputs.append(inp)
        inputs = inputs[0] if self.n_inputs == 1 else tuple(inputs)

        label = ImageOperationsUtil.load_image_op(label_file)
        label = ImageOperationsUtil.resize_image_op(label, self.image_shape_original_label, self.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label = ImageOperationsUtil.one_hot_encode_image_op(label, self.one_hot_palette_label)

        return inputs, label

    def _build_training_pipeline(self, data_files, label_files):
        training_data = tensorflow.data.Dataset.from_tensor_slices( (tuple(data_files), label_files) )
        training_data = training_data.shuffle(buffer_size=self.max_training_samples, reshuffle_each_iteration=True)
        training_data = training_data.map(self._parse_sample, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
        training_data = training_data.batch(self.batch_size, drop_remainder=True)
        training_data = training_data.repeat(self.training_epochs)
        self.training_data = training_data.prefetch(1)
        print("Built data pipeline for training")

    def _build_validation_pipeline(self, data_files, label_files):
        validation_label = tensorflow.data.Dataset.from_tensor_slices( (tuple(data_files), label_files) )
        validation_label = validation_label.map(self._parse_sample, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
        validation_label = validation_label.batch(1)
        validation_label = validation_label.repeat(self.epochs)
        self.validation_label = validation_label.prefetch(1)
        print("Built data pipeline for validation")

    def run(self):

        self.model.fit(
            self.training_data,
            epochs=self.training_epochs, 
            steps_per_epoch=len(self.training_label_files) // self.batch_size,
            validation_data=self.validation_label, 
            validation_freq=1,
            validation_steps=len(self.validation_label_files),
            callbacks=self.callbacks
        )