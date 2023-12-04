from datetime import datetime
import numpy as np
import os
import tensorflow

from ..utils import ImageOperationsUtil, MetricUtil, PathUtil

class Trainer:

    def __init__(self, training_input_paths, training_label_path, max_training_samples, 
                 validation_input_paths, validation_label_path, max_validation_samples, 
                 target_image_shape, input_one_hot_palette_path, label_one_hot_palette_path, model_path, 
                 homography_path, training_epochs, batch_size, learning_rate, class_loss_weights, 
                 early_stopping_patience, model_save_interval, output_directory, pretrained_model_weights):
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
        self.model_path = PathUtil.get_abs_path(model_path) # Python file defining the neural network
        self.homography_path = PathUtil.get_abs_path(homography_path) if homography_path else None # Python file defining a list H of homographies to be used in model
        self.training_epochs = training_epochs # number of epochs for training
        self.batch_size = batch_size # batch size for training
        self.learning_rate = learning_rate # learning rate of Adam optimizer for training
        self.class_loss_weights = class_loss_weights # factors for weighting classes differently in loss function
        self.early_stopping_patience = early_stopping_patience # patience for early-stopping due to converged validation mIoU
        self.model_save_interval = model_save_interval # epoch interval between exports of the model
        self.output_directory = PathUtil.get_abs_path(output_directory) # output dir for TensorBoard and models
        self.pretrained_model_weights = PathUtil.get_abs_path(pretrained_model_weights) if pretrained_model_weights else None # weights file of trained model for training continuation

        self.architecture = self._load_model_architecture(self.model)
        self._build_data_pipelines()
    
    def _load_model_architecture(self, model_path):
        # Load the network architecture module
        return PathUtil.load_module(model_path)
    
    def _prepare_training_data(self):
        num_inputs = len(self.training_input_paths)
        training_input_files = [PathUtil.get_files_in_folder(folder) for folder in self.training_input_paths]
        training_label_files = PathUtil.get_files_in_folder(self.training_label_path)

        _, indices = PathUtil.sample_list(training_label_files, n_samples=self.max_training_samples)
        sampled_training_inputs = [np.take(files, indices) for files in training_input_files]
        sampled_training_labels = np.take(training_label_files, indices)
        
        original_input_shape = PathUtil.load_image(sampled_training_inputs[0][0]).shape[0:2]
        original_label_shape = PathUtil.load_image(sampled_training_labels[0]).shape[0:2]
        print(f"Found {len(sampled_training_labels)} training samples")

    def _prepare_validation_data(self):
        validation_input_files = [PathUtil.get_files_in_folder(folder) for folder in self.validation_input_paths]
        validation_label_files = PathUtil.get_files_in_folder(self.validation_label_path)

        _, indices = PathUtil.sample_list(validation_label_files, n_samples=self.max_validation_samples)
        sampled_validation_inputs = [np.take(files, indices) for files in validation_input_files]
        sampled_validation_labels = np.take(validation_label_files, indices)
        print(f"Found {len(self.sampled_validation_labels)} validation samples")

    def _parse_one_hot_conversion(self):
        self.one_hot_palette_input = PathUtil.parse_convert_xml(self.one_hot_palette_input)
        self.one_hot_palette_label = PathUtil.parse_convert_xml(self.one_hot_palette_label)
        self.n_classes_input = len(self.one_hot_palette_input)
        self.n_classes_label = len(self.one_hot_palette_label)
    
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

    def build_data_pipelines(self):
        # Construct data pipelines for training and validation
        self._build_training_pipeline()
        self._build_validation_pipeline()

    def _build_training_pipeline(self, data_files, label_files ):
        dataTrain = tensorflow.data.Dataset.from_tensor_slices( (tuple(data_files), label_files) )
        dataTrain = dataTrain.shuffle(buffer_size=self.max_samples_training, reshuffle_each_iteration=True)
        dataTrain = dataTrain.map(self._parse_sample, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
        dataTrain = dataTrain.batch(self.batch_size, drop_remainder=True)
        dataTrain = dataTrain.repeat(self.epochs)
        dataTrain = dataTrain.prefetch(1)
        print("Built data pipeline for training")

    def _build_validation_pipeline(self, data_files, label_files ):
        dataValid = tensorflow.data.Dataset.from_tensor_slices( (tuple(data_files), label_files) )
        dataValid = dataValid.map(self._parse_sample, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
        dataValid = dataValid.batch(1)
        dataValid = dataValid.repeat(self.epochs)
        dataValid = dataValid.prefetch(1)
        print("Built data pipeline for validation")

    def build_model(self):
        if self.homographies:
            modelHomographies = PathUtil.load_module(self.homographies)
            self.model = self.architecture.get_network((self.image_shape[0], self.image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label, n_inputs=self.n_inputs, 
                                                       thetas=modelHomographies.H)
        else:
            self.model = self.architecture.get_network((self.image_shape[0], self.image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label)

        if self.model_weights:
            self.model.load_weights(self.model_weights)

        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = MetricUtil.weighted_categorical_crossentropy(self.loss_weights) if self.loss_weights else tensorflow.keras.losses.CategoricalCrossentropy()
        metrics = [ tensorflow.keras.metrics.CategoricalAccuracy(), MetricUtil.MeanIoUWithOneHotLabels(num_classes=self.n_classes_label) ]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"Compiled model {os.path.basename(self.model)}")

    def create_output_directories(self):
        model_output_dir = os.path.join(self.output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.tensorboard_dir = os.path.join(model_output_dir, "TensorBoard")
        self.checkpoint_dir = os.path.join(model_output_dir, "Checkpoints")

        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def create_callbacks(self):
        n_batches_train = len(self.files_train_label) // self.batch_size
        n_batches_valid = len(self.files_valid_label)

        tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(
            self.tensorboard_dir, 
            update_freq="epoch", 
            profile_batch=0
        )
        checkpoint_cb = tensorflow.keras.callbacks.ModelCheckpoint(
            os.path.join(self.checkpoint_dir, "e{epoch:03d}_weights.hdf5"), 
            save_freq=n_batches_train * self.save_interval, 
            save_weights_only=True
        )
        best_checkpoint_cb = tensorflow.keras.callbacks.ModelCheckpoint(
            os.path.join(self.checkpoint_dir, "best_weights.hdf5"), 
            save_best_only=True, 
            monitor="val_mean_io_u_with_one_hot_labels", 
            mode="max", save_weights_only=True
        )
        early_stopping_cb = tensorflow.keras.callbacks.EarlyStopping(
            monitor="val_mean_io_u_with_one_hot_labels",
            mode="max", 
            patience=self.early_stopping_patience, verbose=1
        )
        self.callbacks = [tensorboard_cb, checkpoint_cb, best_checkpoint_cb, early_stopping_cb]

    def run(self):
        self.model.fit(
            dataTrain,
            epochs=self.epochs, 
            steps_per_epoch=n_batches_train,
            validation_data=dataValid, 
            validation_freq=1,
            validation_steps=n_batches_valid,
            callbacks=self.callbacks
        )