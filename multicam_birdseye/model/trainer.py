from datetime import datetime
import numpy as np
import os
import tensorflow

from ..utils import PathUtils

class Trainer:

    def __init__(self, input_training, label_training, max_samples_training, input_validation, label_validation, 
                 max_samples_validation, image_shape, one_hot_palette_input, one_hot_palette_label, model, 
                 homographies, epochs, batch_size, learning_rate, loss_weights, 
                 early_stopping_patience, save_interval, output_dir, model_weights):
        # Initialize all the parameters
        self.training_data = [PathUtils.get_abs_path(path) for path in input_training]
        self.training_labels = PathUtils.get_abs_path(label_training)
        self.max_samples_training = max_samples_training
        self.validation_data = [PathUtils.get_abs_path(path) for path in input_validation]
        self.validation_labels = PathUtils.get_abs_path(label_validation)
        self.max_samples_validation = max_samples_validation
        self.image_shape = image_shape
        self.one_hot_palette_input = PathUtils.get_abs_path(one_hot_palette_input)
        self.one_hot_palette_label = PathUtils.get_abs_path(one_hot_palette_label)
        self.model = PathUtils.get_abs_path(model)
        # Python file defining a list H of homographies to be used in model
        self.homographies = PathUtils.get_abs_path(homographies) if homographies else None
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.early_stopping_patience = early_stopping_patience
        self.save_interval = save_interval
        self.output_dir = PathUtils.get_abs_path(output_dir)
        self.model_weights = PathUtils.get_abs_path(model_weights) if model_weights else None

        self.architecture = self._load_architecture(self.model)
        self._build_data_pipelines()
    
    def _load_architecture(self, model):
        # Load the network architecture module
        return PathUtils.load_module(model)
    
    def _sample_training_data(self):
        n_inputs = len(self.training_data)
        files_train_input = [PathUtils.get_files_in_folder(folder) for folder in self.training_data]
        files_train_label = PathUtils.get_files_in_folder(self.training_labels)

        _, idcs = PathUtils.sample_list(files_train_label, n_samples=self.max_samples_training)
        self.files_train_input = [np.take(f, idcs) for f in files_train_input]
        self.files_train_label = np.take(files_train_label, idcs)
        
        self.image_shape_original_input = PathUtils.load_image(self.files_train_input[0][0]).shape[0:2]
        self.image_shape_original_label = PathUtils.load_image(self.files_train_label[0]).shape[0:2]
        print(f"Found {len(self.files_train_label)} training samples")

    def _sample_validation_data(self):
        files_valid_input = [PathUtils.get_files_in_folder(folder) for folder in self.validation_data]
        files_valid_label = PathUtils.get_files_in_folder(self.validation_labels)

        _, idcs = PathUtils.sample_list(files_valid_label, n_samples=self.max_samples_validation)
        self.files_valid_input = [np.take(f, idcs) for f in files_valid_input]
        self.files_valid_label = np.take(files_valid_label, idcs)
        print(f"Found {len(self.files_valid_label)} validation samples")

    def _parse_one_hot_conversion(self):
        self.one_hot_palette_input = PathUtils.parse_convert_xml(self.one_hot_palette_input)
        self.one_hot_palette_label = PathUtils.parse_convert_xml(self.one_hot_palette_label)
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
            inp = utils.load_image_op(inp_file)
            inp = utils.resize_image_op(inp, self.image_shape_original_input, self.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            inp = utils.one_hot_encode_image_op(inp, self.one_hot_palette_input)
            inputs.append(inp)
        inputs = inputs[0] if self.n_inputs == 1 else tuple(inputs)

        label = utils.load_image_op(label_file)
        label = utils.resize_image_op(label, self.image_shape_original_label, self.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label = utils.one_hot_encode_image_op(label, self.one_hot_palette_label)

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
            modelHomographies = PathUtils.load_module(self.homographies)
            self.model = self.architecture.get_network((self.image_shape[0], self.image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label, n_inputs=self.n_inputs, 
                                                       thetas=modelHomographies.H)
        else:
            self.model = self.architecture.get_network((self.image_shape[0], self.image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label)

        if self.model_weights:
            self.model.load_weights(self.model_weights)

        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = utils.weighted_categorical_crossentropy(self.loss_weights) if self.loss_weights else tensorflow.keras.losses.CategoricalCrossentropy()
        metrics = [ tensorflow.keras.metrics.CategoricalAccuracy(), utils.MeanIoUWithOneHotLabels(num_classes=self.n_classes_label) ]
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

    def run():
        self.model.fit(
            dataTrain,
            epochs=self.epochs, 
            steps_per_epoch=n_batches_train,
            validation_data=dataValid, 
            validation_freq=1,
            validation_steps=n_batches_valid,
            callbacks=self.callbacks
        )