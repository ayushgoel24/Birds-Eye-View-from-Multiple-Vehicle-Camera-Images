from datetime import datetime
import os
import tensorflow

from ...utils import ImageOperationsUtil, MetricUtil, PathUtil

class ModelManager:
    def __init__(self, model_path, image_shape, model_weights=None):
        """
        Initializes the ModelManager with common properties used by Trainer, Evaluator, and Predictor.

        Args:
        model_path (str): Path to the Python file defining the neural network model.
        image_shape (tuple): Image dimensions (HxW) expected by the model.
        model_weights (str, optional): Path to the weights file of the trained model.
        """
        self.model_path = model_path
        self.image_shape = image_shape
        self.model_weights = model_weights
        
        self.architecture = self._load_model_architecture(model_path)

    def _load_model_architecture(self, model_path):
        # Load the network architecture module
        return PathUtil.load_module(model_path)
    
    def _parse_one_hot_conversion(self, one_hot_palette_input, one_hot_palette_label):
        self.one_hot_palette_input = PathUtil.parse_convert_xml(one_hot_palette_input)
        self.one_hot_palette_label = PathUtil.parse_convert_xml(one_hot_palette_label)
        self.n_classes_input = len(self.one_hot_palette_input)
        self.n_classes_label = len(self.one_hot_palette_label)

    # TODO: refine this method from Trainer class
    def _parse_sample(self, input_files, label_file):
        """
        Parses and processes input and label images.

        Args:
        input_files (list): List of file paths for the input images.
        label_file (str): File path for the label image.
        """
        input_images = [ImageOperationsUtil.load_image(file, self.image_shape) for file in input_files]
        label_image = ImageOperationsUtil.load_image(label_file, self.image_shape)

        input_images = ImageOperationsUtil.normalize_images(input_images)
        label_image = ImageOperationsUtil.normalize_image(label_image)

        input_images = ImageOperationsUtil.one_hot_encode_images(input_images, self.one_hot_palette_input)
        label_image = ImageOperationsUtil.one_hot_encode_image(label_image, self.one_hot_palette_label)

        return input_images, label_image
    
    # TODO:
    def _build_model(self):
        """
        Builds the model using the loaded architecture and weights.
        """
        if self.homographies:
            modelHomographies = PathUtil.load_module(self.homographies)
            self.model = self.architecture.get_network((self.image_shape[0], self.image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label, n_inputs=len(self.training_input_paths), 
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

    def _create_output_directories(self, output_directory):
        """
        Creates the output directories for TensorBoard and model exports.

        Args:
        output_directory (str): Path to the output directory.
        """
        model_output_dir = os.path.join(self.output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.tensorboard_dir = os.path.join(model_output_dir, "TensorBoard")
        self.checkpoint_dir = os.path.join(model_output_dir, "Checkpoints")

        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # TODO:
    def _create_callbacks(self, model_save_interval):
        """
        Creates the callbacks for TensorBoard and model exports.

        Args:
        model_save_interval (int): Epoch interval between exports of the model.
        """
        n_batches_train = len(self.files_train_label) // self.batch_size

        self.callbacks = [
            TrainerCallbacks.create_tensorboard_callback(tensorboard_dir=self.tensorboard_dir),
            TrainerCallbacks.create_model_checkpoint_callback(checkpoint_dir=self.checkpoint_dir, num_training_batches=n_batches_train, save_interval=self.save_interval),
            TrainerCallbacks.create_best_model_checkpoint_callback(checkpoint_dir=self.checkpoint_dir),
            TrainerCallbacks.create_early_stopping_callback(patience=self.early_stopping_patience)
        ]
    
    def _build_training_pipeline(self, training_input_files, training_label_files):
