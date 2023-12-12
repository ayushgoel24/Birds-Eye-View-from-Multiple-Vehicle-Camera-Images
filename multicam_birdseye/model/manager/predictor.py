import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from .model_manager import ModelManager
from ...utils import PathUtil
from .common.data_loader import DataLoader

class Predictor(ModelManager):

    def __init__(self, test_input_paths, max_test_samples, 
                 target_image_shape, input_one_hot_palette_path, label_one_hot_palette_path, 
                 class_names, model_path, homography_path, pretrained_model_weights):


        self.test_input_paths = [PathUtil.get_absolute_path(path) for path in test_input_paths] # directory/directories of input samples for testing
        self.max_test_samples = max_test_samples # maximum number of test samples
        self.target_image_shape = target_image_shape # image dimensions (HxW) of inputs and labels for network
        self.input_one_hot_palette_path = PathUtil.get_abs_path(input_one_hot_palette_path) # xml-file for one-hot-conversion of input images
        self.label_one_hot_palette_path = PathUtil.get_abs_path(label_one_hot_palette_path) # xml-file for one-hot-conversion of label images
        self.model_path = PathUtil.get_abs_path(model_path) # Python file defining the neural network
        self.homography_path = PathUtil.get_abs_path(homography_path) if homography_path else None # Python file defining a list H of homographies to be used in model
        self.pretrained_model_weights = PathUtil.get_abs_path(pretrained_model_weights) if pretrained_model_weights else None # weights file of trained model for training continuation

        self.architecture = super()._load_model_architecture(self.model_path)
        
        self.validation_input_files, self.validation_label_files = DataLoader.prepare_data(data_paths=self.test_input_paths, label_path=self.validation_label_path, max_samples=self.max_validation_samples)
        super()._parse_one_hot_conversion()
        self._build_model()

    def _build_model(self):
        """
        Builds the model using the loaded architecture and weights.
        """
        if self.homography_path:
            modelHomographies = PathUtil.load_module(self.homography_path)
            self.model = self.architecture.get_network((self.target_image_shape[0], self.target_image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label, n_inputs=len(self.test_input_paths), 
                                                       thetas=modelHomographies.H)
        else:
            self.model = self.architecture.get_network((self.target_image_shape[0], self.target_image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label, n_inputs=len(self.test_input_paths))
        self.model.load_weights(self.pretrained_model_weights)
        print(f"Reloaded model from {self.pretrained_model_weights}")

    def _run_predictions(self, prediction_directory):
        """
        Run predictions on the provided data and save the results to the specified directory.

        Args:
        prediction_directory (str): Path to the directory where prediction results will be saved.
        """
        # Create output directory if it does not exist
        if not os.path.exists(prediction_directory):
            os.makedirs(prediction_directory)

        print(f"Running predictions and writing to {prediction_directory} ...")
        for k in tqdm.tqdm(range(self.n_samples)):
            input_files = [self.files_input[i][k] for i in range(self.n_inputs)]

            # Load sample
            inputs = self.parse_sample(input_files)

            # Add batch dimension
            inputs = [np.expand_dims(inp, axis=0) for inp in inputs] if self.n_inputs > 1 else np.expand_dims(inputs, axis=0)

            # Run prediction
            prediction = self.model.predict(inputs).squeeze()

            # Convert to output image
            prediction_image = utils.one_hot_decode_image(prediction, self.one_hot_palette_label)

            # Write to disk
            output_file = os.path.join(prediction_directory, os.path.basename(input_files[0]))
            cv2.imwrite(output_file, cv2.cvtColor(prediction_image, cv2.COLOR_RGB2BGR))


    def run(self):
        """
        Run the prediction.
        """
        self._run_predictions()