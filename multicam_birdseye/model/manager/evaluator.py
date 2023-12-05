import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import seaborn
import tensorflow
import tqdm

from .model_manager import ModelManager
from ...utils import PathUtil
from .common.data_loader import DataLoader

class Evaluator(ModelManager):

    def __init__(self, validation_input_paths, validation_label_path, max_validation_samples, 
                 target_image_shape, input_one_hot_palette_path, label_one_hot_palette_path, 
                 class_names, model_path, homography_path, pretrained_model_weights):
        """
        Initialize the Evaluation class with validation data paths, model details, and evaluation configurations.

        Args:
        input_validation (list): List of paths for validation input data.
        label_validation (str): Path for validation label data.
        max_samples_validation (int): Maximum number of validation samples to use.
        image_shape (tuple): Image dimensions (HxW) of inputs and labels for network.
        one_hot_palette_input (str): Path to xml-file for one-hot-conversion of input images.
        one_hot_palette_label (str): Path to xml-file for one-hot-conversion of label images.
        class_names (list): Class names to annotate confusion matrix axes.
        model_path (str): Path to the Python file defining the neural network.
        homography_path (str): Path to the Python file defining a list of homographies.
        model_weights (str): Path to the weights file of the trained model.
        """


        self.validation_input_paths = [PathUtil.get_absolute_path(path) for path in validation_input_paths] # directory/directories of input samples for validation
        self.validation_label_path = PathUtil.get_absolute_path(validation_label_path) # directory of label samples for validation
        self.max_validation_samples = max_validation_samples # maximum number of validation samples
        self.target_image_shape = target_image_shape # image dimensions (HxW) of inputs and labels for network
        self.input_one_hot_palette_path = PathUtil.get_abs_path(input_one_hot_palette_path) # xml-file for one-hot-conversion of input images
        self.label_one_hot_palette_path = PathUtil.get_abs_path(label_one_hot_palette_path) # xml-file for one-hot-conversion of label images
        self.class_names = class_names
        self.model_path = PathUtil.get_abs_path(model_path) # Python file defining the neural network
        self.homography_path = PathUtil.get_abs_path(homography_path) if homography_path else None # Python file defining a list H of homographies to be used in model
        self.pretrained_model_weights = PathUtil.get_abs_path(pretrained_model_weights) if pretrained_model_weights else None # weights file of trained model for training continuation

        self.architecture = super()._load_model_architecture(self.model_path)
        
        self.validation_input_files, self.validation_label_files = DataLoader.prepare_data(data_paths=self.validation_input_paths, label_path=self.validation_label_path, max_samples=self.max_validation_samples)
        super()._parse_one_hot_conversion()
        self._build_model()

    def _build_model(self):
        """
        Builds the model using the loaded architecture and weights.
        """
        if self.homography_path:
            modelHomographies = PathUtil.load_module(self.homography_path)
            self.model = self.architecture.get_network((self.target_image_shape[0], self.target_image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label, n_inputs=len(self.validation_input_paths), 
                                                       thetas=modelHomographies.H)
        else:
            self.model = self.architecture.get_network((self.target_image_shape[0], self.target_image_shape[1], self.n_classes_input), 
                                                       self.n_classes_label, n_inputs=len(self.validation_input_paths))
        self.model.load_weights(self.pretrained_model_weights)
        print(f"Reloaded model from {self.pretrained_model_weights}")

    def _normalize_confusion_matrix(self):
        """
        Normalize the rows of the confusion matrix.
        """
        self.confusion_matrix_norm = self.confusion_matrix / np.sum(self.confusion_matrix, axis=1)[:, np.newaxis]

    def _compute_confusion_matrix(self):
        """
        Compute the confusion matrix for the given dataset.
        """
        print("Evaluating confusion matrix ...")
        # TODO:
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        for k in tqdm.tqdm(range(len(self.label_files))):
            input_files_sample = [self.validation_input_files[i][k] for i in range(len(self.validation_input_paths))]
            label_file = self.validation_label_files[k]
            inputs, label = self._parse_sample(input_files=input_files_sample, label_file=label_file)

            # add batch dim
            if len(self.validation_input_paths) > 1:
                inputs = [np.expand_dims(i, axis=0) for i in inputs]
            else:
                inputs = np.expand_dims(inputs, axis=0)

            # Predict and compute confusion matrix for the sample
            prediction = self.model.predict(inputs).squeeze()
            label = np.argmax(label, axis=-1)
            prediction = np.argmax(prediction, axis=-1)
            sample_confusion_matrix = tensorflow.math.confusion_matrix(
                label.flatten(), 
                prediction.flatten(),
                num_classes=self.num_classes
                ).numpy()
            # sum confusion matrix over dataset
            confusion_matrix += sample_confusion_matrix

        self.confusion_matrix = confusion_matrix
        self._normalize_confusion_matrix()
    
    def _compute_class_iou(self):
        """
        Compute the Intersection over Union (IoU) for each class.
        """
        row_sum = np.sum(self.confusion_matrix, axis=0)
        col_sum = np.sum(self.confusion_matrix, axis=1)
        diag = np.diag(self.confusion_matrix)
        intersection = diag
        union = row_sum + col_sum - diag
        self.ious = intersection / union
    
    def _print_evaluation_metrics(self):
        """
        Print the evaluation metrics, including per-class IoU and confusion matrices.
        """
        print("\nPer-class IoU:")
        for idx, value in enumerate(self.ious):
            print(f"  {self.class_names[idx]}: {100*value:3.2f}%")
        
        print("\nConfusion Matrix:")
        print(self.confusion_matrix)

        print("\nNormalized Confusion Matrix:")
        print(self.confusion_matrix_norm)
    
    def _plot_confusion_matrix(self, output_folder):
        """
        Plot and save the confusion matrix.

        Args:
        output_folder (str): Directory path to save the confusion matrix plot and metrics.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Plotting the confusion matrix
        confusion_matrix_df = pandas.DataFrame(self.confusion_matrix_norm*100, self.class_names, self.class_names)
        plt.figure(figsize=(8,8))
        heatmap = seaborn.heatmap(
            confusion_matrix_df, 
            annot=True, 
            fmt=".2f", 
            square=True, 
            vmin=0, 
            vmax=100, 
            cbar_kws={"label": "%", "shrink": 0.8}, 
            cmap=plt.cm.Blues
        )
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # Save the plot and metrics to files
        confusion_matrix_file = os.path.join(output_folder, "confusion_matrix.txt")
        np.savetxt(confusion_matrix_file, self.confusion_matrix, fmt="%d")

        class_iou_file = os.path.join(output_folder, "class_iou.txt")
        np.savetxt(class_iou_file, self.ious, fmt="%f")
        
        confusion_matrix_plot_file = os.path.join(output_folder, "confusion_matrix.pdf")
        plt.savefig(confusion_matrix_plot_file, bbox_inches="tight")

    def run(self):
        """
        Run the evaluation.
        """
        self._compute_confusion_matrix()
        self._compute_class_iou()
        self._print_evaluation_metrics()

        eval_folder = os.path.join( os.path.dirname(self.pretrained_model_weights), os.pardir, "Evaluation" )
        self._plot_confusion_matrix(output_folder=eval_folder)