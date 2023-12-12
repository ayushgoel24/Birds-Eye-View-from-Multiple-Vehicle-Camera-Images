import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerFeatureExtractor
import matplotlib.pyplot as plt

class SemanticSegmentationPipeline:
    """
    A pipeline for semantic segmentation using a Segformer model.

    Attributes:
        device (torch.device): The device (CPU or CUDA) used for computations.
        processor (AutoImageProcessor): The processor for image preprocessing.
        model (SegformerForSemanticSegmentation): The Segformer model for semantic segmentation.
        feature_extractor (SegformerFeatureExtractor): The feature extractor for the model.
        palette (list): The color palette for segmentation visualization.
    """

    def __init__(self, model_name="nvidia/mit-b5", palette=None):
        """
        Initializes the semantic segmentation pipeline.

        Args:
            model_name (str): The name of the model to be loaded from Hugging Face's model hub.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.feature_extractor = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
        self.palette = palette

    def preprocess_image(self, image):
        """
        Preprocesses the input image for the model.

        Args:
            image (PIL.Image or np.ndarray): The input image to be processed.

        Returns:
            torch.Tensor: The processed pixel values.
        """
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.to(self.device)
        return pixel_values

    def run_inference(self, pixel_values):
        """
        Runs inference on the processed image.

        Args:
            pixel_values (torch.Tensor): The processed pixel values of the image.

        Returns:
            torch.Tensor: The logits output by the model.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        return logits

    def postprocess_logits(self, logits, image):
        """
        Postprocesses the logits to create a segmented image.

        Args:
            logits (torch.Tensor): The logits from the model.
            image (PIL.Image or np.ndarray): The original input image.

        Returns:
            np.ndarray: The segmented image.
        """
        upsampled_logits = nn.functional.interpolate(logits, size=image.shape[:-1], mode='bilinear', align_corners=False)
        seg = upsampled_logits.argmax(dim=1)[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(self.palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]  # Convert to BGR
        return color_seg

    def display_results(self, image, color_seg):
        """
        Displays the original and segmented images.

        Args:
            image (PIL.Image or np.ndarray): The original input image.
            color_seg (np.ndarray): The segmented image.
        """
        img = np.array(image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(img)
        axs[1].imshow(color_seg)
        plt.show()

    def run(self, image):
        """
        Runs the entire pipeline: preprocessing, inference, postprocessing, and displaying results.

        Args:
            image (PIL.Image or np.ndarray): The input image for segmentation.
        """
        pixel_values = self.preprocess_image(image)
        logits = self.run_inference(pixel_values)
        color_seg = self.postprocess_logits(logits, image)
        self.display_results(image, color_seg)
