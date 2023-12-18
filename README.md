# Bird’s-Eye-View Transformation from Multiple Vehicle Camera Images

## Introduction
This repository contains the implementation of our methodology for transforming multiple vehicle-mounted camera images into a bird’s-eye view (BEV). This approach is crucial in applications like traffic management, autonomous driving, and urban planning. We leverage deep learning architectures to model spatial relationships and depth cues in on-ground images, enhancing the accuracy and realism of the top-down transformation.

## Team Members
- Ayush Goel - aygoel@seas.upenn.edu
- Ojas Mandlecha - ojasm@seas.upenn.edu
- Renu Reddy Kasala - renu1@seas.upenn.edu

## Implementation Details
- **Dataset**: We used the CARLA simulator dataset, comprising images from left, right, front, rear-view cameras, and an aerial top-down view from a drone’s perspective.
- **Methodology**: 
    - We employed a U-Net backbone for semantic segmentation of images.
    - Our network architecture features a multi-input encoder-decoder framework with a spatial transformer module.
    - The model utilizes SegFormer for obtaining semantically segmented real images.
    - We implemented Inverse Perspective Mapping (IPM) to improve spatial consistency.
- **Training Environment**: 
    - AWS g5.4xLarge instance with NVIDIA A100 Tensor Core GPUs, 24GiB GPU Memory, 16 vCPUs, 600GiB CPU Memory.

## Setup and Training Instructions
1. **Environment Setup**:
    - Clone the repository: `https://github.com/ayushgoel24/Birds-Eye-View-from-Multiple-Vehicle-Camera-Images`.
    - Install the required libraries: `pip install -r requirements.txt`.

2. **Model Training**:
    - Run the training script: `python train.py` from the root directory.
    - Custom arguments could be passed through the training scripts from the configuration file: `configurations/dataset_configurations.yaml`

3. **Model Evaluation**:
    - After training, evaluate the model using `evaluate.py` script.
    - `python evaluate.py`.

5. **Reproducing Results**:
    - To reproduce our results, use the same dataset and follow the training instructions.
    - Hyperparameters: 
        - Batch Size = 8
        - Learning Rate = 5e-5
        - Betas = (0.9, 0.99)
        - Epochs = 30
        - Weight Decay = 1e-6.

## Results
Our methodology achieved a Mean Intersection-over-Union (MIoU) of 55.31% on the validation set. Class IoU scores further validate the effectiveness of our model, particularly in predicting large-area semantic classes.

## Future Work
We plan to explore enhancements that address finer details in semantic segmentation, refining the network architecture for more nuanced feature extraction, and extending our methodology to diverse real-world scenarios.

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.