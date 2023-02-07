"""
This project will implement a Convolutional - Regression Network (PyTorch) to predict
the rotation angle of a given image.

Email: eric@ericcanas.com
Date: 26-01-2023
Github: https://github.com/Eric-Canas
"""

import torch

from dataset.yolo_regression_dataset import YoloRegressionDataset
from model.strainet import StraiNet
import os
from PIL import Image
import numpy as np
import cv2

from train.trainer import Trainer
from utils import letterbox

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

TEST_IMAGE_PATH = os.path.join('.', 'test.jpeg')

if __name__ == '__main__':
    #img = np.array(Image.open(TEST_IMAGE_PATH), dtype=np.uint8)
    # Resize the image to 299x299 without changing the aspect ratio (use letterbox to fill the rest)
    #img = letterbox(img, new_shape=(299, 299))
    # Test with a single image
    model = StraiNet()
    model.eval()
    train_dataset = YoloRegressionDataset(source='C:/datasets/validation', images_folder='images', bbox_folder='rotation_yolo')
    val_dataset = YoloRegressionDataset(source='C:/datasets/validation', images_folder='images', bbox_folder='rotation_yolo')

    trainer = Trainer(model, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=32, num_workers=4)
    trainer.train(epochs=10)
