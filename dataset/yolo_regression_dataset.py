"""
Reads the dataset and returns the data in a dictionary, applying data augmentation

Email: eric@ericcanas.com
Date: 26-01-2023
Github: https://github.com/Eric-Canas
"""
from __future__ import annotations

import os
import numpy as np
from loguru import logger

import torch
# We should never have the full dataset in memory, so we will use a generator
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import cv2

from dataset.utils import rotate_point, rotate_bbox, rotate_image, rotate_and_crop, crop_and_rotate


class YoloRegressionDataset(Dataset):
    def __init__(self, source='C:/datasets/validation', images_folder='images', bbox_folder='rotation_yolo'):
        super(YoloRegressionDataset, self).__init__()
        self.source = source
        self.images_folder = images_folder
        self.bbox_folder = bbox_folder

        self.data = self._read_dataset()

        # Define the transformations to apply to the images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Resize the image to 299x299 without changing the aspect ratio (use letterbox to fill the rest)
            transforms.Resize((299, 299)),
            # Shift left and right, up and down
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            # Change the brightness
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])

    def __get_rotate_crop_function(self, prob_crop_first:float = 1.0) -> callable:
        """
        Returns the crop_and_rotate or the rotate and crop function with a probability of prob_crop_first

        :param prob_crop_first: float. Probability of returning crop_and_rotate
        :return: callable. The function to use
        """
        return crop_and_rotate if np.random.rand() < prob_crop_first else rotate_and_crop

    def __len__(self):
        return len(self.data)

    def _read_dataset(self) -> list[dict[str, str | tuple[float, float, float, float] | float]]:
        """
        Reads the dataset and returns the data in a dictionary that contains, for each image containing a
        bounding box, the path, the bounding box and the regression label (rotation angle)

        :return: The data in a dictionary
        """

        # Get the list of images
        image_filenames = os.listdir(os.path.join(self.source, self.images_folder))

        data = []
        for image_filename in image_filenames:
            # Get the image path
            image_path = os.path.join(self.source, self.images_folder, image_filename)
            bboxes, attributes = self.read_bboxes_and_labels(image_filename=image_filename)
            if len(bboxes) == 0 or len(attributes) == 0:
                continue
            for (xc, yc, w, h), label in zip(bboxes, attributes):

                data.append({'image_path': image_path,
                             'bbox_xcycwh': (xc, yc, w, h),
                             'bbox_xyxy': (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2),
                             'label': label})

        return data

    def read_bboxes_and_labels(self, image_filename):
        """
        Reads the bounding boxes from the file

        :param image_filename: The name of the image
        :return: The bounding boxes
        """
        filename, ext = os.path.splitext(image_filename)

        # Get the bounding box file path
        bbox_path = os.path.join(self.source, self.bbox_folder, f'{filename}.txt')

        if not os.path.exists(bbox_path):
            logger.warning(f'Bounding box file not found for image {image_filename}')
            return [], []

        # Read the bounding boxes
        with open(bbox_path, 'r') as f:
            bboxes = f.readlines()
            attributes = [bbox.strip().split(' ')[5:] for bbox in bboxes]
            # There must be only one attribute
            assert all([len(attr) == 1 for attr in attributes]), f'Attributes must be one for each bounding box'
            attributes = [float(attr[0]) for attr in attributes]
            bboxes = [bbox.strip().split(' ')[1:5] for bbox in bboxes]
            bboxes = [[float(coord) for coord in bbox] for bbox in bboxes]

        return bboxes, attributes

    def bbox_crop_and_rotate(self, image: np.ndarray, bbox: tuple[float, float, float, float], rotation: float) -> \
        tuple[np.ndarray, float]:
        """
        Returns the cropped element with a random rotation applied to it
        :param image: np.ndarray. The image to crop and rotate
        :param bbox: tuple[float, float, float, float]. The bounding box of the element to crop and rotate, in format
                                                        (x1, y1, x2, y2). Range: [0, 1]
        :param rotation: float. The current rotation label of the element to crop and rotate

        :return: tuple[np.ndarray, float]. The cropped and rotated image and the new rotation label
        """
        # Get the new random rotation angle TODO: It does not work
        angle = np.random.randint(0, 180)
        # Get the crop and rotate function
        rotate_crop_function = self.__get_rotate_crop_function()
        # Crop and rotate the image
        image = rotate_crop_function(img=image, bbox=bbox, angle=angle)

        # Update the rotation label
        rotation = (rotation + angle) % 360

        return image, rotation


    def __getitem__(self, idx):
        # Get the data
        data = self.data[idx]
        # Read the image
        image = cv2.imread(data['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, label = self.bbox_crop_and_rotate(image=image, bbox=data['bbox_xyxy'], rotation=data['label'])

        # Apply the transformations
        image = self.transform(image)
        # Get the label
        label = torch.tensor(label, dtype=torch.float32)

        return image, label
