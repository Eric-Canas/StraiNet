"""
Converts the Datumaro to YOLO with Attributes format. The dataset must have the following structure:
    - class xc yc w h attribute
"""

import os
import numpy as np
import json
from loguru import logger
import cv2

ATTRIBUTES = ('Orientation',)
IMAGES_PATH = 'C:/datasets/validation/images'
DATASET_PATH = 'C:/datasets/validation/rotation'
OUTPUT_PATH = 'C:/datasets/validation/rotation_yolo'

def parse_dataset(src_path: str, output_path: str, attributes: tuple[str]):
    """
    Parses the dataset in the Datumaro to YOLO with Attributes format

    :param src_path: str. Path to the dataset
    :param output_path: str. Path to the output folder
    :param attributes: list[str]. List of attributes
    """

    # Create the output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Parse every json file in the dataset
    for json_filename in os.listdir(src_path):
        # Get the json file path
        json_path = os.path.join(src_path, json_filename)
        # Cast it to a dictionary
        with open(json_path, 'r') as f:
            data = json.load(f)
        assert all(attribute in data['categories']['label']['attributes'] for attribute in attributes), \
            f'Attributes do not match for file {json_filename}'
        data = data['items']
        for image in data:
            save_image_data(data=image, output_path=output_path, attributes=attributes)

def save_image_data(data: dict, output_path: str = OUTPUT_PATH, attributes: tuple[str] = ATTRIBUTES, source_path: str = IMAGES_PATH):
    """
    Stores the data of an image in a txt file
    :param data: dict. The data of the image
    :param output_path: str. The path to the output folder
    :param attributes: list[str]. The to save after bbox data
    """

    filename = os.path.basename(data['id'])
    # Get the image filename
    image_path = os.path.join(source_path, filename)
    # Get the filename with the extension
    for ext in ('.jpg', '.png', '.jpeg'):
        new_path = f'{image_path}{ext}'
        if os.path.isfile(new_path):
            image_path = new_path
            break
    else:
        logger.error(f'Image {filename} not found')
        return

    # Get the image data
    image_data = data['annotations']
    # Get the image size
    h, w, _ = cv2.imread(image_path).shape

    # Get the output file path
    output_file_path = os.path.join(output_path, f'{filename}.txt')

    # Write the data to the file
    with open(output_file_path, 'w') as f:
        for i, annotation in enumerate(data['annotations']):
            # Get the class
            class_ = annotation['label_id']
            # Get the bbox
            bbox = annotation['bbox']
            # Get the attributes
            bbox = bbox_to_yolo(bbox=bbox, w=w, h=h)
            assert all(0 <= coord <= 1 for coord in bbox), f'Invalid bbox for file {filename}'
            attributes_ = [int(annotation['attributes'][attribute]) for attribute in attributes]
            # Write the data to the file, one line per bbox
            f.write(f'{class_} {" ".join(map(str, bbox))} {" ".join(map(str, attributes_))}')
            # Add a new line
            if i < len(data['annotations']) - 1:
                f.write('\n')



def bbox_to_yolo(bbox: list[float], w: int, h: int) -> tuple[float]:
    """
    Converts the bbox to YOLO format
    :param bbox: list[float]. The bbox
    :param w: int. The image width
    :param h: int. The image height
    :return: list[float]. The bbox in YOLO format
    """

    # Get the bbox coordinates
    x, y, w_, h_ = bbox
    # Get the center of the bbox
    xc, yc = x + w_ / 2, y + h_ / 2
    # To range [0, 1]
    xc, yc, w_, h_ = xc / w, yc / h, w_ / w, h_ / h
    return (xc, yc, w_, h_)

if __name__ == '__main__':
    parse_dataset(src_path=DATASET_PATH, output_path=OUTPUT_PATH, attributes=ATTRIBUTES)