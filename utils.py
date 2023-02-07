"""
This file contains some general purpose functions that will be used in the project.

Email: eric@ericcanas.com
Date: 26-01-2023
Github: https://github.com/Eric-Canas
"""

import cv2
import numpy as np

def letterbox(img: np.ndarray, new_shape: tuple=(299, 299), color: tuple=(114, 114, 114))->np.ndarray:
    """
    Resize an image to a given shape while maintaining the aspect ratio by adding padding.
    :param img: np.ndarray, the image to resize.
    :param new_shape: tuple, the shape to resize the image to. Default: (299, 299) [InceptionV3 input shape].
    :param color: tuple, the color to fill the padding with. Default: (114, 114, 114) [InceptionV3 mean].
    :return: np.ndarray, the resized image.
    """

    # Get the shape of the image
    (h, w), (new_h, new_w) = img.shape[:2], new_shape
    # Get the ratio of the new shape to the old shape
    r = min(new_w / w, new_h / h)
    # Get the new shape
    new_unpad = (int(w * r), int(h * r))
    # Resize the image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)
    # Get the padding
    dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]
    # Apply it centered
    top, left, bottom, right = dh // 2, dw // 2, dh - (dh // 2), dw - (dw // 2)
    # Add the padding
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img
