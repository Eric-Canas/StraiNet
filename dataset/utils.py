from __future__ import annotations

import numpy as np
import imutils
import cv2


def rotate_bbox(bbox: tuple[float, float, float, float], angle: float) -> tuple[float, float, float, float]:
    """
    Returns the bounding box coordinates that correspond to rotating the original image by angle degrees from its
    center (0.5, 0.5)
    :param bbox: tuple[float, float, float, float]. The bounding box to rotate in format x1, y1, x2, y2
                                                    width, height. In range [0., 1.]
    :param angle: float. The angle to rotate the image in degrees
    :return: tuple[float, float, float, float]. The new bounding box in format x_center, y_center, width, height. In
                                                range [0., 1.]
    """

    xtr, ytr = rotate_point(x_center + width / 2, y_center - height / 2, angle)
    xbr, ybr = rotate_point(x_center + width / 2, y_center + height / 2, angle)
    xbl, ybl = rotate_point(x_center - width / 2, y_center + height / 2, angle)
    xtl, ytl = rotate_point(x_center - width / 2, y_center - height / 2, angle)

    # Get the new bounding box coordinates
    x1, y1 = min(xtr, xbr, xbl, xtl), min(ytr, ybr, ybl, ytl)
    x2, y2 = max(xtr, xbr, xbl, xtl), max(ytr, ybr, ybl, ytl)

    # Get the new width and height
    width, height = x2 - x1, y2 - y1
    # Get the new center
    x_center, y_center = x1 + width / 2, y1 + height / 2

    return x_center, y_center, width, height


def rotate_point(x: float, y: float, angle: float) -> tuple[float, float]:
    """
    Returns the coordinates of the point (x, y) after rotating it by angle degrees from the origin (0.5, 0.5)
    :param x: float. The x coordinate of the point. It must be in the range [0., 1.]
    :param y: float. The y coordinate of the point. It must be in the range [0., 1.]
    :param angle: float. The angle to rotate the [1, 1] space containing the point. In degrees. Clockwise
    :return: tuple[float, float]. The new coordinates of the point
    """
    # Get the angle in radians
    angle = np.deg2rad(angle)

    # Get the coordinates of the point after rotating the image from its center
    x_new = 0.5 + (x - 0.5) * np.cos(angle) - (y - 0.5) * np.sin(angle)
    y_new = 0.5 + (x - 0.5) * np.sin(angle) + (y - 0.5) * np.cos(angle)

    return x_new, y_new

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by angle degrees from its center. The image is padded with zeros. The output size will change in
    order to fit the rotated image
    :param img: np.ndarray. The image to rotate
    :param angle: float. The angle to rotate the image. In degrees. Clockwise
    :return: np.ndarray. The rotated image
    """

    # Use OpenCV warp to apply it on both the image and the bounding box
    h, w = img.shape[:2]
    # Get the center of the image
    cX, cY = (w / 2, h / 2)

    M, (new_h, new_w) = _get_rotation_matrix_and_output_width(x=cX, y=cY, h=h, w=w, angle=angle)
    # Rotate the image
    img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))

    return img

def _get_rotation_matrix_and_output_width(x: float|int, y: float|int, h: int, w: int, angle: float|int) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Returns the rotation matrix and the output width of the image after rotating it by angle degrees from x, y point.
    The output height and width will perfectly fit the rotated image without any crop
    :param x: float. The x coordinate of the point where to rotate the image.
    :param y: float. The y coordinate of the point where to rotate the image.
    :param h: int. The height of the original image
    :param w: int. The width of the original image
    :param angle: float. The angle to rotate the image. In degrees. Clockwise
    :return: tuple[np.ndarray, int]. The rotation matrix and the output height and width of the image
    """
    M = cv2.getRotationMatrix2D((x, y), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - x
    M[1, 2] += (nH / 2) - y

    return M, (nH, nW)

def rotate_and_crop(img: np.ndarray, bbox: tuple[float, float, float, float], angle: float) -> np.ndarray:
    """
    Rotate the image by angle degrees from its center and crop the image by the bounding box (after rotating it)
    :param img: np.ndarray. The image to rotate and crop
    :param bbox: tuple[float, float, float, float]. The bounding box to rotate in format x1, y1, x2, y2. In range [0., 1.]
    :param angle: float. The angle to rotate the image. In degrees. Clockwise
    :return: np.ndarray. The rotated and cropped image containing only the rotated bounding box
    """

    # Use OpenCV warp to apply it on both the image and the bounding box
    h, w = img.shape[:2]
    # Get the center of the image
    cX, cY = (w / 2, h / 2)

    # Resize the bounding box from [0., 1.] to [0, h] and [0, w]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

    # Transform the bounding box to a polygon
    bbox = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)

    M, (new_h, new_w) = _get_rotation_matrix_and_output_width(x=cX, y=cY, h=h, w=w, angle=angle)
    # Rotate the image
    img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))

    rotated_points = cv2.transform(bbox.reshape(1, -1, 2), M).reshape(-1, 2)
    # Depending on the rotation, x can turn into y and vice versa
    # X and Y coordinates are swapped when rotating by

    # Get the new bounding box
    (x1, y1), (x2, y2) = np.min(rotated_points, axis=0), np.max(rotated_points, axis=0)
    # Clip it
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(new_w, x2), min(new_h, y2)
    return img[int(y1):int(y2), int(x1):int(x2)]


def crop_and_rotate(img: np.ndarray, bbox: tuple[float, float, float, float], angle: float) -> np.ndarray:
    """
    First crops the image by the bounding box and then rotates it by angle degrees from its center. It will produce
    paddings inside the image.
    :param img: np.ndarray. The image to crop and rotate
    :param bbox: tuple[float, float, float, float]. The bounding box to crop in format x1, y1, x2, y2. In range [0., 1.]
    :param angle: float. The angle to rotate the cropped image. In degrees. Clockwise
    :return: np.ndarray. The cropped and rotated image
    """

    # Resize the bounding box from [0., 1.] to [0, h] and [0, w]
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

    # Crop the image
    img = img[int(y1):int(y2), int(x1):int(x2)]

    # Rotate the image
    return rotate_image(img, angle)