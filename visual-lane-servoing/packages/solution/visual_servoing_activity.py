from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    steer_matrix_left = np.zeros(shape, dtype=np.float32)
    
    # Define the steering weights for the left lane markings
    height, width = shape
    for y in range(height):
        for x in range(width):
            if x < width // 2:
                steer_matrix_left[y, x] = -1.0 * (width // 2 - x) / (width // 2)
            else:
                steer_matrix_left[y, x] = 1.0 * (x - width // 2) / (width // 2)

    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    steer_matrix_right = np.zeros(shape, dtype=np.float32)
    
    # Define the steering weights for the right lane markings
    height, width = shape
    for y in range(height):
        for x in range(width):
            if x < width // 2:
                steer_matrix_right[y, x] = -1.0 * (width // 2 - x) / (width // 2)
            else:
                steer_matrix_right[y, x] = 1.0 * (x - width // 2) / (width // 2)

    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
        right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape


    white_lower_color = np.array([0, 0, 180])       # Adjusted for white
    white_upper_color = np.array([179, 50, 255])    # Adjusted for white
    yellow_lower_color = np.array([20, 100, 100])   # Adjusted for yellow
    yellow_upper_color = np.array([30, 255, 255])   # Adjusted for yellow

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    gaussian_blur = cv2.GaussianBlur(img_gray_scale, (0,0), np.pi)
    sobel_x_crd = cv2.Sobel(gaussian_blur, cv2.CV_64F,1,0)
    sobel_y_crd = cv2.Sobel(gaussian_blur, cv2.CV_64F,0,1)
    G = np.sqrt(sobel_x_crd*sobel_x_crd + sobel_y_crd*sobel_y_crd)

    threshold_w = 27
    threshold_y = 47

    mask_bool_white = (G > threshold_w)
    mask_bool_yellow = (G > threshold_y)

    real_mask_white = cv2.inRange(img_hsv, white_lower_color, white_upper_color)
    real_mask_yellow = cv2.inRange(img_hsv, yellow_lower_color, yellow_upper_color)

    left_mask = np.ones(sobel_x_crd.shape)
    left_mask[:,int(np.floor(w/2)):w+1] = 0
    right_mask = np.ones(sobel_x_crd.shape)
    right_mask[:,0:int(np.floor(w/2))] = 0
    left_mask[:int(sobel_x_crd.shape[0]/2)] = 0
    right_mask[:int(sobel_x_crd.shape[0]/2)] = 0
    sobel_x_pos_mask = (sobel_x_crd > 0)
    sobel_x_neg_mask = (sobel_x_crd < 0)
    sobel_y_pos_mask = (sobel_y_crd > 0)
    sobel_y_neg_mask = (sobel_y_crd < 0)

    left_edge_mask = left_mask * mask_bool_yellow * sobel_x_neg_mask * sobel_y_neg_mask * real_mask_yellow
    right_edge_mask = right_mask * mask_bool_white * sobel_x_pos_mask * sobel_y_neg_mask * real_mask_white

    return left_edge_mask, right_edge_mask