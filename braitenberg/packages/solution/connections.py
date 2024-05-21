from typing import Tuple

import numpy as np

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Generates a matrix to influence the right motor.
    We put 1s on the left half to promote turning away from obstacles on the left.
    """
    res = np.zeros(shape, dtype="float32")
    res[:, :shape[1]//2] = 1  # Fill the left half with 1s
    # res[:, shape[1]//2:] = -1  # Fill the right half with 1s
    return res
    # Create a gradient from 1 to 0 for the left half
    # res = np.zeros(shape, dtype="float32")
    # # Create a linear vertical gradient from 0 to 1
    # linear_gradient = np.linspace(0, 1, shape[0], endpoint=True).reshape(-1, 1)
    # # Apply a power to skew the gradient
    # skewed_gradient = np.power(linear_gradient, 2)
    # # Apply the skewed gradient to the left half
    # res[:, :shape[1]//2] = skewed_gradient

    return res

def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Generates a matrix to influence the left motor.
    We put 1s on the right half to promote turning away from obstacles on the right.
    """
    res = np.zeros(shape, dtype="float32")
    # res[:, :shape[1]//2] = -1  # Fill the left half with 1s
    res[:, shape[1]//2:] = 1  # Fill the right half with 1s
    return res
    # res = np.zeros(shape, dtype="float32")
    # # Create a linear vertical gradient from 0 to 1
    # linear_gradient = np.linspace(0, 1, shape[0], endpoint=True).reshape(-1, 1)
    # # Apply a power to skew the gradient
    # skewed_gradient = np.power(linear_gradient, 2)
    # # Apply the skewed gradient to the right half
    # res[:, shape[1]//2:] = skewed_gradient

    # return res
 