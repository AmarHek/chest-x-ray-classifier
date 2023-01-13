import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from typing import List


def preprocess_gray(image: np.ndarray, mean: float, std: float):
    image = (image - mean) / std
    return image


def imagenet_preprocess(image):
    return preprocess_input(image, mode='torch')


def preprocess_rgb(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    for j in range(image.shape[-1]):
        image[:, :, j] = (image[:, :, j] - mean[j]) / std[j]

    return image


def get_preprocessing_function(weights: str, data_format: str = 'rgb'):
    if weights == "imagenet":
        return imagenet_preprocess
    elif weights == "chestxray14":
        mean = 125.71
        std = 63.07
    elif weights == "chexpert_all":
        mean = 129.10
        std = 73.83
    elif weights == "chexpert_ignore":
        mean = 129.10
        std = 73.84
    elif weights == "chexpert_frontal":
        mean = 129.25
        std = 74.33
    elif weights == "chexpert_lateral":
        mean = 123.23
        std = 77.47
    elif weights == "chexpert_dual":
        mean = 126.21
        std = 75.99
    else:
        raise ValueError("Invalid weights given")

    if data_format == 'rgb':
        if type(mean) != list:
            mean = [mean, mean, mean]
            std = [std, std, std]
        return lambda x: preprocess_rgb(x, mean, std)
    elif data_format == 'gray':
        if type(mean) == list:
            mean = mean[0]
            std = std[0]
        return lambda x: preprocess_gray(x, mean, std)

