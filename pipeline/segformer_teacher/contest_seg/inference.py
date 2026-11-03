import numpy as np
import torch


def normalize_image(array):
    array = array.astype(np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    return torch.from_numpy(array.transpose(2, 0, 1)).float()

