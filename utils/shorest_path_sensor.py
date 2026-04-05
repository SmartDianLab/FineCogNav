import os
import tqdm
import time
import numpy as np
import numba as nb
import json
from pathlib import Path

@nb.njit(nogil=True, cache=True)
def EuclideanDistance3(point_a: np.array, point_b: np.array) -> float:
    """
    Euclidean distance of two given point (3D)
    """
    # return float(np.linalg.norm((point_a - point_b), ord=2))
    return float(np.sqrt(np.sum(np.square(np.subtract(point_a, point_b)))))


@nb.njit(nogil=True, cache=True)
def EuclideanDistance1(point_a: np.array, point_b: np.array) -> float:
    """
    Euclidean distance of two given point (1D)
    """
    return float(np.sqrt(np.square(float(point_a - point_b))))


def Distance(edge, termial, token_dict):
    point_a = np.array(token_dict[edge])
    point_b = np.array(token_dict[termial])
    distance = EuclideanDistance3(point_a, point_b)
    return float(distance)