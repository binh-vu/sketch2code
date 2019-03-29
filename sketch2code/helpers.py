#!/usr/bin/python
# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np
from pathlib import Path
from typing import *


def read_file(fpath: Union[Path, str]):
    with open(fpath, 'r') as f:
        return f.read()


def norm_rgb_imgs(imgs: np.ndarray) -> np.ndarray:
    """Convert the RGB image data to range 0 and 1 (0 is white)"""
    return (255 - imgs) / 255


def shrink_img(img: np.ndarray, scale_factor: float) -> np.ndarray:
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)


def viz_grid(imgs: np.ndarray, padding: int=1):
    N, H, W, C = imgs.shape
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size + 1)
    grid_width = W * grid_size + padding * (grid_size + 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = padding, H + padding
    for y in range(grid_size):
        x0, x1 = padding, W + padding
        for x in range(grid_size):
            if next_idx < N:
                img = imgs[next_idx]
                grid[y0:y1, x0:x1] = img
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid
