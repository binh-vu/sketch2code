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


def norm_rgb_imgs(imgs: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Convert the RGB image data to range 0 and 1 (0 is white)"""
    return np.divide((255 - imgs), 255, dtype=dtype)


def shrink_img(img: np.ndarray, scale_factor: float) -> np.ndarray:
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)


def viz_grid(imgs: np.ndarray, padding: int = 1, padding_color: float = 0, n_img_w: Optional[int] = None):
    N, H, W, C = imgs.shape

    if n_img_w is None:
        n_img_w = int(math.ceil(math.sqrt(N)))
    n_img_h = math.ceil(N / n_img_w)

    grid_width = W * n_img_w + padding * (n_img_w + 1)
    grid_height = H * n_img_h + padding * (n_img_h + 1)
    grid = np.ones((grid_height, grid_width, C)) * padding_color

    next_idx = 0
    y0, y1 = padding, H + padding
    for y in range(n_img_h):
        x0, x1 = padding, W + padding
        for x in range(n_img_w):
            if next_idx < N:
                img = imgs[next_idx]
                grid[y0:y1, x0:x1] = img
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


class Placeholder:

    def __repr__(self):
        content = []
        for k, v in self.__dict__.items():
            if v is None or isinstance(v, (int, float)):
                content.append(f"[{k}={v}]")
            elif isinstance(v, str):
                content.append(f"[{k}={v[:5]}...]")
            else:
                content.append(f"[{k}~{type(v)}]")
        return " ".join(content)
