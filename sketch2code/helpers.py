#!/usr/bin/python
# -*- coding: utf-8 -*-

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

