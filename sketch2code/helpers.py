#!/usr/bin/python
# -*- coding: utf-8 -*-
import math

import cv2
import h5py
import numpy as np
from pathlib import Path
from typing import *
import pickle as cpickle

from sketch2code.config import ROOT_DIR


def inc_folder_no(fpath: Path):
    """Give a path to folder (prefix) and return path of next folder should be. for example:

    input: /runs/exp_ => output is: /runs/exp_1, /runs/exp_2, ...
    """
    prefix = fpath.name
    existing_dirs = []
    for dpath in fpath.parent.iterdir():
        if dpath.name.startswith(prefix):
            existing_dirs.append(int(dpath.name.replace(prefix, "")))
    existing_dirs.sort()

    if len(existing_dirs) == 0:
        new_dir = fpath.parent / f"{prefix}1"
    else:
        new_dir = fpath.parent / f"{prefix}{existing_dirs[-1] + 1}"
    return str(new_dir)


def read_file(fpath: Union[Path, str]):
    with open(fpath, 'r') as f:
        return f.read()


def norm_rgb_imgs(imgs: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Convert the RGB image data to range 0 and 1 (0 is white)"""
    return np.divide((255 - imgs), 255, dtype=dtype)


def shrink_img(img: np.ndarray, scale_factor: float, interpolation=cv2.INTER_AREA) -> np.ndarray:
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation)


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


def cache_arrays(id: str, create_arrays: Callable[[], Tuple[np.ndarray]], pickle: bool=False):
    cache_file = ROOT_DIR / "tmp" / f"cache.{id}.{'pkl' if pickle else 'hdf5'}"
    if cache_file.exists():
        if pickle:
            with open(cache_file, "rb") as f:
                return cpickle.load(f)
        dataset = h5py.File(cache_file, "r")
        return [dataset[k][:] for k in sorted(dataset.keys())]
    else:
        if pickle:
            arrays = create_arrays()
            with open(cache_file, "wb") as f:
                cpickle.dump(arrays, f)
        else:
            with h5py.File(cache_file, "w") as f:
                arrays = []
                for i, array in enumerate(create_arrays()):
                    f.create_dataset(f"a{i}", data=array)
                    arrays.append(array)
        return arrays


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
