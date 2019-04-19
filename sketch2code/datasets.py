#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import random

import h5py
import imageio
import numpy as np
import ujson
from typing import *

from tqdm import tqdm

from sketch2code.config import ROOT_DIR
from sketch2code.data_model import Pix2CodeTag, Tag, ToyTag
from sketch2code.helpers import shrink_img
from sketch2code.render_engine import RemoteRenderEngine


def load_dataset(dset: str):
    cls = None
    if dset == "pix2code":
        cls = Pix2CodeTag
    elif dset == "toy":
        cls = ToyTag
    else:
        assert False

    with open(ROOT_DIR / f"datasets/{dset}/data.json", "r") as f:
        tags = [cls.deserialize(o) for o in ujson.load(f)]

    data = h5py.File(ROOT_DIR / f"datasets/{dset}/data.hdf5", "r")
    return tags, data["images"]


def generate_toy_data(n_examples: int):
    """
    Generate toy example
    """

    # Instruction: uncomment this if we want to generate from pix2code
    # def recursively_prune(tag: Pix2CodeTag) -> Optional[ToyTag]:
    #     if tag.name in {"html", "button", "div"}:
    #         children = ["" if isinstance(c, str) else recursively_prune(c) for c in tag.children]
    #         return ToyTag(tag.name, tag.cls, [c for c in children if c is not None])
    #     if tag.name in {'p', 'a', 'ul', 'li', 'nav', 'h5'}:
    #         return None
    #
    #     raise NotImplementedError(f"Not handle {tag.name} yet")
    #
    # with open(ROOT_DIR / "datasets/pix2code/data.json", "r") as f:
    #     pages = [Pix2CodeTag.deserialize(o) for o in ujson.load(f)]

    max_n_rows = 3
    max_n_cols = 4
    btn_types = ["btn-danger", "btn-warning", "btn-success"]
    tags = []
    existing_tags = set()

    for _ in tqdm(range(n_examples * 2)):
        n_rows = random.randint(1, max_n_rows)
        tag = ToyTag("div", ["container-fluid"], [])

        for i in range(n_rows):
            n_cols = random.randint(1, max_n_cols)
            rtag = ToyTag("div", ["row"], [])

            if n_cols == 1:
                col_class = "col-12"
                max_n_btn = 6
            elif n_cols == 2:
                col_class = "col-6"
                max_n_btn = 3
            elif n_cols == 3:
                col_class = "col-4"
                max_n_btn = 2
            elif n_cols == 4:
                col_class = "col-3"
                max_n_btn = 1
            else:
                # 1 <= n_cols <= max_n_cols
                assert False

            for j in range(n_cols):
                ctag = ToyTag("div", ["grey-background"], [])
                for k in range(random.randint(1, max_n_btn)):
                    ctag.children.append(ToyTag("button", ["btn", np.random.choice(btn_types)], []))
                rtag.children.append(ToyTag("div", [col_class], [ctag]))
            tag.children.append(rtag)

        tag = ToyTag("html", [], [tag])
        assert tag.is_valid()

        if tag.to_body() in existing_tags:
            continue

        existing_tags.add(tag.to_body())
        tags.append(tag)

        if len(tags) >= n_examples:
            break

        # with open(ROOT_DIR / "datasets/toy/test.html", "w") as f:
        #     f.write(tag.to_html())
        #     render_engine = RemoteRenderEngine.get_instance(tags[0].to_html(), 480, 300)
        #     img = render_engine.render_page(tag)
        #     imageio.imwrite(str(ROOT_DIR / "datasets/toy/test.jpeg"), shrink_img(img, 0.5))
        #     break

    print("Generate total ", len(tags), "images")
    with open(ROOT_DIR / "datasets/toy/data.json", "w") as f:
        ujson.dump([o.serialize() for o in tags], f)


def make_dataset(dataset_name: str):
    print("Make hdf5 file...")
    dpath = ROOT_DIR / f"datasets/{dataset_name}/data.json"
    if dataset_name == "pix2code":
        viewport_width = 800
        viewport_height = 750
        TagClass = Pix2CodeTag
    elif dataset_name == "toy":
        viewport_width = 480
        viewport_height = 300
        TagClass = ToyTag
    else:
        raise Exception(f"Invalid dataset {dataset_name}")

    with open(dpath, "r") as f:
        tags = [TagClass.deserialize(e) for e in ujson.load(f)]

    render_engine = RemoteRenderEngine.get_instance(tags[0].to_html(), viewport_width, viewport_height)

    print("Rendering pages...")
    results = render_engine.render_pages(tags)

    assert max(x.shape[1] for x in results) == viewport_width, f"Max width: {max(x.shape[1] for x in results)}"
    assert max(x.shape[0] for x in results) == viewport_height, f"Max height: {max(x.shape[0] for x in results)}"
    results = np.asarray(results)

    print("Dumping to hdf5...")
    with h5py.File(ROOT_DIR / f"datasets/{dataset_name}/data.hdf5", "w") as f:
        f.create_dataset("images", data=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('-d', '--dataset', type=str, choices=['toy'])
    parser.add_argument('-n', '--n_examples', type=int, default=2000, help="Number of examples to generate")
    parser.add_argument(
        '-r', '--render', default=False, action='store_true', help='render images for a dataset')
    args = parser.parse_args()

    if args.render:
        make_dataset(args.dataset)
    else:
        if args.dataset == 'toy':
            generate_toy_data(args.n_examples)
