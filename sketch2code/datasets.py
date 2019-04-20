#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import random
import shutil
from pathlib import Path

import h5py
import imageio
import numpy as np
import ujson
from typing import *

from faker import Faker
from parsimonious import Grammar
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


def make_pix2code(pix2code_dir: Path):
    """Download file from google drive, run pre-processing to generate correct data format"""
    grammar = Grammar(r"""
program = group_token+
group_token = (_token_ "{" group_token (","? group_token)* _ "}" _) / _token_
_token_ = _ token _
token = ~"[A-Z-0-9]+"i
_ = ~"[ \n]*"
""")
    fake = Faker()

    def read_peg(node, output: list):
        def get_token(token_node):
            return token_node.text.strip()

        if node.expr_name == 'group_token':
            if node.children[0].expr_name == '_token_':
                output.append(get_token(node.children[0]))
            else:
                node = node.children[0]
                group_token = get_token(node.children[0])
                output.append({"name": group_token, "children": []})

                read_peg(node.children[2], output[-1]['children'])
                for c in node.children[3].children:
                    read_peg(c.children[1], output[-1]['children'])
        elif node.expr_name == 'token':
            output.append(get_token(node.text))

    def keep_n_words(text: str, n: int):
        tokens = []
        for token in text.split(" "):
            for x in token.split("/"):
                tokens.append(x)
                if len(tokens) >= n:
                    break

            if len(tokens) >= n:
                break
        return " ".join(tokens)

    def tree2tag(group_node, tag: Tag):
        if group_node['name'] == 'header':
            ctag = Pix2CodeTag('nav', [], [])
            for x in group_node['children']:
                if x == 'btn-active':
                    atag = Pix2CodeTag('button', ['btn', 'btn-primary'],
                                       [keep_n_words(fake.name(), 1)])
                elif x == 'btn-inactive':
                    atag = Pix2CodeTag('button', ['btn', 'btn-secondary'],
                                       [keep_n_words(fake.name(), 1)])
                else:
                    raise NotImplementedError(f"Doesn't support type {x} yet")
                ctag.children.append(atag)

            tag.children.append(
                Pix2CodeTag('div', ['row'], [Pix2CodeTag('div', ['col-12'], [ctag])]))
        elif group_node['name'] == 'row':
            ctag = Pix2CodeTag('div', ['row'], [])
            for x in group_node['children']:
                assert isinstance(x, dict), f"{x} must be a group node"
                tree2tag(x, ctag)

            tag.children.append(ctag)
        elif group_node['name'] in {'single', 'double', 'quadruple'}:
            if group_node['name'] == 'single':
                class_name = 'col-12'
            elif group_node['name'] == 'double':
                class_name = 'col-6'
            elif group_node['name'] == 'quadruple':
                class_name = 'col-3'
            else:
                raise NotImplementedError(f"Doesn't support group node {group_node['name']} yet")

            ctag = Pix2CodeTag('div', ['grey-background'], [])
            for x in group_node['children']:
                if x == 'small-title':
                    ctag.children.append(Pix2CodeTag("h5", [], [keep_n_words(fake.job(), 3)]))
                elif x == 'text':
                    ctag.children.append(Pix2CodeTag("p", [], [keep_n_words(fake.text(), 7)]))
                elif x == 'btn-orange':
                    ctag.children.append(
                        Pix2CodeTag("button", ["btn", "btn-warning"],
                                    [keep_n_words(fake.company(), 2)]))
                elif x == 'btn-red':
                    ctag.children.append(
                        Pix2CodeTag("button", ["btn", "btn-danger"],
                                    [keep_n_words(fake.company(), 2)]))
                elif x == 'btn-green':
                    ctag.children.append(
                        Pix2CodeTag("button", ["btn", "btn-success"],
                                    [keep_n_words(fake.company(), 2)]))
                else:
                    raise NotImplementedError(f"Doesn't support type {x} yet")
            tag.children.append(Pix2CodeTag('div', [class_name], [ctag]))
        else:
            raise NotImplementedError(f"Doesn't support group node {group_node} yet")

    tags = []
    origin_dsls = []
    for file in sorted(pix2code_dir.iterdir()):
        if not file.name.endswith(".gui"):
            continue

        with open(file, 'r') as f:
            dsl = f.read().replace("\n", " ")
            origin_dsls.append(dsl)

        program = grammar.parse(dsl)
        tree = []
        for c in program.children:
            read_peg(c, tree)

        # uncomment for debug
        # print(json.dumps(tree, indent=4))
        tag = Pix2CodeTag('div', ['container-fluid'], [])
        for gn in tree:
            tree2tag(gn, tag)
        tag = Pix2CodeTag("html", [], [tag])
        assert tag.is_valid(), tag.to_body("\n")
        tags.append(tag)

        # # TODO: uncomment to debug
        # if file.stem == "0BA2A4B4-4193-4506-8818-31564225EF8B":
        #     with open(ROOT_DIR / "datasets/pix2code/test.html", "w") as f:
        #         shutil.copy(file, str(ROOT_DIR / "datasets/pix2code/test.txt"))
        #         shutil.copy(
        #             str(file).replace(".gui", ".png"),
        #             str(ROOT_DIR / "datasets/pix2code/test.origin.png"))
        #         f.write(tag.to_html())
        #         render_engine = RemoteRenderEngine.get_instance(tags[0].to_html(), 1024, 640)
        #         img = render_engine.render_page(tag)
        #         imageio.imwrite(str(ROOT_DIR / "datasets/pix2code/test.jpeg"), shrink_img(img, 0.5))
        #         break

    with open(str(ROOT_DIR / "datasets" / "pix2code" / "data.json"), "w") as f:
        ujson.dump([o.serialize() for o in tags], f)
                                          
    with open(str(ROOT_DIR / "datasets" / "pix2code" / "data.original.json"), "w") as f:
        ujson.dump(origin_dsls, f)


def make_dataset(dataset_name: str, full_page: bool=False):
    print("Make hdf5 file...")
    dpath = ROOT_DIR / f"datasets/{dataset_name}/data.json"
    if dataset_name == "pix2code":
        viewport_width = 1024
        viewport_height = 640
        TagClass = Pix2CodeTag
    elif dataset_name == "toy":
        viewport_width = 480
        viewport_height = 300
        TagClass = ToyTag
    else:
        raise Exception(f"Invalid dataset {dataset_name}")

    with open(dpath, "r") as f:
        tags = [TagClass.deserialize(e) for e in ujson.load(f)]

    render_engine = RemoteRenderEngine.get_instance(tags[0].to_html(), viewport_width,
                                                    viewport_height, full_page=full_page)

    print("\t+ Rendering pages...")
    results = render_engine.render_pages(tags)

    assert max(x.shape[1]
               for x in results) == viewport_width, f"Max width: {max(x.shape[1] for x in results)}"
    assert max(
        x.shape[0]
        for x in results) == viewport_height, f"Max height: {max(x.shape[0] for x in results)}"
    results = np.asarray(results)

    print("\t+ Dumping to hdf5...")
    with h5py.File(ROOT_DIR / f"datasets/{dataset_name}/data.hdf5", "w") as f:
        f.create_dataset("images", data=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('-d', '--dataset', type=str, choices=['toy', 'pix2code'])
    parser.add_argument(
        '-n', '--n_examples', type=int, default=2000, help="Number of examples to generate")
    parser.add_argument(
        '-r', '--render', default=False, action='store_true', help='render images for a dataset')
    parser.add_argument(
        '-f', '--full_page', default=False, action='store_true', help='whether we should render the whole page')
    args = parser.parse_args()

    if args.render:
        make_dataset(args.dataset, args.full_page)
    else:
        if args.dataset == 'toy':
            generate_toy_data(args.n_examples)
        elif args.dataset == "pix2code":
            make_pix2code(ROOT_DIR / "../all_data")
