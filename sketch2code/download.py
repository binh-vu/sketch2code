#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import h5py
import ujson
from pathlib import Path

import numpy
from faker import Faker
from parsimonious.grammar import Grammar

from sketch2code.config import ROOT_DIR
from sketch2code.data_model import Tag, Pix2CodeTag
from sketch2code.render_engine import RemoteRenderEngine
"""
Use to obtain the data and preprocess it to get to our desired format
"""


def download_file_from_google_drive(id, destination):
    """
    This code is partially borrowed from
    https://gist.github.com/charlesreid1/4f3d676b33b95fce83af08e4ec261822
    """
    import requests

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_pix2code(download_path: Path):
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
            ctag = Pix2CodeTag('ul', ['navbar-nav'], [])
            for x in group_node['children']:
                atag = Pix2CodeTag('a', ['nav-link'], [keep_n_words(fake.name(), 1)])
                if x == 'btn-active':
                    ctag.children.append(Pix2CodeTag('li', ['nav-item', 'active'], [atag]))
                elif x == 'btn-inactive':
                    ctag.children.append(Pix2CodeTag('li', ['nav-item'], [atag]))
                else:
                    raise NotImplementedError(f"Doesn't support type {x} yet")

            tag.children.append(Pix2CodeTag('nav', ['navbar', 'navbar-expand-sm', 'bg-light'], [ctag]))
        elif group_node['name'] == 'row':
            ctag = Pix2CodeTag('div', ['row'], [])
            for x in group_node['children']:
                assert isinstance(x, dict), f"{x} must be a group node"
                tree2tag(x, ctag)

            if tag.name == 'html':
                tag.children.append(Pix2CodeTag('div', ['container-fluid'], [ctag]))
            else:
                tag.children.append(ctag)
        elif group_node['name'] in {'single', 'double', 'quadruple'}:
            if group_node['name'] == 'single':
                class_name = 'col-sm-12'
            elif group_node['name'] == 'double':
                class_name = 'col-sm-6'
            elif group_node['name'] == 'quadruple':
                class_name = 'col-sm-3'
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
                        Pix2CodeTag("button", ["btn", "btn-warning"], [keep_n_words(fake.company(), 2)]))
                elif x == 'btn-red':
                    ctag.children.append(
                        Pix2CodeTag("button", ["btn", "btn-danger"], [keep_n_words(fake.company(), 2)]))
                elif x == 'btn-green':
                    ctag.children.append(
                        Pix2CodeTag("button", ["btn", "btn-success"], [keep_n_words(fake.company(), 2)]))
                else:
                    raise NotImplementedError(f"Doesn't support type {x} yet")
            tag.children.append(Pix2CodeTag('div', [class_name], [ctag]))
        else:
            raise NotImplementedError(f"Doesn't support group node {group_node} yet")

    download_path.mkdir(exist_ok=True, parents=True)
    examples = []
    for file in sorted((ROOT_DIR / "../all_data").iterdir()):
        if not file.name.endswith(".gui"):
            continue

        with open(file, 'r') as f:
            dsl = f.read().replace("\n", " ")

        program = grammar.parse(dsl)
        tree = []
        for c in program.children:
            read_peg(c, tree)

        # uncomment for debug
        # print(json.dumps(tree, indent=4))
        tag = Pix2CodeTag("html", [], [])
        for gn in tree:
            tree2tag(gn, tag)

        # print(tag.to_html(2))
        assert tag.is_valid()
        examples.append(tag.serialize())

    # save gui
    with open(str(download_path / "data.json"), "w") as f:
        ujson.dump(examples, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset for Sketch2Code')
    parser.add_argument('-d', '--datasets', metavar='N', type=str, nargs='+', choices=['pix2code'])
    args = parser.parse_args()
    datasets = {x.lower() for x in args.datasets}

    if 'pix2code' in datasets:
        download_pix2code(ROOT_DIR / "datasets" / "pix2code")
