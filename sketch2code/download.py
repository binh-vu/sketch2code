#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

from parsimonious.grammar import Grammar
from typing import *

from sketch2code.config import ROOT_DIR

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


def download_pix2code(download_path):
    """Download file from google drive, run pre-processing to generate correct data format"""
    grammar = Grammar(r"""
program = group_token+
group_token = (_token_ "{" group_token (","? group_token)* _ "}" _) / _token_
_token_ = _ token _
token = ~"[A-Z-0-9]+"i
_ = ~"[ \n]*"
""")

    def dsl2tree(node, output_tree: list):
        def get_token(token_node):
            return token_node.text.strip()

        if node.expr_name == 'group_token':
            if node.children[0].expr_name == '_token_':
                output_tree.append(node.children[0].text)
            else:
                node = node.children[0]
                group_token = get_token(node.children[0])

                output_tree.append({ "name": group_token, "children": [] })
                [c.expr_name for c in node.children if c.expr_name == '_token_' or c.expr_name == 'group_token']

        for child in node.children:


    for file in Path("/home/rook/workspace/CSCI559/Project/pix2code/datasets/web/all_data").iterdir():
        if not file.name.endswith(".gui"):
            continue

        with open(file, 'r') as f:
            dsl = f.read().replace("\n", " ")

        program = grammar.parse(dsl)

        print(program)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset for Sketch2Code')
    parser.add_argument('-d', '--datasets', metavar='N', type=str, nargs='+', choices=['pix2code'])

    args = parser.parse_args()
    datasets = {x.lower() for x in args.datasets}

    if 'pix2code' in datasets:
        download_pix2code(ROOT_DIR / "datasets" / "pix2code")