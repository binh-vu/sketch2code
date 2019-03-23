#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy
from pathlib import Path
from typing import *

from sketch2code.config import ROOT_DIR
from sketch2code.helpers import read_file


class Tag:
    # supported tags and its classes
    supported_tags = {}
    css_files = []
    stylesheets = [read_file(fpath) for fpath in css_files]

    def __init__(self, name: str, cls: List[str], children: List[Union['Tag', str]]):
        self.name = name
        self.cls = cls
        self.children = children

    @classmethod
    def deserialize(cls, o: dict):
        return cls(o['name'], o['class'], [cls.deserialize(v) if isinstance(v, dict) else v for v in o['children']])

    def serialize(self):
        return {
            "name": self.name,
            "class": self.cls,
            "children": [x.serialize() if isinstance(x, Tag) else x for x in self.children]
        }

    def clone(self) -> 'Tag':
        return self.__class__(self.name, copy.copy(self.cls),
                              [c if isinstance(c, str) else c.clone for c in self.children])

    def is_valid(self) -> bool:
        return self.name in self.supported_tags and all(c in self.supported_tags[self.name] for c in self.cls) and all(
            c.is_valid() if isinstance(c, Tag) else True for c in self.children)

    def to_body(self):
        children = "\n".join(x.to_body() if isinstance(x, Tag) else x for x in self.children)

        if self.name == "html":
            return children
        return f"<{self.name} class=\"{' '.join(self.cls)}\">{children}</{self.name}>"

    def to_html(self, indent: int = 2, continuous_indent: int = 2):
        space = " " * continuous_indent
        children = "\n".join(
            x.to_html(indent, continuous_indent + indent) if isinstance(x, Tag) else space + (" " * indent) + x
            for x in self.children)

        if self.name == "html":
            # css_files = "\n".join([
            #     f'{space}{space}<link rel="stylesheet" href="{x.replace(str(ROOT_DIR), "http://localhost:8080")}" />'
            #     for x in self.css_files
            # ])
            stylesheets = "\n".join([f'<style>{x}</style>' for x in self.stylesheets])

            return f"""<html>
{space}<head>
{stylesheets}
{space}</head>
{space}<body>
{children}
{space}</body>
</html>
"""

        return f"""{space}<{self.name} class=\"{' '.join(self.cls)}\">
{children}
{space}</{self.name}>"""


class LinearizedTag:

    def __init__(self, tokens: List[str], opening_tags: List[int]) :
        self.tokens = tokens
        self.opening_tags = opening_tags

    def clone(self):
        return LinearizedTag(copy.copy(self.tokens), copy.copy(self.opening_tags))

    def add_open_tag(self, tag_name: str):
        self.opening_tags.append(len(self.tokens))
        self.tokens.append(tag_name)

    def add_close_tag(self) -> bool:
        if len(self.opening_tags) == 0:
            return False
        self.tokens[self.opening_tags[-1]].append()

    def to_body(self):
        return "".join(self.tokens)


class Pix2CodeTag(Tag):
    supported_tags = {
        "html": set([]),
        "nav": {"navbar", "navbar-expand-sm", "bg-light"},
        "a": {"nav-link"},
        "div": {"row", "col-sm-12", "col-sm-6", "col-sm-3", "container-fluid", "grey-background"},
        "p": set([]),
        "h5": set([]),
        "button": {"btn", "btn-danger", "btn-warning", "btn-success"},
        "li": {"nav-item", "active"},
        "ul": {"navbar-nav"},
    }
    css_files = [
        ROOT_DIR / "datasets/pix2code/css/main.css",
        ROOT_DIR / "datasets/pix2code/css/bootstrap.min.css",
    ]
    stylesheets = [read_file(fpath) for fpath in css_files]


class ToyTag(Tag):
    supported_tags = {
        "html": set([]),
        "div": {"row", "col-12", "col-6", "col-4", "col-3", "container-fluid", "grey-background"},
        "button": {"btn", "btn-danger", "btn-warning", "btn-success"},
    }
    css_files = [
        ROOT_DIR / "datasets/toy/css/main.css",
        ROOT_DIR / "datasets/toy/css/bootstrap.min.css",
    ]
    stylesheets = [read_file(fpath) for fpath in css_files]
