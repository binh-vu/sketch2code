#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *


class Tag:
    # supported tags and its classes
    supported_tags = {
        "html": set([]),
        "nav": {"navbar", "navbar-expand-sm", "bg-light"},
        "a": {"nav-link"},
        "div": {"row", "col-sm-12", "col-sm-6", "col-sm-3", "container", "grey-background"},
        "p": set([]),
        "h4": set([]),
        "button": {"btn", "btn-danger", "btn-warning", "btn-success"},
        "li": {"nav-item", "active"},
        "ul": {"navbar-nav"},
    }

    stylesheets = [
        "http://localhost:8080/css/main.css",
        "http://localhost:8080/css/bootstrap.min.css",
    ]

    def __init__(self, name: str, clazz: List[str], children: List[Union['Tag', str]]):
        self.name = name
        self.clazz = clazz
        self.children = children

    @staticmethod
    def deserialize(o: dict):
        return Tag(o['name'], o['class'],
                   [Tag.deserialize(v) if isinstance(v, dict) else v for v in o['children']])

    def serialize(self):
        return {
            "name": self.name,
            "class": self.clazz,
            "children": [x.serialize() if isinstance(x, Tag) else x for x in self.children]
        }

    def is_valid(self) -> bool:
        return self.name in Tag.supported_tags and all(
            c in Tag.supported_tags[self.name]
            for c in self.clazz) and all(c.is_valid() if isinstance(c, Tag) else True
                                         for c in self.children)

    def to_body(self):
        children = "\n".join(x.to_body() if isinstance(x, Tag) else x for x in self.children)

        if self.name == "html":
            return children
        return f"<{self.name} class=\"{' '.join(self.clazz)}\">{children}</{self.name}>"

    def to_html(self, indent: int = 2, continuous_indent: int = 2):
        space = " " * continuous_indent
        children = "\n".join(
            x.to_html(indent, continuous_indent + indent) if isinstance(x, Tag) else space +
            (" " * indent) + x for x in self.children)

        if self.name == "html":
            stylesheets = "\n".join(
                [f'{space}{space}<link rel="stylesheet" href="{x}" />' for x in self.stylesheets])
            return f"""<html>
{space}<head>
{stylesheets}
{space}</head>
{space}<body>
{children}
{space}</body>
</html>
"""

        return f"""{space}<{self.name} class=\"{' '.join(self.clazz)}\">
{children}
{space}</{self.name}>"""
