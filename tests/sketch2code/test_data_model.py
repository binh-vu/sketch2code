#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *

from sketch2code.data_model import LinearizedTag


def test_linearized_tag():
    tag = LinearizedTag.default()
    tag.add_open_tag("div")

    assert tag.to_body() == "<div></div>"

    tag.add_open_tag("p")
    tag.add_class("header")
    tag.add_class("bold")

    tag.add_text("vi sao")

    assert tag.to_body() == '<div><p class="header bold">vi sao</p></div>'
