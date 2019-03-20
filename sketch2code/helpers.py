#!/usr/bin/python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import *


def read_file(fpath: Union[Path, str]):
    with open(fpath, 'r') as f:
        return f.read()
