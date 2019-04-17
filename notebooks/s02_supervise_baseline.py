#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from typing import *


def make_toy_vocab():
    """
        Create a vocabulary of the toy dataset. Return the vocab: <token> => <index> and its reversed
    """
    _tokens = []
    for c in ["row", "col-12", "col-6", "col-4", "col-3", "container-fluid", "grey-background"]:
        _tokens.append(f'<div class="{c}">')
    _tokens.append(f"</div>")
    for c in ["btn-danger", "btn-warning", "btn-success"]:
        _tokens.append(f'<button class="btn {c}">')
    _tokens.append(f"</button>")

    vocab = {'<pad>': 0}
    for i, token in enumerate(_tokens, start=1):
        vocab[token] = i

    assert len(vocab) == len(_tokens) + 1
    ivocab = {v: k for k, v in vocab.items()}
    return vocab, ivocab


def make_dataset(imgs, tags, vocab, min_length: int):
    Ximgs = []
    X = []
    y = []

    for img, tag in zip(imgs, tags):
        program = [vocab[x] for x in tag.linearize().str_tokens]
        for i in range(min_length, len(program) - 1):
            Ximgs.append(img)
            X.append(program[:i])
            y.append(program[i])

    print("#examples", len(X))
    return torch.tensor(Ximgs), torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
