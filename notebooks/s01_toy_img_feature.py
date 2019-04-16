#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *

import math
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sketch2code.data_model import Tag
from sketch2code.methods.dqn import conv2d_size_out, pool2d_size_out
"""
Helps functions for the jupyter notebook
"""


def tag2class(tag: Tag):
    # number of buttons of different classes
    buttons = {"btn-danger": 0, "btn-warning": 0, "btn-success": 0}
    n_buttons = 0

    # number of columns on each row
    rows = []

    for token, classes in tag.linearize().tokens:
        if token == "<button>":
            for c in classes:
                if c in buttons:
                    buttons[c] += 1
            n_buttons += 1
        elif token == '<div>':
            if 'row' in classes:
                # we are encounter new row,
                rows.append(None)
            elif any(c.startswith("col-") for c in classes) and rows[-1] is None:
                col = next(c for c in classes if c.startswith('col-'))
                rows[-1] = col.replace('col-', '')

    assert all(r is not None for r in rows) and len(rows) <= 3, rows
    # btn_class_str = "-".join([str(buttons[btn]) for btn in ['btn-danger', 'btn-warning', 'btn-success']])
    n_btn_class_str = str(n_buttons)
    row_class_str = "-".join(rows)
    return n_btn_class_str, row_class_str, n_buttons, buttons, rows


def make_vocab(tags: List[Tag]):
    """Create a vocabulary that encodes class string to int number"""
    nbtn_class2id = {}
    row_class2id = {}

    for tag in tags:
        nbtn_class, row_class, _, _, _ = tag2class(tag)
        if nbtn_class not in nbtn_class2id:
            nbtn_class2id[nbtn_class] = len(nbtn_class2id)

        if row_class not in row_class2id:
            row_class2id[row_class] = len(row_class2id)

    return nbtn_class2id, row_class2id


def make_dataset(imgs: List[np.ndarray], tags: List[Tag], row_class2id: Dict[str, int], nbtn_class2id: Dict[str, int]):
    X = []
    nbtn_y = []
    row_y = []

    for img, tag in zip(imgs, tags):
        nbtn_class, row_class, _, _, _ = tag2class(tag)
        X.append(img)
        nbtn_y.append(nbtn_class2id[nbtn_class])
        row_y.append(row_class2id[row_class])

    return torch.tensor(X), torch.tensor(nbtn_y), torch.tensor(row_y)


def iter_batch(batch_size: int, X, y1, y2, shuffle: bool=False, device=None): 
    index = list(range(len(X)))
    if shuffle:
        np.random.shuffle(index)
        
    for i in range(0, len(X), batch_size):
        batch_idx = index[i:i+batch_size]
        bx = torch.stack([X[j] for j in batch_idx]).to(device)
        by1 = torch.tensor([y1[j] for j in batch_idx], dtype=torch.long, device=device)
        by2 = torch.tensor([y2[j] for j in batch_idx], dtype=torch.long, device=device)

        yield (bx, by1, by2)


def train(model: nn.Module, loss_func1, loss_func2, optimizer, X, y1, y2, n_epoches: int, batch_size: int):
    n_epoches = 2
    batch_size = 100
    histories_1 = {'train': [], 'val': [], 'test': []}
    histories_2 = {'train': [], 'val': [], 'test': []}

    for i in range(n_epoches):
        batches = tqdm(iter_batch(batch_size, X, y1, y2, shuffle=True, device=device),
                       total=math.ceil(len(X) / batch_size))
        for bx, by1, by2 in batches:
            model.zero_grad()
            by1_pred = model(bx, by1)
            loss1 = loss_func1(by1_pred, by1)
            loss1.backward()
            optimizer.step()
                
            model.zero_grad()
            by2_pred = model(bx, by2)
            loss2 = loss_func2(by2_pred, by2)
            loss2.backward()
            optimizer.step()
            
            histories_1['train'].append(loss1)
            histories_2['train'].append(loss2)
            
            batches.set_description(f'train_loss1 = {loss1:.5f} -- train_loss2 = {loss2:.5f}')
            batches.refresh()
        
        eval
        vloss, vacc = eval(model, valid_imgs, valid_X, valid_y, device)
        print("Epoch", i, 'valid', f'loss={vloss:.5f}', f'acc={vacc:.5f}', flush=True)
        tloss, tacc = eval(model, test_imgs, test_X, test_y, device)
        print("Epoch", i, 'test', f'loss={tloss:.5f}', f'acc={tacc:.5f}', flush=True)
        histories['val'].append((vloss, vacc))
        histories['test'].append((tloss, tacc))