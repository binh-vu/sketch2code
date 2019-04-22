#!/usr/bin/python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import *

import math
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from sketch2code.config import ROOT_DIR
from sketch2code.data_model import Tag
from sketch2code.helpers import inc_folder_no
from sketch2code.methods.cnn import conv2d_size_out, pool2d_size_out
from tensorboardX import SummaryWriter
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

    return torch.tensor(
        X, dtype=torch.float32), torch.tensor(
            nbtn_y, dtype=torch.long), torch.tensor(
                row_y, dtype=torch.long)


def iter_batch(batch_size: int, X, y1, y2, shuffle: bool = False, device=None):
    index = list(range(len(X)))
    if shuffle:
        np.random.shuffle(index)

    for i in range(0, len(X), batch_size):
        batch_idx = index[i:i + batch_size]
        bx = X[batch_idx].to(device)
        by1 = y1[batch_idx].to(device)
        by2 = y2[batch_idx].to(device)

        yield (bx, by1, by2)


def eval(model, loss_func1, loss_func2, X, y1, y2, device=None):
    task1_losses = []
    task1_accuracies = []
    task2_losses = []
    task2_accuracies = []

    batch_size = 500
    model.eval()
    with torch.no_grad():
        # for bx, by1, by2 in tqdm(
        #         iter_batch(batch_size, X, y1, y2, device=device), desc='eval', total=math.ceil(len(X) / batch_size)):
        for bx, by1, by2 in iter_batch(batch_size, X, y1, y2, device=device):
            by1_pred = model.forward_task1(bx)
            loss1 = loss_func1(by1_pred, by1)

            by2_pred = model.forward_task2(bx)
            loss2 = loss_func2(by2_pred, by2)

            task1_losses.append(loss1.item())
            task2_losses.append(loss2.item())

            task1_accuracies.append((torch.argmax(by1_pred, dim=1) == by1).sum().item())
            task2_accuracies.append((torch.argmax(by2_pred, dim=1) == by2).sum().item())

    model.train()
    return {
        "task1_loss": np.mean(task1_losses),
        "task2_loss": np.mean(task2_losses),
        "task1_accuracies": sum(task1_accuracies) / len(X),
        "task2_accuracies": sum(task2_accuracies) / len(X)
    }


def train(model: nn.Module, loss_func1, loss_func2, scheduler, optimizer, datasets, n_epoches: int, batch_size: int, device=None):
    task1_histories = {'train': [], 'valid': [], 'test': []}
    task2_histories = {'train': [], 'valid': [], 'test': []}

    train_X, train_y1, train_y2 = datasets['train']
    valid_X, valid_y1, valid_y2 = datasets['valid']
    test_X, test_y1, test_y2 = datasets['test']

    writer = SummaryWriter(log_dir=inc_folder_no(ROOT_DIR / "runs" / f"s01_exp_"))
    global_step = 0

    try:
        with tqdm(
                range(n_epoches), desc='epoch') as epoches, tqdm(
                    total=math.ceil(len(train_X) / batch_size) * n_epoches, desc='training') as pbar:
            for i in epoches:
                scheduler.step()
                for bx, by1, by2 in iter_batch(batch_size, train_X, train_y1, train_y2, shuffle=True, device=device):
                    pbar.update()

                    global_step += 1

                    model.zero_grad()
                    by1_pred = model.forward_task1(bx)
                    loss1 = loss_func1(by1_pred, by1)
                    loss1.backward()
                    optimizer.step()

                    model.zero_grad()
                    by2_pred = model.forward_task2(bx)
                    loss2 = loss_func2(by2_pred, by2)
                    loss2.backward()
                    optimizer.step()

                    task1_histories['train'].append(loss1)
                    task2_histories['train'].append(loss2)
                    writer.add_scalar('train/loss1', loss1, global_step)
                    writer.add_scalar('train/loss2', loss2, global_step)

                    pbar.set_postfix(train_loss1=f"{loss1:.5f}", train_loss2=f'{loss2:.5f}')

                valid_res = eval(model, loss_func1, loss_func2, valid_X, valid_y1, valid_y2, device)

                epoches.set_postfix()
                writer.add_scalar('valid/loss1', valid_res['task1_loss'], global_step)
                writer.add_scalar('valid/loss2', valid_res['task2_loss'], global_step)
                writer.add_scalar('valid/accuracies1', valid_res['task1_accuracies'], global_step)
                writer.add_scalar('valid/accuracies2', valid_res['task2_accuracies'], global_step)

                test_res = eval(model, loss_func1, loss_func2, test_X, test_y1, test_y2, device)

                writer.add_scalar('test/loss1', test_res['task1_loss'], global_step)
                writer.add_scalar('test/loss2', test_res['task2_loss'], global_step)
                writer.add_scalar('test/accuracies1', test_res['task1_accuracies'], global_step)
                writer.add_scalar('test/accuracies2', test_res['task2_accuracies'], global_step)

                task1_histories['valid'].append(valid_res)
                task2_histories['test'].append(test_res)

                epoches.set_postfix(
                    valid_l1=f'{valid_res["task1_loss"]:.5f}',
                    valid_l2=f'{valid_res["task2_loss"]:.5f}',
                    valid_a1=f'{valid_res["task1_accuracies"]:.5f}',
                    valid_a2=f'{valid_res["task2_accuracies"]:.5f}',
                    test_l1=f'{test_res["task1_loss"]:.5f}',
                    test_l2=f'{test_res["task2_loss"]:.5f}',
                    test_a1=f'{test_res["task1_accuracies"]:.5f}',
                    test_a2=f'{test_res["task2_accuracies"]:.5f}',
                )
    finally:
        writer.close()
    return task1_histories, task2_histories
