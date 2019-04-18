#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
from collections import defaultdict, namedtuple
import torch.nn as nn
import torch, numpy as np
from typing import *
from tqdm.autonotebook import tqdm

from notebooks.s01_toy_img_feature import tag2class
from sketch2code.helpers import inc_folder_no
from sketch2code.config import ROOT_DIR
from sketch2code.methods.lstm import prepare_batch_sent_lbls, prepare_batch_sents, padded_aware_nllloss
from tensorboardX import SummaryWriter
import math


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


Example = namedtuple('Example', ['img_idx', 'context_tokens', 'next_token', 'next_tokens', 'nbtn_class', 'row_class'])


def make_dataset(tags, dsl_vocab, img_vocabs, min_length: int, seed: int = 10232):
    # dict of examples: len(program) => [(index of image, program, next_token)]
    examples = defaultdict(lambda: [])

    for img_idx, tag in enumerate(tags):
        nbtn_class, row_class, _, _, _ = tag2class(tag)
        program = [dsl_vocab[x] for x in tag.linearize().str_tokens]
        nbtn_class = img_vocabs['nbtn_classes'][nbtn_class]
        row_class = img_vocabs['row_classes'][row_class]

        for i in range(min_length, len(program) - 1):
            examples[i].append(Example(img_idx, program[:i], program[i], program[1:i + 1], nbtn_class, row_class))

    # then we convert the dict to list of list of examples
    examples = [examples[k] for k in sorted([int(x) for x in examples.keys()])]
    random.seed(seed)

    for i in range(len(examples)):
        random.shuffle(examples[i])

    print("#examples", sum([len(x) for x in examples]))
    return examples


def drop_examples(examples: List[List[Example]], keep_prob: float, seed: int = 10232):
    """Dropout some examples to speed up evaluation"""
    random.seed(seed)
    for i in range(len(examples)):
        examples[i] = [e for e in examples[i] if random.random() < keep_prob]
    return examples


def iter_batch(batch_size: int, images: torch.tensor, examples: List[List[Example]], shuffle: bool = False,
               device=None):
    if shuffle:
        for i in range(len(examples)):
            random.shuffle(examples[i])

    pivot = 0
    internal_pivot = 0
    n_examples = len(examples)

    while pivot < n_examples:
        batch_examples = examples[pivot][internal_pivot:internal_pivot + batch_size]
        while len(batch_examples) < batch_size:
            pivot += 1
            if pivot >= n_examples:
                break
            internal_pivot = batch_size - len(batch_examples)
            batch_examples += examples[pivot][:internal_pivot]

        if len(batch_examples) == 0:
            break

        internal_pivot += batch_size
        bimgs = images[[e.img_idx for e in batch_examples]]
        bx, bnx, bxlen = prepare_batch_sent_lbls([e.context_tokens for e in batch_examples],
                                                 [e.next_tokens for e in batch_examples],
                                                 device=device)

        bnt = torch.tensor([e.next_token for e in batch_examples], dtype=torch.long, device=device)
        bnbtn = torch.tensor([e.nbtn_class for e in batch_examples], dtype=torch.long, device=device)
        bnrow = torch.tensor([e.row_class for e in batch_examples], dtype=torch.long, device=device)

        yield (bimgs, bx, bnx, bxlen, bnt, bnbtn, bnrow)


def eval(model, images, examples, device=None, batch_size=500):
    losses = []
    sublosses = defaultdict(lambda: [])

    nt_accuracies = []
    nbtn_accuracies = []
    nrow_accuracies = []

    n_examples = 0

    batch_size = 500
    model.eval()
    with torch.no_grad():
        # for bx, by1, by2 in tqdm(
        #         iter_batch(batch_size, X, y1, y2, device=device), desc='eval', total=math.ceil(len(X) / batch_size)):
        for bimgs, bx, bnx, bxlen, bnt, bnbtn, bnrow in iter_batch(batch_size, images, examples, device=device):
            bnt_pred, bnts_pred, bnbtn_pred, bnrow_pred = model(bimgs, bx, bxlen)

            # loss, mask, btokens = padded_aware_nllloss(bypred, by0)
            loss, subloss = model.loss_func(bnts_pred, bnx, bnt_pred, bnt, bnbtn_pred, bnbtn, bnrow_pred, bnrow)
            losses.append(loss.item() * bnt.shape[0])
            for j, l in enumerate(subloss):
                sublosses[f'loss_{j}'].append(l.item() * bnt.shape[0])

            nt_accuracies.append((torch.argmax(bnt_pred, dim=1) == bnt).sum().item())
            nbtn_accuracies.append((torch.argmax(bnbtn_pred, dim=1) == bnbtn).sum().item())
            nrow_accuracies.append((torch.argmax(bnrow_pred, dim=1) == bnrow).sum().item())

            n_examples += bnt.shape[0]

    model.train()
    res = {
        'loss': sum(losses) / n_examples,
        "nt_accuracy": sum(nt_accuracies) / n_examples,
        "nbtn_accuracy": sum(nbtn_accuracies) / n_examples,
        "nrow_accuracy": sum(nrow_accuracies) / n_examples,
    }

    for k, v in sublosses.items():
        res[k] = sum(v) / n_examples

    return res


def train(model: nn.Module,
          scheduler,
          optimizer,
          images,
          datasets,
          n_epoches: int,
          batch_size: int,
          eval_valid_freq: int = 1,
          eval_test_freq: int = 3,
          device=None):

    log_dir = inc_folder_no(ROOT_DIR / "runs" / "s02_exp" / "run_")
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    model.train()

    valid_res = {'loss': float('nan'), 'nt_accuracy': float('nan'), 'nbtn_accuracy': float('nan'), 'nrow_accuracy': float('nan')}
    test_res = {'loss': float('nan'), 'nt_accuracy': float('nan'), 'nbtn_accuracy': float('nan'), 'nrow_accuracy': float('nan')}
    best_performance = 0.0

    train_examples, valid_examples, test_examples = datasets

    try:
        with tqdm(
                range(n_epoches), desc='epoch') as epoches, tqdm(
                    total=math.ceil(sum([len(x) for x in train_examples]) / batch_size) * n_epoches,
                    desc='training') as pbar:
            for i in epoches:
                scheduler.step()
                for bimgs, bx, bnx, bxlen, bnt, bnbtn, bnrow in iter_batch(
                        batch_size, images, train_examples, shuffle=True, device=device):
                    pbar.update()
                    global_step += 1

                    model.zero_grad()
                    bnt_pred, bnts_pred, bnbtn_pred, bnrow_pred = model(bimgs, bx, bxlen)

                    bnt_acc = (torch.argmax(bnt_pred, dim=1) == bnt).float().mean().item()
                    bnbtn_acc = (torch.argmax(bnbtn_pred, dim=1) == bnbtn).float().mean().item()
                    bnrow_acc = (torch.argmax(bnrow_pred, dim=1) == bnrow).float().mean().item()

                    loss, sublosses = model.loss_func(bnts_pred, bnx, bnt_pred, bnt, bnbtn_pred, bnbtn, bnrow_pred, bnrow)
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar('train/total_loss', loss, global_step)
                    for j, l in enumerate(sublosses):
                        writer.add_scalar(f'train/loss_{j}', l.item(), global_step)
                    writer.add_scalar('train/nt_accuracy', bnt_acc, global_step)
                    writer.add_scalar('train/nbtn_accuracy', bnbtn_acc, global_step)
                    writer.add_scalar('train/nrow_accuracy', bnrow_acc, global_step)

                    pbar.set_postfix(
                        loss=f"{loss:.5f}",
                        nt_accuracy=f"{bnt_acc:.5f}",
                        nbtn_accuracy=f"{bnbtn_acc:.5f}",
                        nrow_accuracy=f"{bnrow_acc:.5f}",
                        **{f"loss_{j}": f"{l.item():.5f}"
                           for j, l in enumerate(sublosses)})

                if (i + 1) % eval_valid_freq == 0:
                    valid_res = eval(model, images, valid_examples, device)
                    for k, v in valid_res.items():
                        writer.add_scalar(f'valid/{k}', v, global_step)

                    if valid_res['nt_accuracy'] > best_performance:
                        best_performance = valid_res['nt_accuracy']
                        torch.save(model, log_dir + f"/model.{i}.bin")

                if (i + 1) % eval_test_freq == 0:
                    test_res = eval(model, images, test_examples, device)
                    for k, v in test_res.items():
                        writer.add_scalar(f'test/{k}', v, global_step)

                epoches.set_postfix(
                    v_l=f'{valid_res["loss"]:.5f}',
                    v_nt_a=f'{valid_res["nt_accuracy"]:.5f}',
                    v_nbtn_a=f'{valid_res["nbtn_accuracy"]:.5f}',
                    v_nrow_a=f'{valid_res["nrow_accuracy"]:.5f}',
                    t_l=f'{test_res["loss"]:.5f}',
                    t_nt_a=f'{test_res["nt_accuracy"]:.5f}',
                    t_nbtn_a=f'{test_res["nbtn_accuracy"]:.5f}',
                    t_nrow_a=f'{test_res["nrow_accuracy"]:.5f}',
                )
    finally:
        writer.close()
