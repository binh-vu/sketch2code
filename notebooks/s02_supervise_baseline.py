#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
from collections import defaultdict
import torch.nn as nn
import torch, numpy as np
from typing import *
from tqdm.autonotebook import tqdm
from sketch2code.helpers import inc_folder_no
from sketch2code.config import ROOT_DIR
from sketch2code.methods.lstm import prepare_batch_sent_lbls, prepare_batch_sents
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


Example = Tuple[int, List[int], int]


def make_dataset(tags, vocab, min_length: int, seed: int=10232):
    # dict of examples: len(program) => [(index of image, program, next_token)]
    examples = defaultdict(lambda: [])

    for img_idx, tag in enumerate(tags):
        program = [vocab[x] for x in tag.linearize().str_tokens]
        for i in range(min_length, len(program) - 1):
            examples[i].append((img_idx, program[:i], program[i]))

    # then we convert the dict to list of list of examples
    examples = [examples[k] for k in sorted([int(x) for x in examples.keys()])]
    random.seed(seed)

    for i in range(len(examples)):
        random.shuffle(examples[i])

    print("#examples", sum([len(x) for x in examples]))
    return examples


def drop_examples(examples: List[List[Example]], keep_prob: float, seed: int=10232):
    """Dropout some examples to speed up evaluation"""
    random.seed(seed)
    for i in range(len(examples)):
        examples[i] = [e for e in examples[i] if random.random() < keep_prob]
    return examples


def iter_batch(batch_size: int, images: torch.tensor, examples: List[List[Example]], shuffle: bool = False, device=None):
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
        bimgs = images[[idx for idx, _, _ in batch_examples]]
        by = torch.tensor([y for _, _, y in batch_examples], dtype=torch.long, device=device)
        bx, bxlen = prepare_batch_sents([prog for _, prog, _ in batch_examples], device=device)
        
        yield (bimgs, bx, by, bxlen)


def eval(model, loss_func, images, examples, device=None, batch_size=500):
    losses = []
    accuracies = []
    n_examples = 0

    batch_size = 500
    model.eval()
    with torch.no_grad():
        # for bx, by1, by2 in tqdm(
        #         iter_batch(batch_size, X, y1, y2, device=device), desc='eval', total=math.ceil(len(X) / batch_size)):
        for bimgs, bx, by, bxlen in iter_batch(batch_size, images, examples, device=device):
            bypred = model(bimgs, bx, bxlen)
            loss = loss_func(bypred, by)

            losses.append(loss.item())
            accuracies.append((torch.argmax(bypred, dim=1) == by).sum().item())
            n_examples += by.shape[0]

    model.train()
    return {
        "loss": sum(losses) / n_examples,
        "accuracy": sum(accuracies) / n_examples,
    }


def train(model: nn.Module, loss_func, scheduler, optimizer, images, datasets, n_epoches: int, batch_size: int, eval_valid_freq: int=1, eval_test_freq: int=3, device=None):
    histories = {'train': [], 'valid': [], 'test': []}
    
    log_dir = inc_folder_no(ROOT_DIR / "runs" / f"s02_exp_")
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    model.train()
    
    valid_res = {'loss': float('nan'), 'accuracy': float('nan')}
    test_res = {'loss': float('nan'), 'accuracy': float('nan')}
    best_performance = 0.0

    train_examples, valid_examples, test_examples = datasets
    
    try:
        with tqdm(
                range(n_epoches), desc='epoch') as epoches, tqdm(
                    total=math.ceil(sum([len(x) for x in train_examples]) / batch_size) * n_epoches, desc='training') as pbar:
            for i in epoches:
                scheduler.step()
                for bimgs, bx, by, bxlen in iter_batch(batch_size, images, train_examples, shuffle=True, device=device):
                    pbar.update()
                    global_step += 1

                    model.zero_grad()
                    bypred = model(bimgs, bx, bxlen)
                    accuracy = (torch.argmax(bypred, dim=1) == by).float().mean().item()
                    loss = loss_func(bypred, by)
                    loss.backward()
                    optimizer.step()

                    histories['train'].append((loss, accuracy))
                    writer.add_scalar('train/loss', loss, global_step)
                    writer.add_scalar('train/accuracy', accuracy, global_step)
                    pbar.set_postfix(train_loss=f"{loss:.5f}", train_accuracy=f"{accuracy:.5f}")
                
                if (i + 1) % eval_valid_freq == 0:
                    valid_res = eval(model, loss_func, images, valid_examples, device)
                    writer.add_scalar('valid/loss', valid_res['loss'], global_step)
                    writer.add_scalar('valid/accuracy', valid_res['accuracy'], global_step)
                    histories['valid'].append(valid_res)
                    
                    if valid_res['accuracy'] > best_performance:
                        best_performance = valid_res['accuracy']
                        torch.save(model, log_dir + f"/model.{i}.bin")

                if (i + 1) % eval_test_freq == 0:
                    test_res = eval(model, loss_func, images, test_examples, device)
                    writer.add_scalar('test/loss', test_res['loss'], global_step)
                    writer.add_scalar('test/accuracy', test_res['accuracy'], global_step)
                    histories['test'].append(test_res)

                epoches.set_postfix(
                    valid_l=f'{valid_res["loss"]:.5f}',
                    valid_a=f'{valid_res["accuracy"]:.5f}',
                    test_l=f'{test_res["loss"]:.5f}',
                    test_a=f'{test_res["accuracy"]:.5f}',
                )
    finally:
        writer.close()
    return histories
