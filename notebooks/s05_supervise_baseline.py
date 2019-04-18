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

    vocab = {'<pad>': 0, '<begin>': 1, '<end>': 2}
    for i, token in enumerate(_tokens, start=len(vocab)):
        vocab[token] = i

    assert len(vocab) == len(_tokens) + 3
    ivocab = {v: k for k, v in vocab.items()}
    return vocab, ivocab


Example = namedtuple('Example', ['img_idx', 'context_tokens', 'next_token', 'next_tokens', 'nbtn_class', 'row_class'])


def make_dataset(tags, dsl_vocab, img_vocabs, min_length: int, seed: int = 10232):
    # dict of examples: len(program) => [(index of image, program, next_token)]
    examples = []

    for img_idx, tag in enumerate(tags):
        nbtn_class, row_class, _, _, _ = tag2class(tag)
        program = [dsl_vocab[x] for x in tag.linearize().str_tokens]
#         program.append(dsl_vocab['<end>'])
        
        nbtn_class = img_vocabs['nbtn_classes'][nbtn_class]
        row_class = img_vocabs['row_classes'][row_class]

        examples.append(Example(img_idx, program[:-1], program[-1], program[1:], nbtn_class, row_class))
        
    print("#examples", len(examples))
    return examples


def drop_examples(examples: List[Example], keep_prob: float, seed: int = 10232):
    """Dropout some examples to speed up evaluation"""
    random.seed(seed)
    examples = [e for e in examples if random.random() < keep_prob]
    return examples


def iter_batch(batch_size: int, images: torch.tensor, examples: List[Example], shuffle: bool = False,
               device=None):
    index = list(range(len(examples)))
    if shuffle:
        random.shuffle(index)

    for i in range(0, len(examples), batch_size):
        batch_examples = [examples[j] for j in index[i:i + batch_size]]
        
        bimgs = images[[e.img_idx for e in batch_examples]]
        bx, bnts, bxlen = prepare_batch_sent_lbls([e.context_tokens for e in batch_examples],
                                                 [e.next_tokens for e in batch_examples],
                                                 device=device)

        bnt = torch.tensor([e.next_token for e in batch_examples], dtype=torch.long, device=device)
        bnbtn = torch.tensor([e.nbtn_class for e in batch_examples], dtype=torch.long, device=device)
        bnrow = torch.tensor([e.row_class for e in batch_examples], dtype=torch.long, device=device)

        yield (bimgs, bx, bnts, bxlen, bnt, bnbtn, bnrow)
        

def eval(model, images, examples, device=None, batch_size=500):
    losses = []
    sublosses = defaultdict(lambda: [])

    nts_accuracies = []

    n_examples = 0

    batch_size = 500
    model.eval()
    with torch.no_grad():
        for bimgs, bx, bnts, bxlen, bnt, bnbtn, bnrow in iter_batch(batch_size, images, examples, device=device):
            bnts_pred = model(bimgs, bx, bxlen)

            loss, mask, btokens = padded_aware_nllloss(bnts_pred, bnts)
            losses.append(loss.item() * btokens)

            accuracy = ((torch.argmax(bnts_pred, dim=1) == bnts.view(-1)).float() * mask).sum().item()
            nts_accuracies.append(accuracy)
            n_examples += btokens

    model.train()
    res = {
        'loss': sum(losses) / n_examples,
        "accuracy": sum(nts_accuracies) / n_examples,
    }

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

    log_dir = inc_folder_no(ROOT_DIR / "runs" / "s04_exp" / "run_")
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    model.train()

    valid_res = {'loss': float('nan'), 'accuracy': float('nan')}
    test_res = {'loss': float('nan'), 'accuracy': float('nan')}
    best_performance = 0.0

    train_examples, valid_examples, test_examples = datasets

    try:
        with tqdm(
                range(n_epoches), desc='epoch--') as epoches, tqdm(
                    total=math.ceil(len(train_examples) / batch_size) * n_epoches,
                    desc='training') as pbar:
            for i in epoches:
                scheduler.step()
                for bimgs, bx, bnts, bxlen, bnt, bnbtn, bnrow in iter_batch(
                        batch_size, images, train_examples, shuffle=True, device=device):
                    pbar.update()
                    global_step += 1

                    model.zero_grad()
                    bnts_pred = model(bimgs, bx, bxlen)

                    loss, mask, btokens = padded_aware_nllloss(bnts_pred, bnts)
                    accuracy = ((torch.argmax(bnts_pred, dim=1) == bnts.view(-1)).float() * mask).sum().item() / btokens

                    loss.backward()
                    optimizer.step()

                    writer.add_scalar('train/loss', loss, global_step)
                    writer.add_scalar('train/accuracy', accuracy, global_step)

                    pbar.set_postfix(
                        loss=f"{loss:.5f}",
                        accuracy=f"{accuracy:.5f}")

                if (i + 1) % eval_valid_freq == 0:
                    valid_res = eval(model, images, valid_examples, device)
                    for k, v in valid_res.items():
                        writer.add_scalar(f'valid/{k}', v, global_step)

                    if valid_res['accuracy'] > best_performance:
                        best_performance = valid_res['accuracy']
                        torch.save(model, log_dir + f"/model.{i}.bin")

                if (i + 1) % eval_test_freq == 0:
                    test_res = eval(model, images, test_examples, device)
                    for k, v in test_res.items():
                        writer.add_scalar(f'test/{k}', v, global_step)

                epoches.set_postfix(
                    v_l=f'{valid_res["loss"]:.5f}',
                    v_a=f'{valid_res["accuracy"]:.5f}',
                    t_l=f'{test_res["loss"]:.5f}',
                    t_a=f'{test_res["accuracy"]:.5f}',
                )
    finally:
        writer.close()
