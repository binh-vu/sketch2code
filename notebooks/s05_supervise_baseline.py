#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import shutil
from collections import defaultdict, namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np
from typing import *

from torch.nn.utils import clip_grad_norm_
from tqdm.autonotebook import tqdm

from notebooks.s01_toy_img_feature import tag2class
from sketch2code.helpers import inc_folder_no
from sketch2code.config import ROOT_DIR
from sketch2code.methods.dqn import conv2d_size_out, pool2d_size_out
from sketch2code.methods.lstm import prepare_batch_sent_lbls, prepare_batch_sents, padded_aware_nllloss, \
    LSTMNoEmbedding, prepare_batch_sentences
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

    vocab = {'<pad>': 0, '<program>': 1, '</program>': 2}
    for i, token in enumerate(_tokens, start=len(vocab)):
        vocab[token] = i

    assert len(vocab) == len(_tokens) + 3
    ivocab = {v: k for k, v in vocab.items()}
    return vocab, ivocab


Example = namedtuple('Example', ['img_idx', 'context_tokens', 'next_token', 'next_tokens'])

# class BLSuper5(nn.Module):
#
#     def __init__(self, img_h: int, img_w: int, img_repr_size: int, dsl_vocab: Dict[int, str], dsl_embedding_dim: int,
#                  dsl_hidden_dim: int):
#         super().__init__()
#         self.img_w = img_w
#         self.img_h = img_h
#         self.img_repr_size = img_repr_size
#         self.dsl_vocab = dsl_vocab
#         self.dsl_hidden_dim = dsl_hidden_dim
#         self.dsl_embedding_dim = dsl_embedding_dim
#
#         self.__build_model()
#
#     def __build_model(self):
#         # network compute features of target image
#         self.conv11 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
#         self.conv12 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
#         self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv21 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
#         self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv31 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
#         self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         imgsize = [self.img_h, self.img_w]
#         for i, s in enumerate(imgsize):
#             s = conv2d_size_out(s, 3, 1)
#             s = conv2d_size_out(s, 3, 1)
#             s = pool2d_size_out(s, 2, 2)
#             s = conv2d_size_out(s, 3, 1)
#             s = conv2d_size_out(s, 3, 1)
#             s = pool2d_size_out(s, 2, 2)
#             s = conv2d_size_out(s, 3, 1)
#             s = conv2d_size_out(s, 3, 1)
#             s = pool2d_size_out(s, 2, 2)
#             imgsize[i] = s
#
#         linear_input_size = imgsize[0] * imgsize[1] * 128
#
#         self.fc1 = nn.Linear(linear_input_size, 1024)
#         self.dropout1 = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(1024, self.img_repr_size)
#         self.dropout2 = nn.Dropout(0.3)
#
#         # lstm that compute the context of the program
#         self.word_embedding = nn.Embedding(
#             num_embeddings=len(self.dsl_vocab), embedding_dim=self.dsl_embedding_dim, padding_idx=0)
#         self.lstm = LSTMNoEmbedding(
#             input_size=self.dsl_embedding_dim + self.img_repr_size,
#             hidden_size=self.dsl_hidden_dim,
#             n_layers=1)
#         self.lstm2token = nn.Linear(self.dsl_hidden_dim, len(self.dsl_vocab))
#
#     def forward(self, x1, x2, x2_lens):
#         """
#             @x1: desired images (batch: N x C x W x H)
#             @x2: current programs (batch: N x T)
#             @x2_lens: lengths of current programs (N)
#         """
#         batch_size = x2_lens.shape[0]
#
#         # STEP 1: compute output from CNN
#         # X is N x C x W x H
#         x1 = F.selu(self.conv11(x1))
#         x1 = F.selu(self.bn1(self.conv12(x1)))
#         x1 = self.pool1(x1)
#
#         x1 = F.selu(self.conv21(x1))
#         x1 = F.selu(self.bn2(self.conv22(x1)))
#         x1 = self.pool2(x1)
#
#         x1 = F.selu(self.conv31(x1))
#         x1 = F.selu(self.bn3(self.conv32(x1)))
#         x1 = self.pool3(x1)
#
#         # flatten to N x (C * W * H)
#         x1 = x1.view(batch_size, -1)
#         x1 = F.relu(self.fc1(x1))
#         x1 = self.dropout1(x1)
#         x1 = F.relu(self.fc2(x1))
#         x1 = self.dropout2(x1)
#
#         # STEP 2: compute feature from lstm
#         x2 = self.word_embedding(x2)
#         x1 = x1.view(batch_size, 1, self.img_repr_size).expand(batch_size, x2.shape[1], x1.shape[1])
#         x2 = torch.cat([x1, x2], dim=2)
#         x2, (hn, cn) = self.lstm(x2, x2_lens, self.lstm.init_hidden(x2, batch_size))
#         # flatten from N x T x H to N x (T * H)
#         hn = hn.view(batch_size, -1)
#         # flatten from N x T x H to (N * T) x H
#         x2 = x2.contiguous().view(-1, self.dsl_hidden_dim)
#         nts = F.log_softmax(self.lstm2token(x2), dim=1)
#         return nts


def make_dataset(tags, dsl_vocab, min_length: int, seed: int = 10232):
    # dict of examples: len(program) => [(index of image, program, next_token)]
    examples = []

    for img_idx, tag in enumerate(tags):
        nbtn_class, row_class, _, _, _ = tag2class(tag)
        program = [dsl_vocab['<program>']]
        program += [dsl_vocab[x] for x in tag.linearize().str_tokens]
        program.append(dsl_vocab['</program>'])

        examples.append(Example(img_idx, program[:-1], program[-1], program[1:]))

    print("#examples", len(examples))
    return examples


def drop_examples(examples: List[Example], keep_prob: float, seed: int = 10232):
    """Dropout some examples to speed up evaluation"""
    random.seed(seed)
    examples = [e for e in examples if random.random() < keep_prob]
    return examples


def iter_batch(batch_size: int,
               images: torch.tensor,
               examples: List[Example],
               shuffle: bool = False,
               device=None):
    index = list(range(len(examples)))
    if shuffle:
        random.shuffle(index)

    for i in range(0, len(examples), batch_size):
        batch_examples = [examples[j] for j in index[i:i + batch_size]]

        bimgs = images[[e.img_idx for e in batch_examples]]
        bx, bnx, bxlen = prepare_batch_sentences([e.context_tokens for e in batch_examples],
                                                  [e.next_tokens for e in batch_examples],
                                                  device=device)

        yield (bimgs, bx, bnx, bxlen)


def eval(model, images, examples, device=None, batch_size=500):
    losses = []
    nts_accuracies = []
    n_examples = 0

    model.eval()
    with torch.no_grad():
        for bimgs, bx, bx, bxlen in iter_batch(
                batch_size, images, examples, device=device):
            bnts_pred = model(bimgs, bx, bxlen)

            loss, mask, btokens = padded_aware_nllloss(bnts_pred, bx)
            losses.append(loss.item() * btokens)

            accuracy = (
                (torch.argmax(bnts_pred, dim=1) == bx.view(-1)).float() * mask).sum().item()
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

    log_dir = inc_folder_no(ROOT_DIR / "runs" / "s05_exp" / "run_")
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
                for bimgs, bx, bx, bxlen in iter_batch(
                        batch_size, images, train_examples, shuffle=True, device=device):
                    pbar.update()
                    global_step += 1

                    model.zero_grad()
                    bnx_pred = model(bimgs, bx, bxlen)
                    loss, mask, btokens = padded_aware_nllloss(bnx_pred, bx)
                    accuracy = ((torch.argmax(bnx_pred, dim=1) == bx.view(-1)).float() *
                                mask).sum().item() / btokens

                    loss.backward()
                    clip_grad_norm_(model.parameters(), 1)  # prevent vanishing
                    optimizer.step()

                    writer.add_scalar('train/loss', loss, global_step)
                    writer.add_scalar('train/accuracy', accuracy, global_step)

                    pbar.set_postfix(loss=f"{loss:.5f}", accuracy=f"{accuracy:.5f}")

                if (i + 1) % eval_valid_freq == 0:
                    valid_res = eval(model, images, valid_examples, device, batch_size=500)
                    for k, v in valid_res.items():
                        writer.add_scalar(f'valid/{k}', v, global_step)

                    if valid_res['accuracy'] > best_performance:
                        best_performance = valid_res['accuracy']
                        torch.save({
                            "epoch": i,
                            "optimizer": optimizer.state_dict(),
                            "model": model.state_dict(),
                        }, log_dir + f"/model.bin")

                if (i + 1) % eval_test_freq == 0:
                    test_res = eval(model, images, test_examples, device, batch_size=500)
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
