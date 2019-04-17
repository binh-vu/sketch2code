#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *
import torch
import torch.nn as nn
from torch.autograd import Variable
"""Basic LSTM building block"""


class LSTM(nn.Module):
    def __init__(self, vocab_size: int, padding_token_idx: int, embedding_dim: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_token_idx = padding_token_idx
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.word_embedding: nn.Embedding
        self.lstm: nn.LSTM

        self.__build_model()

    def __build_model(self):
        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=self.padding_token_idx)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)

    def init_hidden(self, X, batch_size: int):
        hidden_a = torch.randn(self.n_layers, batch_size, self.hidden_size, device=X.device)
        hidden_b = torch.randn(self.n_layers, batch_size, self.hidden_size, device=X.device)

        return Variable(hidden_a), Variable(hidden_b)

    def init_hidden_cn(self, X, batch_size: int):
        return Variable(torch.randn(self.n_layers, batch_size, self.hidden_size, device=X.device))

    def forward(self, X, X_lengths: List[int], h0):
        """
        :param X: batch sentences (N x T)
        :param X_lengths: lengths of each sentence in batch (N)
        :param h0: previous hidden state
        :return:
        """
        # embed the input
        X = self.word_embedding(X)

        # pack_padded_sequence so that LSTM can record which item doesn't need to calculate gradient
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        X, hn = self.lstm(X, h0)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        return X, hn


def padded_aware_nllloss(y_pred, y, pad_idx: int=0):
    # flatten all the labels and predictions
    y = y.view(-1)
    y_pred = y_pred.view(-1, y_pred.shape[-1])

    mask = (y != pad_idx).float()
    n_tokens = mask.sum().item()
    # ignore the padding by multiple with the mask
    y_pred = y_pred[range(y.shape[0]), y] * mask
    ce_loss = - torch.sum(y_pred) / n_tokens
    return ce_loss, mask, n_tokens


def prepare_batch_sents(sents: List[List[int]], pad_w: int = 0, device=None):
    assert pad_w == 0
    sents_lens = sorted([(i, len(s)) for i, s in enumerate(sents)], key=lambda x: x[1], reverse=True)
    padded_sents: torch.FloatTensor = torch.zeros((len(sents), sents_lens[0][1]),
                                                  dtype=torch.long,
                                                  device=device)

    for i, (j, nw) in enumerate(sents_lens):
        padded_sents[i, :nw] = torch.tensor(sents[j])

    return padded_sents, torch.tensor([nw for i, nw in sents_lens], device=device)


def prepare_batch_sent_lbls(sents: List[List[int]], sent_lbls: List[List[int]], pad_idx: int = 0, device=None):
    assert pad_idx == 0
    sents_lens = sorted([(i, len(s)) for i, s in enumerate(sents)], key=lambda x: x[1], reverse=True)

    padded_sents = torch.zeros((len(sents), sents_lens[0][1]), dtype=torch.long, device=device)
    padded_lbls = torch.zeros_like(padded_sents)

    for i, (j, nw) in enumerate(sents_lens):
        padded_sents[i, :nw] = torch.tensor(sents[j])
        padded_lbls[i, :nw] = torch.tensor(sent_lbls[j])

    return padded_sents, padded_lbls, torch.tensor([nw for i, nw in sents_lens], dtype=torch.long, device=device)
