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

    def forward(self, X, X_lengths: List[int]):
        """
        :param X:
        :param X_lengths:
        :return:
        """
        # reset LSTM hidden state
        batch_size, seq_len = X.shape
        h0 = self.init_hidden(X, batch_size)

        # embed the input
        X = self.word_embedding(X)

        # pack_padded_sequence so that LSTM can record which item doesn't need to calculate gradient
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        X, h0 = self.lstm(X, h0)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        return X, h0
