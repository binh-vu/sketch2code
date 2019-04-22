import torch
import torch.nn as nn
import torch.nn.functional as F

from sketch2code.methods.dqn import conv2d_size_out, pool2d_size_out
from sketch2code.methods.lstm import LSTMNoEmbedding
from s10_supervise_baseline import EncoderV1, BLSuperV1


class DecoderV0(nn.Module):

    def __init__(self, img_repr_size: int, dsl_vocab, dsl_hidden_dim, dsl_embedding_dim, padding_idx: int=0):
        super().__init__()
        self.img_repr_size = img_repr_size
        self.dsl_vocab = dsl_vocab
        self.dsl_hidden_dim = dsl_hidden_dim
        self.dsl_embedding_dim = dsl_embedding_dim
        self.padding_idx = padding_idx

        self.__build_model()

    def __build_model(self):
        self.img2hidden = nn.Linear(self.img_repr_size, self.dsl_hidden_dim)
        self.word_embedding = nn.Embedding(
            num_embeddings=len(self.dsl_vocab), embedding_dim=self.dsl_embedding_dim, padding_idx=self.padding_idx)
        self.lstm = LSTMNoEmbedding(
            input_size=self.dsl_embedding_dim,
            hidden_size=self.dsl_hidden_dim,
            n_layers=1)
        self.lstm2token = nn.Linear(self.dsl_hidden_dim, len(self.dsl_vocab))

    def forward(self, x1, x2, x2_lens):
        """
        :param x1: output from encoder program (batch: N x E)
        :param x2: current programs (batch: N x T)
        :param x2_lens: lengths of current programs (N)
        :return:
        """
        batch_size = x2.shape[0]
        x2 = self.word_embedding(x2)
        
        c0 = self.lstm.init_hidden_cn(x2, batch_size)
        h0 = self.img2hidden(x1).view(*c0.shape)

        x2, (hn, cn) = self.lstm(x2, x2_lens, (h0, c0))
        # flatten from N x T x H to N x (T * H)
        hn = hn.view(batch_size, -1)
        # flatten from N x T x H to (N * T) x H
        x2 = x2.contiguous().view(-1, self.dsl_hidden_dim)
        nts = F.log_softmax(self.lstm2token(x2), dim=1)
        return nts.view(batch_size, -1, nts.shape[-1])  # N x T x H
