import torch
import torch.nn as nn
import torch.nn.functional as F

from sketch2code.methods.cnn import conv2d_size_out, pool2d_size_out
from sketch2code.methods.lstm import LSTMNoEmbedding


class EncoderV1(nn.Module):
    
    def __init__(self, img_h: int, img_w: int, img_repr_size: int):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.img_repr_size = img_repr_size

        self.__build_model()

    def __build_model(self):
        # network compute features of target image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(16, momentum=0.9)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32, momentum=0.9)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        imgsize = [self.img_h, self.img_w]
        for i, s in enumerate(imgsize):
            s = conv2d_size_out(s, 7, 1)
            s = pool2d_size_out(s, 3, 2)
            s = conv2d_size_out(s, 5, 1)
            s = pool2d_size_out(s, 3, 2)
            s = conv2d_size_out(s, 5, 1)
            s = pool2d_size_out(s, 3, 2)
            imgsize[i] = s

        linear_input_size = imgsize[0] * imgsize[1] * 64

        self.fc1 = nn.Linear(linear_input_size, self.img_repr_size)
        self.bn4 = nn.BatchNorm2d(self.img_repr_size, momentum=0.9)
        
    def compute_act_layers(self, x):
        """Compute activation at different layers of an image X (C x H x W)"""
        with torch.no_grad():
            x = x.view(1, *x.shape)
            x1 = F.selu(self.bn1(self.conv1(x)))
            x1pool = self.pool1(x1)
            x2 = F.selu(self.bn2(self.conv2(x1pool)))
            x2pool = self.pool2(x2)
            x3 = F.selu(self.bn3(self.conv3(x2pool)))
            x3pool = self.pool3(x3)

            return {
                'layer1': x1[0],
                'layer1pool': x1pool[0],
                'layer2': x2[0],
                'layer2pool': x2pool[0],
                'layer3': x3[0],
                'layer3pool': x3pool[0],
            }

    def forward(self, x1):
        """
        :param x1: desired iamges (batch: N x C x H x W)
        :return:
        """
        x1 = self.pool1(F.selu(self.bn1(self.conv1(x1))))
        x1 = self.pool2(F.selu(self.bn2(self.conv2(x1))))
        x1 = self.pool3(F.selu(self.bn3(self.conv3(x1))))

        # flatten to N x (C * W * H)
        x1 = x1.view(x1.shape[0], -1)
        x1 = F.relu(self.fc1(x1))

        return x1


class DecoderV1(nn.Module):

    def __init__(self, img_repr_size: int, dsl_vocab, dsl_hidden_dim, dsl_embedding_dim, padding_idx: int=0):
        super().__init__()
        self.img_repr_size = img_repr_size
        self.dsl_vocab = dsl_vocab
        self.dsl_hidden_dim = dsl_hidden_dim
        self.dsl_embedding_dim = dsl_embedding_dim
        self.padding_idx = padding_idx

        self.__build_model()

    def __build_model(self):
        self.word_embedding = nn.Embedding(
            num_embeddings=len(self.dsl_vocab), embedding_dim=self.dsl_embedding_dim, padding_idx=self.padding_idx)
        self.lstm = LSTMNoEmbedding(
            input_size=self.dsl_embedding_dim + self.img_repr_size,
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
        x1 = x1.view(batch_size, 1, self.img_repr_size).expand(batch_size, x2.shape[1], x1.shape[1])
        x2 = torch.cat([x1, x2], dim=2)
        x2, (hn, cn) = self.lstm(x2, x2_lens, self.lstm.init_hidden(x2, batch_size))
        # flatten from N x T x H to (N * T) x H
        x2 = x2.contiguous().view(-1, self.dsl_hidden_dim)
        nts = F.log_softmax(self.lstm2token(x2), dim=1)
        return nts.view(batch_size, -1, nts.shape[-1])  # N x T x H


class BLSuperV1(nn.Module):
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x1, x2, x2lens):
        x1 = self.encoder(x1)
        return self.decoder(x1, x2, x2lens)
